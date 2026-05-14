"""A small Transformer language model with pluggable KVM / monotone attention.

`TinyLM` is a standard pre-norm Transformer decoder. The only non-standard part
is the attention layer, which is a block-recurrent KVM-style module
(`KVMAttention` or `MonotoneKVMAttention`). Those modules carry RoPE internally,
so the model itself adds no positional embedding. The chunked recurrence lives
entirely inside the attention `forward`, so from the model's point of view it is
just a causal attention layer.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    KVMAttention,
    KVMConfig,
    MonotoneKVMAttention,
    MonotoneKVMConfig,
    PlainAttention,
    PlainAttentionConfig,
)


@dataclass
class TinyLMConfig:
    vocab_size: int
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_expansion: int = 4
    tie_embeddings: bool = True

    # which attention: "plain", "kvm", or "monotone"
    attn: str = "monotone"
    # forward path: "auto" (Triton kernels when usable), "naive", "triton", "flex"
    attn_impl: str = "auto"

    # shared attention config
    chunk_len: int = 32
    n_bswa_chunks: int = 2
    sink_len: int = 1
    rope_partial_dim: int | None = None
    rope_theta: float = 10000.0
    use_merge_gate: bool = True
    use_head_temps: bool = True

    # monotone-only
    schedule: str = "log"
    schedule_kwargs: dict = field(default_factory=dict)
    use_logsize_bias: bool = True

    # kvm-only
    state_budget_mode: str = "fixed"
    state_min_len: int = 64
    state_growth_factor: float = 1.0
    state_growth_exponent: float = 0.5
    state_saturation_n: int = 1024
    n_max_d_chunks: int = 10000
    use_vlens: bool = True


def build_attention(cfg: TinyLMConfig) -> nn.Module:
    """Construct the attention layer named by `cfg.attn`."""
    if cfg.attn == "plain":
        return PlainAttention(
            PlainAttentionConfig(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                rope_partial_dim=cfg.rope_partial_dim,
                rope_theta=cfg.rope_theta,
            )
        )
    if cfg.attn == "kvm":
        return KVMAttention(
            KVMConfig(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                rope_partial_dim=cfg.rope_partial_dim,
                rope_theta=cfg.rope_theta,
                chunk_len=cfg.chunk_len,
                n_bswa_chunks=cfg.n_bswa_chunks,
                sink_len=cfg.sink_len,
                state_budget_mode=cfg.state_budget_mode,
                state_min_len=cfg.state_min_len,
                state_growth_factor=cfg.state_growth_factor,
                state_growth_exponent=cfg.state_growth_exponent,
                state_saturation_n=cfg.state_saturation_n,
                n_max_d_chunks=cfg.n_max_d_chunks,
                use_vlens=cfg.use_vlens,
                use_merge_gate=cfg.use_merge_gate,
                use_head_temps=cfg.use_head_temps,
            )
        )
    if cfg.attn == "monotone":
        return MonotoneKVMAttention(
            MonotoneKVMConfig(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                rope_partial_dim=cfg.rope_partial_dim,
                rope_theta=cfg.rope_theta,
                chunk_len=cfg.chunk_len,
                n_bswa_chunks=cfg.n_bswa_chunks,
                sink_len=cfg.sink_len,
                schedule=cfg.schedule,
                schedule_kwargs=dict(cfg.schedule_kwargs),
                use_merge_gate=cfg.use_merge_gate,
                use_head_temps=cfg.use_head_temps,
                use_logsize_bias=cfg.use_logsize_bias,
            )
        )
    raise ValueError(f"unknown attn {cfg.attn!r}; choices: 'plain', 'kvm', 'monotone'")


class MLP(nn.Module):
    def __init__(self, hidden: int, expansion: int):
        super().__init__()
        inner = hidden * expansion
        self.fc = nn.Linear(hidden, inner)
        self.proj = nn.Linear(inner, hidden)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(ln1 x); x + mlp(ln2 x)."""

    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = build_attention(cfg)
        self.attn_impl = cfg.attn_impl
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = MLP(cfg.hidden_size, cfg.mlp_expansion)

    def _attend(self, a):
        """Dispatch to the configured attention forward path. "auto" uses the
        Triton kernels when the input + config support them (each module's
        `forward_auto`), else falls back to the naive recurrence."""
        impl = self.attn_impl
        if impl == "auto":
            fn = getattr(self.attn, "forward_auto", None)
            return fn(a) if fn is not None else self.attn(a)
        if impl == "triton":
            return self.attn.forward_triton(a)
        if impl == "flex":
            return self.attn.forward_flex(a)
        return self.attn(a)  # "naive"

    def forward(self, x):
        x = x + self._attend(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """A small pre-norm Transformer LM. `cfg.attn` selects the attention type."""

    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.wte.weight.numel()  # lm_head shares it; count once
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Recomputes the prefix each step (no KV cache)
        -- fine for short demo generations."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
