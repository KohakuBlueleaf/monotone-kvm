"""Official Key-Value Means (KVM) attention -- minimal, faithful reproduction.

Paper: arXiv:2605.09877        Official repo: github.com/recursal/KVM-paper

KVM is plain softmax attention over a concatenation of two things:

    [ compressed STATE slots ] ++ [ recent raw tokens: Block Sliding Window ]

Block-recurrence: every `chunk_len` tokens, the chunk that drops out of the
BSWA window is folded into the state. Each overflow token is either

  * APPENDED  as a fresh state slot -- the `n_append` *least* redundant tokens
              (lowest max cosine similarity to the current state), or
  * MERGED    into its most-similar existing slot (argmax routing), with the
              key and value *summed* into that slot.

At read time state keys are LayerNorm'd (LN(sum) ~= mean direction) and state
values are renormalized to a stored radius -- hence "Key-Value *Means*". How
big the state is allowed to get is a fixed schedule B(ctx): fixed / power-law
(~sqrt) / saturating.

This keeps the full mechanism -- sink slots, partial RoPE, qk-norm, the
data-dependent merge gate, per-head temperatures, all three budget schedules --
but drops the training backbone, token-shift, value-residual and the
single-token KV-cache decode path. The chunked forward *is* the recurrence; it
is exactly what training / prefill runs. No custom kernels.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..helpers import apply_rope, build_rope, lower_right_causal_mask


@dataclass
class KVMConfig:
    hidden_size: int
    num_heads: int
    rope_partial_dim: int | None = None  # default: half the head dim
    rope_theta: float = 10000.0
    chunk_len: int = 64
    n_bswa_chunks: int = 2  # BSWA window = n_bswa_chunks * chunk_len
    sink_len: int = 1  # protected state slots, never a merge target
    # state-size schedule B(ctx):
    state_budget_mode: str = "fixed"  # "fixed" | "power_law" | "saturation"
    state_min_len: int = 64
    state_growth_factor: float = 1.0  # power_law: B = factor * ctx ** exponent
    state_growth_exponent: float = 0.5
    state_saturation_n: int = 1024  # saturation: B = sat_n * ctx / (sat_n + ctx)
    n_max_d_chunks: int = 10000  # hard cap: max_state_len = chunk_len * this
    use_vlens: bool = True  # True: renorm V to stored radius; False: plain mean
    use_merge_gate: bool = True  # data-dependent 1+ELU(xW) weighting before merge
    use_head_temps: bool = True  # learned per-head key temperatures


class KVMAttention(nn.Module):
    """Official KVM block-recurrent attention as a drop-in attention layer."""

    def __init__(self, cfg: KVMConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.num_heads
        assert cfg.hidden_size % H == 0
        d = cfg.hidden_size // H
        self.num_heads, self.d_head = H, d
        self.rope_partial_dim = cfg.rope_partial_dim or (d // 2)
        self.chunk_len = cfg.chunk_len
        self.bswa_len = cfg.n_bswa_chunks * cfg.chunk_len
        self.sink_len = cfg.sink_len

        self.c_q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        # the repo zero-inits c_proj so each layer starts as a no-op in the
        # residual stream; we keep a normal init here so the layer is non-trivial
        # when used standalone (the LM in model.py wraps it in a residual anyway).

        self.ln_q = nn.LayerNorm(d)  # qk-norm
        self.ln_k = nn.LayerNorm(d)
        self.ln_s_k = nn.LayerNorm(d)  # state-key norm (used at merge + read)

        if cfg.use_merge_gate:
            self.key_weighting = nn.Linear(cfg.hidden_size, H, bias=False)
        if cfg.use_head_temps:
            self.front_head_temp = nn.Parameter(torch.ones(H))
            self.state_head_temp = nn.Parameter(torch.ones(H))

        self._trace: list[int] = []  # state slot count per chunk (filled by forward)

    # -- temperatures -------------------------------------------------------
    def _front_temp(self):
        if self.cfg.use_head_temps:
            return self.front_head_temp.view(1, -1, 1, 1)
        return 1.0

    def _state_temp(self):
        if self.cfg.use_head_temps:
            return self.state_head_temp.view(1, -1, 1, 1)
        return 1.0

    # -- projections --------------------------------------------------------
    def project_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        H, d = self.num_heads, self.d_head
        q = self.ln_q(self.c_q(x).view(B, T, H, d)).transpose(1, 2)  # [B,H,T,d]
        k = self.ln_k(self.c_k(x).view(B, T, H, d)).transpose(1, 2)
        v = self.c_v(x).view(B, T, H, d).transpose(1, 2)
        cos, sin = build_rope(
            T, d, self.rope_partial_dim, self.cfg.rope_theta, x.device
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.cfg.use_merge_gate:
            gate = 1.0 + F.elu(self.key_weighting(x))  # [B,T,H]
            gate = gate.transpose(1, 2).unsqueeze(-1)  # [B,H,T,1]
        else:
            gate = torch.ones(B, H, T, 1, device=x.device, dtype=x.dtype)
        return q, k, v, gate

    # -- state-key prep: zero the RoPE channels, then LayerNorm -------------
    def _prepare_state_k(self, k_chunk: torch.Tensor) -> torch.Tensor:
        p = self.rope_partial_dim
        z = torch.cat([torch.zeros_like(k_chunk[..., :p]), k_chunk[..., p:]], dim=-1)
        return self.ln_s_k(z)

    # -- bswa window start for a given total length ------------------------
    def _bswa_begin(self, total_len: int) -> int:
        bswa_end = ((total_len + self.chunk_len - 1) // self.chunk_len) * self.chunk_len
        return max(0, bswa_end - self.bswa_len)

    # -- B(ctx): how many state slots are we allowed right now -------------
    def _desired_state_len(self, ctx_len: int, avail: int, cur: int) -> int:
        c = self.cfg
        if c.state_budget_mode == "fixed":
            target = c.state_min_len
        elif c.state_budget_mode == "power_law":
            target = math.floor(
                c.state_growth_factor * (ctx_len**c.state_growth_exponent)
            )
            target = max(target, c.state_min_len)
        elif c.state_budget_mode == "saturation":
            n = c.state_saturation_n
            target = math.floor(n * ctx_len / (n + ctx_len))
            target = max(target, c.state_min_len)
        else:
            raise ValueError(c.state_budget_mode)
        target = min(target, avail, self.chunk_len * c.n_max_d_chunks)
        return max(target, cur)  # state never shrinks

    # -- split overflow tokens into (append, merge) by redundancy ----------
    def _split_append_merge(self, ov_k_prep, n_append, s_k):
        B, H, L, _ = ov_k_prep.shape
        all_idx = torch.arange(L, device=ov_k_prep.device).view(1, 1, L).expand(B, H, L)
        if n_append <= 0:
            return all_idx[:, :, :0], all_idx
        if n_append >= L or s_k.size(2) == 0:
            return all_idx[:, :, :n_append], all_idx[:, :, n_append:]
        with torch.no_grad():
            k_ref = self.ln_s_k(s_k)  # [B,H,m,d]
            sim = torch.matmul(ov_k_prep, k_ref.transpose(-1, -2))  # [B,H,L,m]
            score = sim.max(dim=-1).values  # [B,H,L]
            order = score.argsort(dim=-1)  # ascending = novel first
            app = order[..., :n_append].sort(dim=-1).values
            mrg = order[..., n_append:].sort(dim=-1).values
        return app, mrg

    @staticmethod
    def _gather(x, idx):
        return x.gather(2, idx.unsqueeze(-1).expand(*idx.shape, x.size(-1)))

    # -- argmax-route each merge token into the most similar state slot ----
    def _merge_into_state(self, k_mrg, v_mrg, s_k, s_v, s_vlen, protected):
        s_k_norm = self.ln_s_k(s_k)
        logits = torch.matmul(k_mrg, s_k_norm.transpose(-1, -2))  # [B,H,n_mrg,m]
        if protected > 0:
            logits[..., :protected] = float("-inf")
        best = logits.argmax(dim=-1, keepdim=True)  # [B,H,n_mrg,1]
        onehot = torch.zeros_like(logits).scatter(-1, best, 1.0)  # [B,H,n_mrg,m]
        s_k = s_k + onehot.transpose(-1, -2) @ k_mrg  # sum keys into slot
        s_v = s_v + onehot.transpose(-1, -2) @ v_mrg  # sum values into slot
        if not self.cfg.use_vlens:
            # plain-mean path: track how many tokens landed in each slot.
            # (the official repo sums the wrong axis here -- a latent bug in an
            #  untested branch, since the shipped configs all use vlens=True.)
            s_vlen = s_vlen + onehot.sum(dim=-2).unsqueeze(-1)
        return s_k, s_v, s_vlen

    # -- fold one overflow chunk into the state ----------------------------
    def _update_state(self, s_k, s_v, s_vlen, ov_k, ov_v, gate, ctx_len, avail):
        if ov_k.size(2) == 0:
            return s_k, s_v, s_vlen
        ov_k_prep = self._prepare_state_k(ov_k)  # LN'd, rope-free
        if self.cfg.use_merge_gate:
            ov_k_g, ov_v_g = ov_k_prep * gate, ov_v * gate
        else:
            ov_k_g, ov_v_g = ov_k_prep, ov_v

        cur = s_k.size(2)
        desired = self._desired_state_len(ctx_len, avail, cur)
        n_append = min(max(desired - cur, 0), ov_k.size(2))

        if n_append > 0:
            app_idx, mrg_idx = self._split_append_merge(ov_k_prep, n_append, s_k)
            k_app, v_app = self._gather(ov_k_g, app_idx), self._gather(ov_v_g, app_idx)
            k_mrg, v_mrg = self._gather(ov_k_g, mrg_idx), self._gather(ov_v_g, mrg_idx)
            s_k = torch.cat([s_k, k_app], dim=2)
            s_v = torch.cat([s_v, v_app], dim=2)
            if self.cfg.use_vlens:
                vlen_app = v_app.norm(dim=-1, keepdim=True)
            else:
                vlen_app = torch.ones_like(v_app[..., :1])
            s_vlen = torch.cat([s_vlen, vlen_app], dim=2)
        else:
            k_mrg, v_mrg = ov_k_g, ov_v_g

        if k_mrg.size(2) > 0:
            protected = min(self.sink_len, s_k.size(2))
            assert s_k.size(2) > protected, "need >=1 non-sink slot to merge into"
            s_k, s_v, s_vlen = self._merge_into_state(
                k_mrg, v_mrg, s_k, s_v, s_vlen, protected
            )
        return s_k, s_v, s_vlen

    # -- one softmax over [ state | bswa ] ---------------------------------
    def _attend(self, q, bswa_k, bswa_v, s_k, s_v, s_vlen):
        s_k_norm = self.ln_s_k(s_k) * self._state_temp()
        k_star = torch.cat([s_k_norm, bswa_k * self._front_temp()], dim=2)
        if self.cfg.use_vlens:
            s_v_read = F.normalize(s_v, dim=-1) * s_vlen  # renorm to stored radius
        else:
            s_v_read = s_v / s_vlen  # plain mean
        v_star = torch.cat([s_v_read, bswa_v], dim=2)
        mask = lower_right_causal_mask(q.size(2), k_star.size(2), q.device)
        return F.scaled_dot_product_attention(q, k_star, v_star, attn_mask=mask)

    # -- forward = the chunked block recurrence ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v, gate = self.project_qkv(x)
        self._trace = []

        # warmup: while everything still fits the BSWA window, plain causal attn
        front = min(T, self.bswa_len)
        outs = [
            F.scaled_dot_product_attention(
                q[:, :, :front],
                k[:, :, :front] * self._front_temp(),
                v[:, :, :front],
                is_causal=True,
            )
        ]

        if T > front:
            # initial state = the first chunk, copied directly (one slot / token)
            init = min(T, self.chunk_len)
            s_k = self._prepare_state_k(k[:, :, :init])
            s_v = v[:, :, :init].clone()
            if self.cfg.use_vlens:
                s_vlen = v[:, :, :init].norm(dim=-1, keepdim=True)
            else:
                s_vlen = torch.ones_like(v[:, :, :init, :1])
            coverage = init
            self._trace.append(s_k.size(2))

            for qb in range(front, T, self.chunk_len):
                qe = min(T, qb + self.chunk_len)
                bswa_b = self._bswa_begin(qe)
                assert bswa_b == coverage, "state coverage drifted from BSWA window"
                outs.append(
                    self._attend(
                        q[:, :, qb:qe],
                        k[:, :, bswa_b:qe],
                        v[:, :, bswa_b:qe],
                        s_k,
                        s_v,
                        s_vlen,
                    )
                )
                nb = self._bswa_begin(min(T, qe + self.chunk_len))
                if nb > bswa_b:
                    s_k, s_v, s_vlen = self._update_state(
                        s_k,
                        s_v,
                        s_vlen,
                        k[:, :, bswa_b:nb],
                        v[:, :, bswa_b:nb],
                        gate[:, :, bswa_b:nb],
                        ctx_len=qe,
                        avail=nb,
                    )
                    coverage = nb
                self._trace.append(s_k.size(2))

        y = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, T, -1)
        return self.c_proj(y)

    # -- parallel prefill: PHASE 1 (merge recurrence) + PHASE 2, both Triton --
    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel prefill with both phases on Triton kernels; fully
        differentiable (the merge routing is frozen, exactly as in `forward`)."""
        from ..triton import kvm_triton_forward

        return kvm_triton_forward(self, x)

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        """The Triton KVM path needs CUDA, a chunk-aligned tail, and the
        vlens / merge-gate / head-temp features off (the kernel's restriction)."""
        T = x.shape[1]
        front = min(T, self.bswa_len)
        return (
            x.is_cuda
            and (T - front) % self.chunk_len == 0
            and not (
                self.cfg.use_vlens or self.cfg.use_merge_gate or self.cfg.use_head_temps
            )
        )

    def forward_auto(self, x: torch.Tensor) -> torch.Tensor:
        """The Triton path when usable, else the naive recurrence."""
        return self.forward_triton(x) if self._can_use_triton(x) else self.forward(x)
