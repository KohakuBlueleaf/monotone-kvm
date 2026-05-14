"""KVM attention with a monotone deterministic bucket schedule.

This is KVM (arXiv:2605.09877) with exactly one piece swapped out. KVM gives us
the *mechanism* -- a Block Sliding Window of recent raw tokens + a compressed
STATE, both read in one softmax pass, updated recurrently. We keep ALL of it,
unchanged. We replace ONLY KVM's *routing decision*.

  official KVM : when a token exits the BSWA window, a data-dependent rule
                 (cosine-novelty rank + argmax similarity) decides whether it is
                 appended as a fresh state slot or merged into an existing one.
  monotone     : a data-independent `BucketScheduler` decides instead. Each
                 exiting *token* is a singleton bucket; the schedule fuses
                 adjacent equal-size buckets by a fixed integer rule.

Everything else is KVM's and unchanged -- in particular the per-token merge
*operation*: a bucket's key is `LN(sum of prepared keys)`, its value is the
token mean (`sum / token-count`), with a `+log(bucket size)` score bias. The
schedule decides *which contiguous run of tokens* forms a bucket; it never
touches *how* the tokens in a bucket combine.

The schedule -- like KVM's routing -- is **token-based**, applied to the tokens
that have left the BSWA window. `chunk_len` is ONLY the BSWA window's block
size; it is never the schedule's unit. A bucket of "size k" is k *tokens*.

Being data-independent, the schedule is fully precomputable -- which is what the
parallel forms (`forward_flex`, `forward_triton`) exploit. `forward` itself is
already parallel here: the bucket summaries of any contiguous token interval are
just prefix-sum differences, so there is no Python token loop.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..helpers import apply_rope, build_rope, lower_right_causal_mask
from ..scheduler import BucketScheduler, get_scheduler, intervals, simulate


@dataclass
class MonotoneKVMConfig:
    hidden_size: int
    num_heads: int
    rope_partial_dim: int | None = None
    rope_theta: float = 10000.0
    chunk_len: int = 64  # ONLY the BSWA window block size -- never the schedule unit
    n_bswa_chunks: int = 2  # BSWA window = n_bswa_chunks * chunk_len
    sink_len: int = 1  # permanent always-visible anchor slots
    schedule: str = "log"  # scheduler.get_scheduler: log/logbudget/sqrt/power/linear
    schedule_kwargs: dict = field(
        default_factory=dict
    )  # e.g. {"coeff": 2.0} or {"alpha": 1/3}
    use_merge_gate: bool = True  # data-dependent 1+ELU(xW) weighting before summing
    use_head_temps: bool = True  # learned per-head key temperatures
    use_logsize_bias: bool = True  # +log(bucket token count) score bias


class MonotoneKVMAttention(nn.Module):
    """KVM mechanism + a monotone deterministic merge schedule (token-based)."""

    def __init__(self, cfg: MonotoneKVMConfig):
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
        self.scheduler: BucketScheduler = get_scheduler(
            cfg.schedule, **cfg.schedule_kwargs
        )

        self.c_q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        self.ln_q = nn.LayerNorm(d)
        self.ln_k = nn.LayerNorm(d)
        self.ln_s_k = nn.LayerNorm(d)

        if cfg.use_merge_gate:
            self.key_weighting = nn.Linear(cfg.hidden_size, H, bias=False)
        if cfg.use_head_temps:
            self.front_head_temp = nn.Parameter(torch.ones(H))
            self.state_head_temp = nn.Parameter(torch.ones(H))

        self._trace: list[int] = []  # state bucket count per query chunk
        self._size_trace: list[list[int]] = []  # bucket sizes (token counts)

    def _front_temp(self):
        return (
            self.front_head_temp.view(1, -1, 1, 1) if self.cfg.use_head_temps else 1.0
        )

    def _state_temp(self):
        return (
            self.state_head_temp.view(1, -1, 1, 1) if self.cfg.use_head_temps else 1.0
        )

    def project_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        H, d = self.num_heads, self.d_head
        q = self.ln_q(self.c_q(x).view(B, T, H, d)).transpose(1, 2)
        k = self.ln_k(self.c_k(x).view(B, T, H, d)).transpose(1, 2)
        v = self.c_v(x).view(B, T, H, d).transpose(1, 2)
        cos, sin = build_rope(
            T, d, self.rope_partial_dim, self.cfg.rope_theta, x.device
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.cfg.use_merge_gate:
            gate = (1.0 + F.elu(self.key_weighting(x))).transpose(1, 2).unsqueeze(-1)
        else:
            gate = torch.ones(B, H, T, 1, device=x.device, dtype=x.dtype)
        return q, k, v, gate

    def _prepare_state_k(self, k_tokens: torch.Tensor) -> torch.Tensor:
        # zero the RoPE channels (merged tokens come from different positions),
        # then LN -- applied per token, exactly as KVM's _prepare_state_update_k.
        p = self.rope_partial_dim
        z = torch.cat([torch.zeros_like(k_tokens[..., :p]), k_tokens[..., p:]], dim=-1)
        return self.ln_s_k(z)

    def _bswa_begin(self, total_len: int) -> int:
        bswa_end = ((total_len + self.chunk_len - 1) // self.chunk_len) * self.chunk_len
        return max(0, bswa_end - self.bswa_len)

    def _schedule_trace(self, n_tokens: int, device) -> list[list[int]]:
        """Cached per-(n_tokens, device): schedule size partition after each of
        the first n_tokens tokens. trace[n-1] = bucket sizes covering [0, n)."""
        cache = self.__dict__.setdefault("_sched_traces", {})
        key = (n_tokens, str(device))
        if key not in cache:
            cache[key] = simulate(self.scheduler, n_tokens)
        return cache[key]

    # -- one softmax over [ sink | scheduled bucket summaries | bswa window ] --
    def _attend(self, q, bswa_k, bswa_v, sink_k, sink_v, s_k, s_v, s_size):
        # s_k[j]   = sum of prepared keys over bucket j's tokens
        # s_v[j]   = sum of (gated) values  over bucket j's tokens
        # s_size[j]= token count of bucket j  (the "size k" of a size-k bucket)
        counts = s_size.to(s_v.dtype)  # [m] token counts
        s_v_read = s_v / counts.view(1, 1, -1, 1)  # bucket value = token mean

        # state / sink keys: LN(sum) ~= mean direction  (the "Key Means")
        s_k_norm = self.ln_s_k(s_k) * self._state_temp()
        sink_k_norm = self.ln_s_k(sink_k) * self._state_temp()
        k_star = torch.cat([sink_k_norm, s_k_norm, bswa_k * self._front_temp()], dim=2)
        v_star = torch.cat([sink_v, s_v_read, bswa_v], dim=2)

        m_sink, m_state, kv_len = sink_k.size(2), s_k.size(2), k_star.size(2)
        causal = lower_right_causal_mask(q.size(2), kv_len, q.device)
        if self.cfg.use_logsize_bias:
            # +log(token count): one summary token stands in for many originals.
            bias = torch.zeros(kv_len, device=q.device, dtype=q.dtype)
            bias[m_sink : m_sink + m_state] = counts.log().to(q.dtype)
            attn_mask = torch.where(
                causal,
                bias.unsqueeze(0).expand(q.size(2), -1),
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
            )
        else:
            attn_mask = causal
        return F.scaled_dot_product_attention(q, k_star, v_star, attn_mask=attn_mask)

    # -- forward = the block recurrence, with the monotone token schedule ------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T need not be a multiple of chunk_len: a partial tail stays in the BSWA
        # window; the schedule only ever sees fully-exited tokens.
        B, T, _ = x.shape
        q, k, v, gate = self.project_qkv(x)
        self._trace, self._size_trace = [], []

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
        if T <= front:
            return self.c_proj(outs[0].transpose(1, 2).reshape(B, T, -1))

        # per-token prepared K / raw V -- exactly what each bucket sums. The
        # merge gate (data-dependent token weighting) multiplies in here; it is
        # part of the *operation*, not the schedule.
        Kp = self._prepare_state_k(k)
        Vp = v
        if self.cfg.use_merge_gate:
            Kp, Vp = Kp * gate, Vp * gate
        # cumsum over tokens: a bucket summary for interval [a, b) is just
        # cum[b] - cum[a]. The schedule's "fold one token + maybe merge a pair"
        # is, for any contiguous interval, exactly this prefix-sum difference --
        # so this is the recurrence, with no Python token loop.
        cumK = F.pad(Kp.cumsum(2), (0, 0, 1, 0))  # [B,H,T+1,d]
        cumV = F.pad(Vp.cumsum(2), (0, 0, 1, 0))

        # sink = first `sink_len` tokens, kept permanently (rope-free, LN'd,
        # ungated -- an always-visible anchor, not a schedule bucket).
        sink_k = self._prepare_state_k(k[:, :, : self.sink_len])
        sink_v = v[:, :, : self.sink_len]

        trace = self._schedule_trace(self._bswa_begin(T), x.device)
        for qb in range(front, T, self.chunk_len):
            qe = min(T, qb + self.chunk_len)
            bswa_b = self._bswa_begin(qe)  # tokens [0, bswa_b) have exited -> state
            # the monotone schedule's bucket partition of the exited tokens
            sizes = trace[bswa_b - 1]  # newest-first token counts
            ivs = intervals(sizes)  # [(a_tok, b_tok), ...] oldest-first
            a = torch.tensor([a for a, _ in ivs], device=x.device)
            b = torch.tensor([b for _, b in ivs], device=x.device)
            s_k = cumK[:, :, b] - cumK[:, :, a]  # [B,H,m,d] bucket key sums
            s_v = cumV[:, :, b] - cumV[:, :, a]
            s_size = b - a  # [m] token counts

            outs.append(
                self._attend(
                    q[:, :, qb:qe],
                    k[:, :, bswa_b:qe],
                    v[:, :, bswa_b:qe],
                    sink_k,
                    sink_v,
                    s_k,
                    s_v,
                    s_size,
                )
            )
            self._trace.append(len(sizes))
            self._size_trace.append(list(sizes))

        y = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, T, -1)
        return self.c_proj(y)

    # -- parallel FlexAttention prefill (same weights, no Python loop) ------
    def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel prefill path: numerically equivalent to `forward`, via a
        token dyadic-pyramid reduction + one flex_attention call. See
        `monotone_flex.py`."""
        from .monotone_flex import flex_forward

        return flex_forward(self, x)

    # -- parallel Triton prefill (PHASE 1 cumsum/gather + PHASE 2 kernel) ---
    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel prefill with both phases on Triton kernels; fully
        differentiable and supports the full feature set (merge gate, head
        temps). See the triton/ package."""
        from ..triton import monotone_triton_forward

        return monotone_triton_forward(self, x)

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        """The Triton monotone path needs CUDA and a chunk-aligned tail."""
        T = x.shape[1]
        front = min(T, self.bswa_len)
        return x.is_cuda and (T - front) % self.chunk_len == 0

    def forward_auto(self, x: torch.Tensor) -> torch.Tensor:
        """The Triton path when usable, else the recurrence."""
        return self.forward_triton(x) if self._can_use_triton(x) else self.forward(x)
