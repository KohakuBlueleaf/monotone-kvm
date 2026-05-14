"""KVM attention with a monotone deterministic bucket schedule.

This is KVM (arXiv:2605.09877) with one piece swapped out. KVM gives us the
*mechanism*: a Block Sliding Window of recent raw tokens + a compressed STATE,
both read in a single softmax pass, updated chunk-recurrently with no custom
kernels. We keep all of that. What we replace is the *merge policy*.

  official KVM  : data-dependent. A budget B(ctx) says how many state slots to
                  keep; overflow tokens are split append/merge by cosine
                  novelty and routed by argmax similarity.
  this module   : structural. Each overflow chunk becomes one bucket; a
                  deterministic `BucketScheduler` (see `scheduler.py`) decides
                  whether to fuse one adjacent equal-size pair of buckets.

Why this might be a better inductive bias -- and why it is also *more
efficient*:

  * the schedule is O(1) integer arithmetic, data-independent -> fully
    precomputable, no argmax routing, no cosine sims (official KVM cannot do
    this -- its routing depends on the actual K values);
  * every bucket is a clean contiguous interval, not an unstructured centroid;
  * bucket count is monotone non-decreasing and sizes are non-decreasing
    newest->oldest -- memory capacity grows smoothly instead of collapsing.

The recent-token resolution that KVM's append step protects is here provided
structurally by the BSWA window, so the state can be aggressively compressed.
The Key-Value *Means* readout is unchanged from KVM: state keys are
LayerNorm'd (LN(sum) ~= mean direction), state values are the bucket mean
(sum / token count), with an optional +log(bucket size) score bias.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import apply_rope, build_rope, lower_right_causal_mask
from .scheduler import BucketScheduler, get_scheduler


@dataclass
class MonotoneKVMConfig:
    hidden_size: int
    num_heads: int
    rope_partial_dim: int | None = None
    rope_theta: float = 10000.0
    chunk_len: int = 64
    n_bswa_chunks: int = 2  # BSWA window = n_bswa_chunks * chunk_len
    sink_len: int = 1  # permanent always-visible anchor slots
    schedule: str = "log"  # see scheduler.get_scheduler: log/sqrt/power/fixed/linear
    schedule_kwargs: dict = field(
        default_factory=dict
    )  # e.g. {"alpha": 1/3} or {"k": 16}
    use_merge_gate: bool = True  # data-dependent 1+ELU(xW) weighting before summing
    use_head_temps: bool = True  # learned per-head key temperatures
    use_logsize_bias: bool = True  # +log(bucket token count) score bias


class MonotoneKVMAttention(nn.Module):
    """KVM mechanism + a monotone deterministic merge schedule."""

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

        self._trace: list[int] = []  # state slot count per chunk
        self._size_trace: list[list[int]] = []  # bucket sizes (chunk units) per chunk

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

    def _prepare_state_k(self, k_chunk: torch.Tensor) -> torch.Tensor:
        # zero the RoPE channels (positions of merged tokens differ), then LN
        p = self.rope_partial_dim
        z = torch.cat([torch.zeros_like(k_chunk[..., :p]), k_chunk[..., p:]], dim=-1)
        return self.ln_s_k(z)

    def _bswa_begin(self, total_len: int) -> int:
        bswa_end = ((total_len + self.chunk_len - 1) // self.chunk_len) * self.chunk_len
        return max(0, bswa_end - self.bswa_len)

    # -- fold one overflow chunk into a bucket, then run the schedule -------
    def _add_bucket(self, s_k, s_v, s_size, t, ov_k, ov_v, gate):
        bk = self._prepare_state_k(ov_k)  # [B,H,L,d], rope-free, LN'd
        bv = ov_v
        if self.cfg.use_merge_gate:
            bk, bv = bk * gate, bv * gate
        bk = bk.sum(dim=2, keepdim=True)  # [B,H,1,d]  bucket key = sum
        bv = bv.sum(dim=2, keepdim=True)  # [B,H,1,d]  bucket value = sum
        s_k = torch.cat([bk, s_k], dim=2)  # prepend newest at index 0
        s_v = torch.cat([bv, s_v], dim=2)

        new_size, pair = self.scheduler.step(s_size, t)
        if pair is not None:
            i, j = pair  # j == i + 1
            s_k = torch.cat(
                [
                    s_k[:, :, :i],
                    s_k[:, :, i : i + 1] + s_k[:, :, j : j + 1],
                    s_k[:, :, j + 1 :],
                ],
                dim=2,
            )
            s_v = torch.cat(
                [
                    s_v[:, :, :i],
                    s_v[:, :, i : i + 1] + s_v[:, :, j : j + 1],
                    s_v[:, :, j + 1 :],
                ],
                dim=2,
            )
        return s_k, s_v, new_size

    # -- one softmax over [ sink | state buckets | bswa ] ------------------
    def _attend(self, q, bswa_k, bswa_v, sink_k, sink_v, s_k, s_v, s_size):
        # state value = bucket MEAN = sum / token_count  (the "Value Means")
        counts = torch.tensor(
            [self.chunk_len * s for s in s_size], device=q.device, dtype=s_v.dtype
        )  # [m]
        s_v_read = s_v / counts.view(1, 1, -1, 1)

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
            bias[m_sink : m_sink + m_state] = counts.log()
            attn_mask = torch.where(
                causal,
                bias.unsqueeze(0).expand(q.size(2), -1),
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
            )
        else:
            attn_mask = causal
        return F.scaled_dot_product_attention(q, k_star, v_star, attn_mask=attn_mask)

    # -- forward = the chunked block recurrence ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T need not be a multiple of chunk_len: a partial tail simply stays in
        # the BSWA window; every chunk that overflows into the state is full.
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

        if T > front:
            # sink = first `sink_len` tokens, kept permanently (rope-free, LN'd).
            # (they also fall inside bucket 0 below -- a negligible sink_len-token
            #  overlap; the sink is an always-visible anchor, the bucket a summary.)
            sink_k = self._prepare_state_k(k[:, :, : self.sink_len])
            sink_v = v[:, :, : self.sink_len].clone()

            # bucket 0 = the first chunk (it has already overflowed the window).
            # `t` counts overflow chunks; it is the schedule's timestep.
            empty = torch.zeros(
                B, self.num_heads, 0, self.d_head, device=x.device, dtype=x.dtype
            )
            t = 1
            s_k, s_v, s_size = self._add_bucket(
                empty,
                empty,
                [],
                t,
                k[:, :, : self.chunk_len],
                v[:, :, : self.chunk_len],
                gate[:, :, : self.chunk_len],
            )
            coverage = self.chunk_len
            self._trace.append(s_k.size(2))
            self._size_trace.append(list(s_size))

            for qb in range(front, T, self.chunk_len):
                qe = min(T, qb + self.chunk_len)
                bswa_b = self._bswa_begin(qe)
                assert bswa_b == coverage, "state coverage drifted from BSWA window"
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
                nb = self._bswa_begin(min(T, qe + self.chunk_len))
                if nb > bswa_b:
                    t += 1
                    s_k, s_v, s_size = self._add_bucket(
                        s_k,
                        s_v,
                        s_size,
                        t,
                        k[:, :, bswa_b:nb],
                        v[:, :, bswa_b:nb],
                        gate[:, :, bswa_b:nb],
                    )
                    coverage = nb
                self._trace.append(s_k.size(2))
                self._size_trace.append(list(s_size))

        y = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, T, -1)
        return self.c_proj(y)
