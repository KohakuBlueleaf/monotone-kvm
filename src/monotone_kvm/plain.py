"""Plain full causal attention -- the reference baseline.

Standard O(T^2) softmax attention. It shares the projection style of the
KVM / monotone modules (qk-norm + partial RoPE) so a sweep against them
isolates the *compression* effect rather than incidental architecture
differences. No block sliding window, no compressed state -- every query
attends to the entire causal prefix.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import apply_rope, build_rope


@dataclass
class PlainAttentionConfig:
    hidden_size: int
    num_heads: int
    rope_partial_dim: int | None = None  # default: half the head dim
    rope_theta: float = 10000.0


class PlainAttention(nn.Module):
    """Standard causal multi-head attention with qk-norm and partial RoPE."""

    def __init__(self, cfg: PlainAttentionConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.num_heads
        assert cfg.hidden_size % H == 0
        d = cfg.hidden_size // H
        self.num_heads, self.d_head = H, d
        self.rope_partial_dim = cfg.rope_partial_dim or (d // 2)

        self.c_q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.ln_q = nn.LayerNorm(d)
        self.ln_k = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, -1))
