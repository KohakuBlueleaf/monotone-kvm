"""Shared attention plumbing: partial rotary embeddings and the causal mask.

Used by both `kvm.py` and `monotone.py`. Faithful to the official KVM repo's
`RotaryEmbedding`: the first `partial_dim` channels rotate, the rest get
angular frequency 0 (cos=1, sin=0 -> identity), so they survive the merge of
tokens from different positions untouched.
"""

import torch


def build_rope(n_pos: int, full_dim: int, partial_dim: int, base: float, device):
    """Cos/sin tables of shape [n_pos, full_dim] for partial rotary embeddings."""
    af = (1.0 / base) ** torch.linspace(0.0, 1.0, partial_dim // 2, device=device)
    af = af.repeat_interleave(2)
    af = torch.cat(
        [af, af.new_zeros(full_dim - partial_dim)]
    )  # tail freq 0 -> identity
    pos = torch.arange(n_pos, device=device, dtype=torch.float32)
    theta = pos[:, None] * af[None, :]
    cos, sin = theta.cos(), theta.sin()
    sin[..., 1::2] *= -1.0
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings. x: [B, H, T, D]; cos/sin: [T, D]."""
    cos, sin = cos[None, None], sin[None, None]
    x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view_as(x)
    return cos * x + sin * x_flip


def lower_right_causal_mask(q_len: int, kv_len: int, device) -> torch.Tensor:
    """Bottom-right-aligned causal mask for kv = [ state ... | window ... ].

    State columns are fully visible to every query; the trailing `q_len`
    window columns are causal. Returns a bool tensor [q_len, kv_len].
    """
    off = kv_len - q_len
    q = torch.arange(q_len, device=device)[:, None]
    k = torch.arange(kv_len, device=device)[None, :]
    return k <= (q + off)
