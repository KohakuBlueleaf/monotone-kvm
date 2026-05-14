"""Triton forward entry points -- PHASE 1 + PHASE 2 wired together.

PyTorch does only the QKV projection, the warmup SDPA, and a little integer
bookkeeping; both phases are Triton kernels and the whole path is differentiable.
"""

import torch
import torch.nn.functional as F

from .kvm_phase1 import kvm_merge_forward
from .monotone_phase1 import monotone_phase1
from .phase2 import chunked_attention_forward


# ======================================================================
# forward entry points (PHASE 1 in PyTorch + PHASE 2 Triton kernel)
# ======================================================================
def _triton_forward(attn, x, phase1_fn):
    B, T, _ = x.shape
    H, D, cl = attn.num_heads, attn.d_head, attn.chunk_len
    q, k, v, gate = attn.project_qkv(x)
    front = min(T, attn.bswa_len)

    warm = F.scaled_dot_product_attention(
        q[:, :, :front],
        k[:, :, :front] * attn._front_temp(),
        v[:, :, :front],
        is_causal=True,
    )
    if T <= front:
        return attn.c_proj(warm.transpose(1, 2).reshape(B, T, -1))

    assert (
        T - front
    ) % cl == 0, "Triton forward needs (T - bswa_len) divisible by chunk_len"
    n_q = (T - front) // cl
    BH = B * H

    buck_k, buck_v, buck_bias = phase1_fn(attn, k, v, gate)
    q_chunks = q[:, :, front:].reshape(BH, n_q, cl, D)
    raw_k = (k * attn._front_temp()).reshape(BH, T, D)
    raw_v = v.reshape(BH, T, D)
    scale = D**-0.5

    o = chunked_attention_forward(
        q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, front, attn.bswa_len, scale
    )
    o = o.reshape(B, H, T - front, D)
    y = torch.cat([warm, o], dim=2).transpose(1, 2).reshape(B, T, -1)
    return attn.c_proj(y)


def monotone_triton_forward(attn, x):
    """Parallel forward for `MonotoneKVMAttention`: PHASE 1 = cumsum/gather,
    PHASE 2 = the Triton chunked-attention kernel."""
    return _triton_forward(attn, x, monotone_phase1)


def kvm_triton_forward(attn, x):
    """Forward for `KVMAttention` with *both* phases in Triton: PHASE 1 = the
    merge-recurrence kernel, PHASE 2 = the shared chunked-attention kernel.
    PyTorch only does the QKV projection and a bit of integer bookkeeping."""
    B, T, _ = x.shape
    H, D, cl = attn.num_heads, attn.d_head, attn.chunk_len
    q, k, v, gate = attn.project_qkv(x)
    front = min(T, attn.bswa_len)

    warm = F.scaled_dot_product_attention(
        q[:, :, :front],
        k[:, :, :front] * attn._front_temp(),
        v[:, :, :front],
        is_causal=True,
    )
    if T <= front:
        return attn.c_proj(warm.transpose(1, 2).reshape(B, T, -1))

    assert (
        T - front
    ) % cl == 0, "Triton forward needs (T - bswa_len) divisible by chunk_len"
    n_q = (T - front) // cl
    BH = B * H

    buck_k, buck_v, buck_bias = kvm_merge_forward(attn, k, v)  # PHASE 1 (Triton)
    q_chunks = q[:, :, front:].reshape(BH, n_q, cl, D)
    raw_k = (k * attn._front_temp()).reshape(BH, T, D)
    raw_v = v.reshape(BH, T, D)

    o = chunked_attention_forward(
        q_chunks,
        buck_k,
        buck_v,
        buck_bias,  # PHASE 2 (Triton)
        raw_k,
        raw_v,
        front,
        attn.bswa_len,
        D**-0.5,
    )
    o = o.reshape(B, H, T - front, D)
    y = torch.cat([warm, o], dim=2).transpose(1, 2).reshape(B, T, -1)
    return attn.c_proj(y)
