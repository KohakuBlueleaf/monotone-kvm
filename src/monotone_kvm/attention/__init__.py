"""Attention layers: plain, KVM, and monotone-KVM (+ the FlexAttention prefill).

All three share one mechanism -- a Block Sliding Window of recent raw tokens
plus a compressed state, read in a single softmax pass. They differ in how the
state is built: KVM uses data-dependent append/merge routing; monotone-KVM uses
a deterministic bucket schedule.
"""

from .kvm import KVMAttention, KVMConfig
from .monotone import MonotoneKVMAttention, MonotoneKVMConfig
from .monotone_flex import flex_forward
from .plain import PlainAttention, PlainAttentionConfig

__all__ = [
    "KVMAttention",
    "KVMConfig",
    "MonotoneKVMAttention",
    "MonotoneKVMConfig",
    "flex_forward",
    "PlainAttention",
    "PlainAttentionConfig",
]
