"""monotone_kvm -- KVM attention and a monotone-bucket-schedule variant.

Two attention implementations sharing one mechanism (Block Sliding Window +
compressed state, read in a single softmax pass, updated chunk-recurrently):

  * `KVMAttention`          -- official Key-Value Means (arXiv:2605.09877):
                               data-dependent append/merge routing.
  * `MonotoneKVMAttention`  -- the same mechanism with the merge policy replaced
                               by a deterministic, data-independent
                               `BucketScheduler`.

Plus `TinyLM`, a small Transformer language model that can use either.
"""

from .scheduler import (
    BucketScheduler,
    FixedScheduler,
    LinearScheduler,
    LogScheduler,
    PowerScheduler,
    SqrtScheduler,
    check_invariants,
    get_scheduler,
    simulate,
)
from .kvm import KVMAttention, KVMConfig
from .monotone import MonotoneKVMAttention, MonotoneKVMConfig
from .plain import PlainAttention, PlainAttentionConfig
from .model import TinyLM, TinyLMConfig, build_attention

__all__ = [
    "BucketScheduler",
    "LogScheduler",
    "SqrtScheduler",
    "PowerScheduler",
    "FixedScheduler",
    "LinearScheduler",
    "get_scheduler",
    "simulate",
    "check_invariants",
    "KVMAttention",
    "KVMConfig",
    "MonotoneKVMAttention",
    "MonotoneKVMConfig",
    "PlainAttention",
    "PlainAttentionConfig",
    "TinyLM",
    "TinyLMConfig",
    "build_attention",
]
