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
    LinearScheduler,
    LogBudgetScheduler,
    LogScheduler,
    PowerScheduler,
    SqrtScheduler,
    check_invariants,
    get_scheduler,
    simulate,
)
from .attention import (
    KVMAttention,
    KVMConfig,
    MonotoneKVMAttention,
    MonotoneKVMConfig,
    PlainAttention,
    PlainAttentionConfig,
    flex_forward,
)
from .triton import kvm_triton_forward, monotone_triton_forward
from .model import TinyLM, TinyLMConfig, build_attention

__all__ = [
    "BucketScheduler",
    "LogScheduler",
    "LogBudgetScheduler",
    "SqrtScheduler",
    "PowerScheduler",
    "LinearScheduler",
    "get_scheduler",
    "simulate",
    "check_invariants",
    "KVMAttention",
    "KVMConfig",
    "MonotoneKVMAttention",
    "MonotoneKVMConfig",
    "flex_forward",
    "monotone_triton_forward",
    "kvm_triton_forward",
    "PlainAttention",
    "PlainAttentionConfig",
    "TinyLM",
    "TinyLMConfig",
    "build_attention",
]
