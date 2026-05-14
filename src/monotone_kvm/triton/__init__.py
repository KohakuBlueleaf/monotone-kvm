"""Triton kernels for KVM / monotone-KVM attention (PHASE 1 + PHASE 2).

Layout:
  common.py           -- shared helpers
  phase2.py           -- the shared chunked-attention kernels (fwd + bwd)
  monotone_phase1.py  -- monotone cumsum/gather kernels (fwd + bwd)
  kvm_phase1.py       -- KVM merge-recurrence kernels (fwd + bwd, incl. tiled)
  entry.py            -- forward entry points wiring PHASE 1 + PHASE 2
"""

from .entry import _triton_forward, kvm_triton_forward, monotone_triton_forward
from .kvm_phase1 import _kvm_budget_plan, kvm_merge_forward
from .monotone_phase1 import _monotone_plan, monotone_phase1
from .phase2 import chunked_attention_forward

__all__ = [
    "monotone_triton_forward",
    "kvm_triton_forward",
    "_triton_forward",
    "chunked_attention_forward",
    "monotone_phase1",
    "_monotone_plan",
    "kvm_merge_forward",
    "_kvm_budget_plan",
]
