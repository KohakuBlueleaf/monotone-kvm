"""Shared helpers for the Triton kernel package."""


def _next_pow2(x):
    return 1 << (max(int(x), 1) - 1).bit_length()
