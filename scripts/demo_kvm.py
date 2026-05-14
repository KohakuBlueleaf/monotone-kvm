"""Demo: official KVM attention.

  (1) equivalence check -- when the BSWA window covers the whole sequence the
      compressed state is never used, so KVM is bit-exact with causal attention;
  (2) the block recurrence with a small window -- watch the state grow under
      each of the three budget schedules.

Run:  python scripts/demo_kvm.py
"""

import torch
import torch.nn.functional as F

from monotone_kvm import KVMAttention, KVMConfig


def main():
    torch.manual_seed(0)
    B, T, H, d = 2, 1024, 4, 32
    hidden = H * d
    x = torch.randn(B, T, hidden)

    # (1) equivalence: BSWA window large enough to cover everything.
    cfg_full = KVMConfig(
        hidden_size=hidden, num_heads=H, chunk_len=64, n_bswa_chunks=64, sink_len=1
    )
    kvm = KVMAttention(cfg_full)
    with torch.no_grad():
        y = kvm(x)
        q, k, v, _ = kvm.project_qkv(x)
        ref = F.scaled_dot_product_attention(
            q, k * kvm._front_temp(), v, is_causal=True
        )
        ref = kvm.c_proj(ref.transpose(1, 2).reshape(B, T, hidden))
    print(
        f"[equivalence] BSWA-covers-all vs causal attn: "
        f"max|diff| = {(y - ref).abs().max().item():.2e}"
    )

    # (2) the real recurrence: small window, watch the compressed state grow.
    schedules = [
        ("fixed", dict(state_budget_mode="fixed", state_min_len=64, n_max_d_chunks=1)),
        (
            "power_law",
            dict(
                state_budget_mode="power_law",
                state_growth_factor=6.0,
                state_growth_exponent=0.5,
                state_min_len=32,
            ),
        ),
        (
            "saturation",
            dict(
                state_budget_mode="saturation", state_saturation_n=256, state_min_len=32
            ),
        ),
    ]
    for name, kw in schedules:
        cfg = KVMConfig(
            hidden_size=hidden,
            num_heads=H,
            chunk_len=64,
            n_bswa_chunks=2,
            sink_len=1,
            **kw,
        )
        kvm = KVMAttention(cfg)
        y = kvm(x)
        y.square().mean().backward()
        gnorm = (
            sum(
                (p.grad.detach() ** 2).sum()
                for p in kvm.parameters()
                if p.grad is not None
            )
            .sqrt()
            .item()
        )
        print(
            f"[{name:10s}] y={tuple(y.shape)}  "
            f"state slots per chunk = {kvm._trace}  "
            f"final={kvm._trace[-1]}  grad_norm={gnorm:.3f}"
        )


if __name__ == "__main__":
    main()
