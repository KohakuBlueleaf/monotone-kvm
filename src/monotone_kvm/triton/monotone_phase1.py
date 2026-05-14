"""PHASE 1 for monotone-KVM -- fully Triton, differentiable.

The bucket schedule is data-independent and **token-based**: it partitions the
tokens that have exited the BSWA window into contiguous intervals. A bucket's
summary over interval ``[a, b)`` is therefore just a prefix-sum difference, so
PHASE 1 is a cumsum over TOKENS (carry = a [D] vector) + an independent
per-query-chunk gather, both with LayerNorms -- no [M,D] stateful recurrence,
hence no SRAM tiling needed. Forward + matching backward; grads flow to k / v,
the ln_s_k LayerNorm, the merge gate, and the per-head state temperature.

`chunk_len` here is ONLY the BSWA window block size (and a convenient tile width
for the token cumsum) -- it is never the schedule's unit. The cumsum runs over
all T tokens; the gather reads token-interval endpoints straight from the
schedule trace.
"""

import math

import torch
import triton
import triton.language as tl

from ..scheduler import intervals, simulate
from .common import _next_pow2


# ======================================================================
# PHASE 1 -- monotone: fully Triton. No matmuls -- it is LayerNorm + sum +
# prefix-sum + gather, so both kernels are reduction/elementwise (fp32 math,
# input-dtype loads/stores). Two kernels:
#   _monotone_csum_kernel  : grid (B*H,)   -- token-level prefix sum (tiled by CL)
#   _monotone_gather_kernel: grid (n_q,B*H)-- gather token-interval buckets +
#                                            readout transform
# ======================================================================
def _monotone_plan(attn, T, device):
    """Data-independent bucket index tables, cached per (T, device).

    The schedule is token-based: `simulate` runs over the tokens that have
    exited the window, and the per-query-chunk buckets are TOKEN intervals.
    """
    cache = attn.__dict__.setdefault("_triton_plans", {})
    key = (T, str(device))
    if key in cache:
        return cache[key]

    cl, SL = attn.chunk_len, attn.sink_len
    front = min(T, attn.bswa_len)
    n_q = (T - front) // cl
    # tokens [0, n_tok) have exited the window by the last query chunk
    n_tok = attn._bswa_begin(front + n_q * cl)
    trace = simulate(attn.scheduler, n_tok)  # token-based size partitions

    # per query chunk ci: the schedule's bucket partition of its exited tokens
    per_chunk = []  # list[list[(a_tok, b_tok)]]
    for ci in range(n_q):
        qe = front + (ci + 1) * cl
        bswa_b = attn._bswa_begin(qe)
        per_chunk.append(intervals(trace[bswa_b - 1]))
    max_b = max(len(ivs) for ivs in per_chunk)
    MAXB = _next_pow2(max_b)  # bucket arange bound (pow2 for tl.arange)
    M = max(16, _next_pow2(SL + max_b))  # full state width (single pow2 -- no waste)

    a_idx = torch.zeros(n_q, MAXB, dtype=torch.int32)
    b_idx = torch.zeros(n_q, MAXB, dtype=torch.int32)
    cnt = torch.ones(n_q, MAXB, dtype=torch.float32)  # token count per bucket
    bias = torch.full((n_q, M), float("-inf"), dtype=torch.float32)
    bias[:, :SL] = 0.0  # sink: always visible
    lb = attn.cfg.use_logsize_bias
    for ci, ivs in enumerate(per_chunk):
        for j, (a, b) in enumerate(ivs):  # TOKEN-unit intervals
            a_idx[ci, j], b_idx[ci, j] = a, b
            cnt[ci, j] = b - a
            bias[ci, SL + j] = math.log(b - a) if lb else 0.0  # +log|bucket|

    plan = dict(
        n_q=n_q,
        max_b=max_b,
        MAXB=MAXB,
        M=M,
        a_idx=a_idx.reshape(-1).to(device),
        b_idx=b_idx.reshape(-1).to(device),
        cnt=cnt.reshape(-1).to(device),
        bias=bias.to(device),
    )
    cache[key] = plan
    return plan


@triton.jit
def _monotone_csum_kernel(
    K,
    V,
    GATE,
    CUMK,
    CUMV,
    LN_W,
    LN_B,
    n_tiles,
    T: tl.constexpr,
    CL: tl.constexpr,
    D: tl.constexpr,
    RPD: tl.constexpr,
    EPS: tl.constexpr,
    USE_GATE: tl.constexpr,
):
    # token-level prefix sum, tiled CL tokens at a time. CUMK/CUMV: [BH, T+1, D],
    # CUMK[t+1] = sum_{j<=t} prepare_state_k(k[j]) (gated). Within a tile an
    # intra-tile cumsum + a running [D] carry gives every token's prefix.
    bh = tl.program_id(0)
    od = tl.arange(0, D)
    ocl = tl.arange(0, CL)
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    kbase = bh * T * D
    cbase = bh * (T + 1) * D
    gbase = bh * T

    tl.store(CUMK + cbase + od, tl.zeros([D], tl.float32))  # cum[0] = 0
    tl.store(CUMV + cbase + od, tl.zeros([D], tl.float32))

    carry_k = tl.zeros([D], tl.float32)
    carry_v = tl.zeros([D], tl.float32)
    for tile in range(0, n_tiles):
        row0 = tile * CL
        rows = row0 + ocl
        kc = tl.load(K + kbase + rows[:, None] * D + od[None, :])
        vc = tl.load(V + kbase + rows[:, None] * D + od[None, :])
        # prepare_state_k: zero RoPE channels, LayerNorm over D (fp32)
        kf = tl.where(od[None, :] < RPD, 0.0, kc.to(tl.float32))
        mu = tl.sum(kf, 1) / D
        kf = kf - mu[:, None]
        var = tl.sum(kf * kf, 1) / D
        pk = kf * tl.rsqrt(var + EPS)[:, None] * ln_w + ln_b  # [CL, D]
        pv = vc.to(tl.float32)
        if USE_GATE:
            g = tl.load(GATE + gbase + rows)[:, None]
            pk = pk * g
            pv = pv * g
        # intra-tile inclusive prefix sum, then add the running carry
        sum_k = tl.sum(pk, 0)
        sum_v = tl.sum(pv, 0)
        ck = tl.cumsum(pk, 0) + carry_k[None, :]  # ck[i] = CUMK[row0+i+1]
        cv = tl.cumsum(pv, 0) + carry_v[None, :]
        tl.store(CUMK + cbase + (row0 + 1 + ocl)[:, None] * D + od[None, :], ck)
        tl.store(CUMV + cbase + (row0 + 1 + ocl)[:, None] * D + od[None, :], cv)
        carry_k = carry_k + sum_k
        carry_v = carry_v + sum_v


@triton.jit
def _monotone_gather_kernel(
    CUMK,
    CUMV,
    A_IDX,
    B_IDX,
    CNT,
    K,
    V,
    LN_W,
    LN_B,
    STEMP,
    BK,
    BV,
    n_q,
    T: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    MAXB: tl.constexpr,
    SL: tl.constexpr,
    H: tl.constexpr,
    RPD: tl.constexpr,
    EPS: tl.constexpr,
    USE_TEMP: tl.constexpr,
):
    ci = tl.program_id(0)
    bh = tl.program_id(1)
    od = tl.arange(0, D)
    ob = tl.arange(0, MAXB)
    osl = tl.arange(0, SL)
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    cbase = bh * (T + 1) * D
    obase = (bh * n_q + ci) * M * D
    stemp = tl.load(STEMP + (bh % H)).to(tl.float32) if USE_TEMP else 1.0
    bmask = (SL + ob)[:, None] < M  # MAXB may exceed M - SL -> mask the stores

    # --- bucket slots [SL : SL+MAXB] : cumK[b] - cumK[a] over TOKEN intervals ---
    a = tl.load(A_IDX + ci * MAXB + ob)
    b = tl.load(B_IDX + ci * MAXB + ob)
    cnt = tl.load(CNT + ci * MAXB + ob)  # token count [MAXB]
    bk = tl.load(CUMK + cbase + b[:, None] * D + od[None, :]) - tl.load(
        CUMK + cbase + a[:, None] * D + od[None, :]
    )  # [MAXB,D] fp32
    bv = tl.load(CUMV + cbase + b[:, None] * D + od[None, :]) - tl.load(
        CUMV + cbase + a[:, None] * D + od[None, :]
    )
    mu = tl.sum(bk, 1) / D
    bkc = bk - mu[:, None]
    var = tl.sum(bkc * bkc, 1) / D
    bk = (bkc * tl.rsqrt(var + EPS)[:, None] * ln_w + ln_b) * stemp
    bv = bv / cnt[:, None]
    tl.store(
        BK + obase + (SL + ob)[:, None] * D + od[None, :],
        bk.to(BK.dtype.element_ty),
        mask=bmask,
    )
    tl.store(
        BV + obase + (SL + ob)[:, None] * D + od[None, :],
        bv.to(BV.dtype.element_ty),
        mask=bmask,
    )

    # --- sink slots [0 : SL] : prepare_state_k then readout LN ---
    kbase = bh * T * D
    sk = tl.load(K + kbase + osl[:, None] * D + od[None, :])
    sv = tl.load(V + kbase + osl[:, None] * D + od[None, :])
    skf = tl.where(od[None, :] < RPD, 0.0, sk.to(tl.float32))
    m1 = tl.sum(skf, 1) / D
    skf = skf - m1[:, None]
    v1 = tl.sum(skf * skf, 1) / D
    s1 = skf * tl.rsqrt(v1 + EPS)[:, None] * ln_w + ln_b  # prepare_state_k
    m2 = tl.sum(s1, 1) / D
    s1c = s1 - m2[:, None]
    v2 = tl.sum(s1c * s1c, 1) / D
    s1 = (s1c * tl.rsqrt(v2 + EPS)[:, None] * ln_w + ln_b) * stemp  # readout LN
    tl.store(BK + obase + osl[:, None] * D + od[None, :], s1.to(BK.dtype.element_ty))
    tl.store(BV + obase + osl[:, None] * D + od[None, :], sv.to(BV.dtype.element_ty))


# ----------------------------------------------------------------------
# monotone PHASE 1 backward -- mirror of the two forward kernels.
#   _monotone_gather_bwd_kernel : grid (n_q, B*H). For each (ci, bh): bucket
#       slots -> atomic-scatter into dCUMK/dCUMV at the TOKEN-interval endpoints
#       (d(CUMK[b]-CUMK[a]) through the readout LN); sink slots -> backprop
#       the double LN into dK[0:SL] and pass dV[0:SL] through.
#   _monotone_csum_bwd_kernel   : grid (B*H,). Suffix-sum dCUMK/dCUMV over
#       TOKENS (the transpose of the forward prefix sum) -- tiled CL at a time
#       with an intra-tile reverse cumsum + a running [D] carry -> dpk / dpv per
#       token, backprop prepare_state_k -> dK / dV.
# Both accumulate dLN_W/dLN_B (one atomic_add per program). The full KVM
# feature set is supported: the merge gate folds into the csum (it weights each
# token before the prefix sum) -> the csum bwd produces dGATE; the per-head
# state temperature folds into the gather readout -> the gather bwd produces
# dSTEMP. front_head_temp lives outside PHASE 1 (it scales raw_k / the warmup
# keys, plain PyTorch ops) so autograd already covers it. fp32 everywhere.
# ----------------------------------------------------------------------
@triton.jit
def _monotone_gather_bwd_kernel(
    DBK,
    DBV,
    CUMK,
    A_IDX,
    B_IDX,
    CNT,
    K,
    LN_W,
    LN_B,
    STEMP,
    DCUMK,
    DCUMV,
    DK,
    DV,
    DSTEMP,
    DLNW,
    DLNB,
    n_q,
    T: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    MAXB: tl.constexpr,
    SL: tl.constexpr,
    H: tl.constexpr,
    RPD: tl.constexpr,
    EPS: tl.constexpr,
    USE_TEMP: tl.constexpr,
):
    ci = tl.program_id(0)
    bh = tl.program_id(1)
    od = tl.arange(0, D)
    ob = tl.arange(0, MAXB)
    osl = tl.arange(0, SL)
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    stemp = tl.load(STEMP + (bh % H)).to(tl.float32) if USE_TEMP else 1.0
    cbase = bh * (T + 1) * D
    obase = (bh * n_q + ci) * M * D
    kbase = bh * T * D
    bmask = (SL + ob)[:, None] < M

    dlnw = tl.zeros([D], tl.float32)
    dlnb = tl.zeros([D], tl.float32)
    dstemp = tl.zeros([D], tl.float32)  # reduced to a scalar at the end

    # ===== bucket slots [SL : SL+MAXB] : bk_out = readout_LN(CUMK[b]-CUMK[a])*stemp
    a = tl.load(A_IDX + ci * MAXB + ob)
    b = tl.load(B_IDX + ci * MAXB + ob)
    cnt = tl.load(CNT + ci * MAXB + ob)
    dbk_slot = tl.load(
        DBK + obase + (SL + ob)[:, None] * D + od[None, :], mask=bmask, other=0.0
    ).to(tl.float32)
    dbv_slot = tl.load(
        DBV + obase + (SL + ob)[:, None] * D + od[None, :], mask=bmask, other=0.0
    ).to(tl.float32)

    # bv_out = (CUMV[b] - CUMV[a]) / cnt  -> linear, no recompute needed
    dbv = dbv_slot / cnt[:, None]
    tl.atomic_add(DCUMV + cbase + b[:, None] * D + od[None, :], dbv)
    tl.atomic_add(DCUMV + cbase + a[:, None] * D + od[None, :], -dbv)

    # bk_out = LN(bk) * stemp; recompute bk = CUMK[b]-CUMK[a], then LN backward
    bk = tl.load(CUMK + cbase + b[:, None] * D + od[None, :]) - tl.load(
        CUMK + cbase + a[:, None] * D + od[None, :]
    )  # [MAXB, D] fp32
    mu = tl.sum(bk, 1) / D
    bkc = bk - mu[:, None]
    rstd = tl.rsqrt(tl.sum(bkc * bkc, 1) / D + EPS)
    xn = bkc * rstd[:, None]
    ln_out = xn * ln_w + ln_b  # readout LN output (pre-stemp)
    dy = dbk_slot * stemp  # grad wrt ln_out
    if USE_TEMP:
        dstemp += tl.sum(dbk_slot * ln_out, 0)
    dxn = dy * ln_w
    dbk = rstd[:, None] * (
        dxn - tl.sum(dxn, 1)[:, None] / D - xn * (tl.sum(dxn * xn, 1)[:, None] / D)
    )
    tl.atomic_add(DCUMK + cbase + b[:, None] * D + od[None, :], dbk)
    tl.atomic_add(DCUMK + cbase + a[:, None] * D + od[None, :], -dbk)
    dlnw += tl.sum(dy * xn, 0)
    dlnb += tl.sum(dy, 0)

    # ===== sink slots [0 : SL] : k -> prepare_state_k -> readout LN, then *stemp =
    sk = tl.load(K + kbase + osl[:, None] * D + od[None, :])
    dsink_k = tl.load(DBK + obase + osl[:, None] * D + od[None, :]).to(tl.float32)
    dsink_v = tl.load(DBV + obase + osl[:, None] * D + od[None, :]).to(tl.float32)
    tl.atomic_add(
        DV + kbase + osl[:, None] * D + od[None, :], dsink_v
    )  # raw passthrough

    skf = tl.where(od[None, :] < RPD, 0.0, sk.to(tl.float32))
    m1 = tl.sum(skf, 1) / D
    skfc = skf - m1[:, None]
    r1 = tl.rsqrt(tl.sum(skfc * skfc, 1) / D + EPS)
    xn1 = skfc * r1[:, None]
    s1 = xn1 * ln_w + ln_b  # prepare LN output
    m2 = tl.sum(s1, 1) / D
    s1c = s1 - m2[:, None]
    r2 = tl.rsqrt(tl.sum(s1c * s1c, 1) / D + EPS)
    xn2 = s1c * r2[:, None]
    s1_out = xn2 * ln_w + ln_b  # readout LN (pre-stemp)
    dy2 = dsink_k * stemp  # grad wrt s1_out
    if USE_TEMP:
        dstemp += tl.sum(dsink_k * s1_out, 0)
    # readout LN backward
    dxn2 = dy2 * ln_w
    ds1 = r2[:, None] * (
        dxn2 - tl.sum(dxn2, 1)[:, None] / D - xn2 * (tl.sum(dxn2 * xn2, 1)[:, None] / D)
    )
    dlnw += tl.sum(dy2 * xn2, 0)
    dlnb += tl.sum(dy2, 0)
    # prepare LN backward (dy = ds1)
    dxn1 = ds1 * ln_w
    dskf = r1[:, None] * (
        dxn1 - tl.sum(dxn1, 1)[:, None] / D - xn1 * (tl.sum(dxn1 * xn1, 1)[:, None] / D)
    )
    dlnw += tl.sum(ds1 * xn1, 0)
    dlnb += tl.sum(ds1, 0)
    dsk = tl.where(od[None, :] < RPD, 0.0, dskf)
    tl.atomic_add(DK + kbase + osl[:, None] * D + od[None, :], dsk)

    tl.atomic_add(DLNW + od, dlnw)
    tl.atomic_add(DLNB + od, dlnb)
    if USE_TEMP:
        tl.atomic_add(DSTEMP + (bh % H), tl.sum(dstemp, 0))


@triton.jit
def _monotone_csum_bwd_kernel(
    DCUMK,
    DCUMV,
    K,
    V,
    GATE,
    LN_W,
    LN_B,
    DK,
    DV,
    DGATE,
    DLNW,
    DLNB,
    n_tiles,
    T: tl.constexpr,
    CL: tl.constexpr,
    D: tl.constexpr,
    RPD: tl.constexpr,
    EPS: tl.constexpr,
    USE_GATE: tl.constexpr,
):
    # dpk[t] = sum_{j > t} dCUMK[j]  -- a suffix sum over TOKENS, the transpose of
    # the forward prefix sum. Walked tile-reverse: an intra-tile reverse cumsum
    # gives the within-tile part, a running [D] carry the rest. With the merge
    # gate, the forward sums g*pk / g*v, so dpk scales by g and each token also
    # gets a dGATE term.
    bh = tl.program_id(0)
    od = tl.arange(0, D)
    ocl = tl.arange(0, CL)
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    kbase = bh * T * D
    cbase = bh * (T + 1) * D
    gbase = bh * T
    ones_cl = tl.full([CL, 1], 1.0, tl.float32)

    carry_k = tl.zeros([D], tl.float32)  # sum of dCUMK[j] past the current tile
    carry_v = tl.zeros([D], tl.float32)
    dlnw = tl.zeros([D], tl.float32)
    dlnb = tl.zeros([D], tl.float32)

    for it in range(0, n_tiles):
        tile = n_tiles - 1 - it
        row0 = tile * CL
        rows = row0 + ocl
        # dCUMK rows [row0+1 .. row0+CL]  (CUMK[0] is the const 0, never read)
        gk = tl.load(DCUMK + cbase + (row0 + 1 + ocl)[:, None] * D + od[None, :])
        gv = tl.load(DCUMV + cbase + (row0 + 1 + ocl)[:, None] * D + od[None, :])
        # reverse-inclusive cumsum: rev[i] = sum_{m>=i} g[m]
        tot_k = tl.sum(gk, 0)
        tot_v = tl.sum(gv, 0)
        dpk = (tot_k[None, :] - tl.cumsum(gk, 0) + gk) + carry_k[None, :]  # [CL,D]
        dpv = (tot_v[None, :] - tl.cumsum(gv, 0) + gv) + carry_v[None, :]
        carry_k = carry_k + tot_k
        carry_v = carry_v + tot_v

        koff = kbase + rows[:, None] * D + od[None, :]
        kc = tl.load(K + koff)
        skf = tl.where(od[None, :] < RPD, 0.0, kc.to(tl.float32))
        m = tl.sum(skf, 1) / D
        skfc = skf - m[:, None]
        rstd = tl.rsqrt(tl.sum(skfc * skfc, 1) / D + EPS)
        xn = skfc * rstd[:, None]  # [CL, D]
        g = tl.load(GATE + gbase + rows)[:, None] if USE_GATE else ones_cl
        dy = dpk * g  # [CL, D] grad wrt pk (post-LN, pre-gate)
        dvc = dpv * g  # [CL, D] grad wrt raw v
        dxn = dy * ln_w
        dskf = rstd[:, None] * (
            dxn - tl.sum(dxn, 1)[:, None] / D - xn * (tl.sum(dxn * xn, 1)[:, None] / D)
        )
        dkc = tl.where(od[None, :] < RPD, 0.0, dskf)
        # gather bwd (separate launch) wrote sink grads into DK[0:SL]; tiles are
        # disjoint and this program owns all T rows -> load+add is safe.
        tl.store(DK + koff, tl.load(DK + koff) + dkc)
        tl.store(DV + koff, tl.load(DV + koff) + dvc)
        if USE_GATE:
            pk = xn * ln_w + ln_b  # [CL, D]
            vc = tl.load(V + koff).to(tl.float32)
            dg = tl.sum(dpk * pk, 1) + tl.sum(dpv * vc, 1)  # [CL]
            tl.store(DGATE + gbase + rows, dg)
        dlnw += tl.sum(dy * xn, 0)
        dlnb += tl.sum(dy, 0)

    tl.atomic_add(DLNW + od, dlnw)
    tl.atomic_add(DLNB + od, dlnb)


class _MonotonePhase1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k,
        v,
        ln_w,
        ln_b,
        gate,
        stemp,
        a_idx,
        b_idx,
        cnt,
        T,
        CL,
        D,
        H,
        n_q,
        MAXB,
        M,
        SL,
        RPD,
        eps,
        USE_GATE,
        USE_TEMP,
    ):
        BH = k.shape[0]
        n_tiles = T // CL
        k, v = k.contiguous(), v.contiguous()
        gate, stemp = gate.contiguous(), stemp.contiguous()
        ln_w_c, ln_b_c = ln_w.contiguous(), ln_b.contiguous()
        cumK = torch.empty(BH, T + 1, D, device=k.device, dtype=torch.float32)
        cumV = torch.empty(BH, T + 1, D, device=k.device, dtype=torch.float32)
        _monotone_csum_kernel[(BH,)](
            k,
            v,
            gate,
            cumK,
            cumV,
            ln_w_c,
            ln_b_c,
            n_tiles,
            T=T,
            CL=CL,
            D=D,
            RPD=RPD,
            EPS=eps,
            USE_GATE=USE_GATE,
            num_warps=4,
            num_stages=1,
        )
        buck_k = torch.zeros(BH, n_q, M, D, device=k.device, dtype=k.dtype)
        buck_v = torch.zeros(BH, n_q, M, D, device=k.device, dtype=k.dtype)
        _monotone_gather_kernel[(n_q, BH)](
            cumK,
            cumV,
            a_idx,
            b_idx,
            cnt,
            k,
            v,
            ln_w_c,
            ln_b_c,
            stemp,
            buck_k,
            buck_v,
            n_q,
            T=T,
            D=D,
            M=M,
            MAXB=MAXB,
            SL=SL,
            H=H,
            RPD=RPD,
            EPS=eps,
            USE_TEMP=USE_TEMP,
            num_warps=4,
        )
        ctx.save_for_backward(k, v, ln_w, ln_b, gate, stemp, cumK, a_idx, b_idx, cnt)
        ctx.dims = (T, CL, D, H, n_q, MAXB, M, SL, RPD, eps, USE_GATE, USE_TEMP)
        return buck_k, buck_v

    @staticmethod
    def backward(ctx, dbuck_k, dbuck_v):
        k, v, ln_w, ln_b, gate, stemp, cumK, a_idx, b_idx, cnt = ctx.saved_tensors
        T, CL, D, H, n_q, MAXB, M, SL, RPD, eps, USE_GATE, USE_TEMP = ctx.dims
        BH = k.shape[0]
        n_tiles = T // CL
        dbuck_k = dbuck_k.contiguous()
        dbuck_v = dbuck_v.contiguous()
        ln_w_c, ln_b_c = ln_w.contiguous(), ln_b.contiguous()

        dK = torch.zeros(BH, T, D, device=k.device, dtype=torch.float32)
        dV = torch.zeros(BH, T, D, device=k.device, dtype=torch.float32)
        dCUMK = torch.zeros(BH, T + 1, D, device=k.device, dtype=torch.float32)
        dCUMV = torch.zeros(BH, T + 1, D, device=k.device, dtype=torch.float32)
        dLNW = torch.zeros(D, device=k.device, dtype=torch.float32)
        dLNB = torch.zeros(D, device=k.device, dtype=torch.float32)
        dSTEMP = torch.zeros(H, device=k.device, dtype=torch.float32)
        dGATE = torch.zeros(BH, T, device=k.device, dtype=torch.float32)

        _monotone_gather_bwd_kernel[(n_q, BH)](
            dbuck_k,
            dbuck_v,
            cumK,
            a_idx,
            b_idx,
            cnt,
            k,
            ln_w_c,
            ln_b_c,
            stemp,
            dCUMK,
            dCUMV,
            dK,
            dV,
            dSTEMP,
            dLNW,
            dLNB,
            n_q,
            T=T,
            D=D,
            M=M,
            MAXB=MAXB,
            SL=SL,
            H=H,
            RPD=RPD,
            EPS=eps,
            USE_TEMP=USE_TEMP,
            num_warps=4,
        )
        _monotone_csum_bwd_kernel[(BH,)](
            dCUMK,
            dCUMV,
            k,
            v,
            gate,
            ln_w_c,
            ln_b_c,
            dK,
            dV,
            dGATE,
            dLNW,
            dLNB,
            n_tiles,
            T=T,
            CL=CL,
            D=D,
            RPD=RPD,
            EPS=eps,
            USE_GATE=USE_GATE,
            num_warps=4,
            num_stages=1,
        )
        return (
            dK.to(k.dtype),
            dV.to(v.dtype),
            dLNW.to(ln_w.dtype),
            dLNB.to(ln_b.dtype),
            dGATE.to(gate.dtype),
            dSTEMP.to(stemp.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def monotone_phase1(attn, k, v, gate):
    """Monotone PHASE 1, fully Triton + differentiable: token prefix-sum kernel +
    gather kernel, with a matching backward. Grads flow to k / v, the ln_s_k
    LayerNorm params, the merge gate, and the per-head state temperature --
    the full KVM feature set. front_head_temp is covered by autograd outside."""
    B, H, T, D = k.shape
    cl, SL = attn.chunk_len, attn.sink_len
    plan = _monotone_plan(attn, T, k.device)
    n_q, MAXB, M = plan["n_q"], plan["MAXB"], plan["M"]
    use_gate = attn.cfg.use_merge_gate
    use_temp = attn.cfg.use_head_temps

    k_ = k.reshape(B * H, T, D)
    v_ = v.reshape(B * H, T, D)
    gate_ = gate.reshape(B * H, T)  # [B,H,T,1] -> [BH,T]
    stemp_ = (
        attn.state_head_temp
        if use_temp
        else torch.ones(H, device=k.device, dtype=k.dtype)
    )
    buck_k, buck_v = _MonotonePhase1.apply(
        k_,
        v_,
        attn.ln_s_k.weight,
        attn.ln_s_k.bias,
        gate_,
        stemp_,
        plan["a_idx"],
        plan["b_idx"],
        plan["cnt"],
        T,
        cl,
        D,
        H,
        n_q,
        MAXB,
        M,
        SL,
        attn.rope_partial_dim,
        attn.ln_s_k.eps,
        use_gate,
        use_temp,
    )
    return buck_k, buck_v, plan["bias"]
