"""PHASE 1 for KVM -- the merge recurrence, as Triton kernels.

A sequential recurrence carrying an [M,D] state across chunks: per chunk, a
data-dependent append/merge split (cosine-novelty rank + argmax routing) folds
the overflow chunk into the state. Forward + backward -- the routing is frozen
(non-differentiable, as in the naive KVM path) and the scatter-add chain + the
LayerNorms are differentiated. State is SRAM-resident for M<=128; M>128 uses a
state-tiled backward (_kvm_merge_bwd_tiled_kernel).
"""

import torch
import triton
import triton.language as tl

from .common import _next_pow2


# ======================================================================
# KVM PHASE 1 -- the merge recurrence, as a Triton kernel (grid (B*H,))
#
# v2: supports a *growing* budget (power_law / saturation / fixed). Per chunk
# the data-independent budget gives n_append; the kernel does the data-dependent
# append/merge split (rank overflow tokens by cosine-novelty, rank via a
# [CL,CL] compare -- no sort), then append-routes the novel ones to fresh slots
# and argmax-routes the rest. State is SRAM-resident (so M is bounded -- see the
# assert in kvm_merge_forward; larger budgets need state tiling). Features
# (use_vlens / use_merge_gate / use_head_temps) are still off.
# ======================================================================
def _kvm_budget_plan(attn, T, device):
    """Data-independent per-chunk budget: n_append[c], cur_before[c], the
    final state size M, and the PHASE-2 bias mask (unfilled slots -> -inf)."""
    cache = attn.__dict__.setdefault("_kvm_budget_plans", {})
    key = (T, str(device))
    if key in cache:
        return cache[key]

    cl = attn.chunk_len
    front = min(T, attn.bswa_len)
    n_q = (T - front) // cl
    init = min(T, cl)
    n_app, cur_before = [init], [0]  # fold 0 = init: append the whole chunk
    cur = init
    for c in range(1, n_q):  # fold c == recurrent overflow ci=c-1
        qe = front + c * cl
        nb = attn._bswa_begin(min(T, qe + cl))
        desired = attn._desired_state_len(qe, nb, cur)
        na = min(max(desired - cur, 0), cl)
        n_app.append(na)
        cur_before.append(cur)
        cur += na
    cur_after = [cb + na for cb, na in zip(cur_before, n_app)]
    M = max(16, _next_pow2(cur))

    bias = torch.full((n_q, M), float("-inf"), dtype=torch.float32)
    for c in range(n_q):
        bias[c, : cur_after[c]] = 0.0  # filled slots visible; rest masked

    plan = dict(
        n_q=n_q,
        M=M,
        n_app=torch.tensor(n_app, dtype=torch.int32, device=device),
        cur_before=torch.tensor(cur_before, dtype=torch.int32, device=device),
        bias=bias.to(device),
    )
    cache[key] = plan
    return plan


@triton.jit
def _kvm_merge_kernel(
    K,
    V,
    BK,
    BV,
    SK_TRAJ,
    SVLEN_TRAJ,
    N_APP,
    CUR,
    LN_W,
    LN_B,  # N_APP,CUR: int32 [n_q]
    n_q,
    T: tl.constexpr,
    CL: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    RPD: tl.constexpr,
    SINK: tl.constexpr,
    EPS: tl.constexpr,
    SAVE_TRAJ: tl.constexpr,
):
    bh = tl.program_id(0)
    od = tl.arange(0, D)
    ocl = tl.arange(0, CL)
    om = tl.arange(0, M)
    in_dtype = K.dtype.element_ty
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    kbase = bh * T * D

    # SRAM-resident fp32 state, carried across the chunk loop. Matmuls run in
    # the input dtype (fp16/bf16); LN stats and accumulators are fp32.
    sk = tl.zeros([M, D], tl.float32)
    sv = tl.zeros([M, D], tl.float32)
    svlen = tl.zeros([M], tl.float32)

    for c in range(0, n_q):  # the sequential recurrence
        row = c * CL
        n_app = tl.load(N_APP + c)
        cur = tl.load(CUR + c)
        kc = tl.load(K + kbase + (row + ocl)[:, None] * D + od[None, :])
        vc = tl.load(V + kbase + (row + ocl)[:, None] * D + od[None, :])

        # prepare_state_k: zero RoPE channels, LayerNorm over D (fp32 stats)
        kf = tl.where(od[None, :] < RPD, 0.0, kc.to(tl.float32))
        mu = tl.sum(kf, 1) / D
        kf = kf - mu[:, None]
        var = tl.sum(kf * kf, 1) / D
        pk = (kf * tl.rsqrt(var + EPS)[:, None] * ln_w + ln_b).to(in_dtype)  # [CL,D]

        # --- SPLIT: rank tokens by cosine sim to the PRE-append state.
        #     The split score includes the sink slots; only slots >= cur masked.
        smu = tl.sum(sk, 1) / D
        sc = sk - smu[:, None]
        svar = tl.sum(sc * sc, 1) / D
        skn = (sc * tl.rsqrt(svar + EPS)[:, None] * ln_w + ln_b).to(in_dtype)
        sim = tl.dot(pk, tl.trans(skn))  # [CL,M] fp32
        sim = tl.where(om[None, :] < cur, sim, -float("inf"))
        score = tl.max(sim, 1)
        lt = score[None, :] < score[:, None]
        eq = (score[None, :] == score[:, None]) & (ocl[None, :] < ocl[:, None])
        rank = tl.sum((lt | eq).to(tl.int32), 1)  # ascending rank [CL]
        is_append = rank < n_app
        is_merge = rank >= n_app
        app_pos = tl.cumsum(is_append.to(tl.int32), 0) - 1  # fresh-slot offset
        route_app = cur + app_pos

        # --- APPEND: the n_app most-novel tokens -> fresh slots [cur, cur+n_app)
        oh_app = (is_append[:, None] & (route_app[:, None] == om[None, :])).to(in_dtype)
        sk = sk + tl.dot(tl.trans(oh_app), pk)
        sv = sv + tl.dot(tl.trans(oh_app), vc)
        svlen = svlen + tl.sum(oh_app.to(tl.float32), 0)

        # --- MERGE: route the rest via argmax against the POST-append state.
        #     Merge routing excludes the sink slots (protected).
        smu = tl.sum(sk, 1) / D
        sc = sk - smu[:, None]
        svar = tl.sum(sc * sc, 1) / D
        skn = (sc * tl.rsqrt(svar + EPS)[:, None] * ln_w + ln_b).to(in_dtype)
        sim = tl.dot(pk, tl.trans(skn))
        valid = (om[None, :] >= SINK) & (om[None, :] < cur + n_app)
        sim = tl.where(valid, sim, -float("inf"))
        route_mrg = tl.argmax(sim, 1)  # [CL]
        oh_mrg = (is_merge[:, None] & (route_mrg[:, None] == om[None, :])).to(in_dtype)
        sk = sk + tl.dot(tl.trans(oh_mrg), pk)
        sv = sv + tl.dot(tl.trans(oh_mrg), vc)
        svlen = svlen + tl.sum(oh_mrg.to(tl.float32), 0)

        # --- save the post-merge state trajectory (for the backward) ---
        if SAVE_TRAJ:
            ttb = (bh * n_q + c) * M * D
            tl.store(SK_TRAJ + ttb + om[:, None] * D + od[None, :], sk)
            tl.store(SVLEN_TRAJ + (bh * n_q + c) * M + om, svlen)

        # --- readout transform (LN(sk), sv / count) -> state_traj[bh, c] ---
        rmu = tl.sum(sk, 1) / D
        rc = sk - rmu[:, None]
        rvar = tl.sum(rc * rc, 1) / D
        rk = rc * tl.rsqrt(rvar + EPS)[:, None] * ln_w + ln_b
        rv = tl.where(svlen[:, None] > 0, sv / svlen[:, None], 0.0)
        obase = (bh * n_q + c) * M * D
        tl.store(BK + obase + om[:, None] * D + od[None, :], rk.to(BK.dtype.element_ty))
        tl.store(BV + obase + om[:, None] * D + od[None, :], rv.to(BV.dtype.element_ty))


# ----------------------------------------------------------------------
# KVM PHASE 1 backward -- the reverse of the merge recurrence.
#
# Routing (the cosine-novelty rank + the argmax merge target) is *frozen*:
# non-differentiable, exactly as the naive KVM path treats it (the split runs
# under no_grad). The differentiable graph is just the scatter-add chain
#   sk_c = sk_{c-1} + OH_c^T @ pk_c ,  sv_c = sv_{c-1} + OH_c^T @ vc_c
# plus prepare_state_k (an LN) and the readout LN. OH_c (the per-token one-hot)
# is recomputed in reverse from the saved state trajectory -- bit-identical to
# the forward, since the recompute sees the same fp32 sk and the same
# input-dtype matmuls. Walking c = n_q-1 .. 0 carries (g_sk, g_sv); the carry
# is the identity (sk_c = sk_{c-1} + ...), the per-chunk readout adds in.
# ----------------------------------------------------------------------
@triton.jit
def _kvm_merge_bwd_kernel(
    K,
    V,
    DBK,
    DBV,
    SK_TRAJ,
    SVLEN_TRAJ,
    N_APP,
    CUR,
    LN_W,
    LN_B,
    DK,
    DV,
    DLNW,
    DLNB,
    n_q,
    T: tl.constexpr,
    CL: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    RPD: tl.constexpr,
    SINK: tl.constexpr,
    EPS: tl.constexpr,
):
    bh = tl.program_id(0)
    od = tl.arange(0, D)
    ocl = tl.arange(0, CL)
    om = tl.arange(0, M)
    in_dtype = K.dtype.element_ty
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    kbase = bh * T * D

    g_sk = tl.zeros([M, D], tl.float32)  # grad carried back through state
    g_sv = tl.zeros([M, D], tl.float32)
    dlnw = tl.zeros([D], tl.float32)
    dlnb = tl.zeros([D], tl.float32)

    for i in range(0, n_q):
        c = n_q - 1 - i
        row = c * CL
        n_app = tl.load(N_APP + c)
        cur = tl.load(CUR + c)
        tbase = (bh * n_q + c) * M * D

        sk_c = tl.load(SK_TRAJ + tbase + om[:, None] * D + od[None, :])
        svlen_c = tl.load(SVLEN_TRAJ + (bh * n_q + c) * M + om)
        cm1 = tl.maximum(c - 1, 0)
        sk_pre = tl.load(
            SK_TRAJ + (bh * n_q + cm1) * M * D + om[:, None] * D + od[None, :]
        )
        sk_pre = sk_pre * (c > 0).to(tl.float32)  # chunk 0 starts from zeros

        kc = tl.load(K + kbase + (row + ocl)[:, None] * D + od[None, :])

        # prepare_state_k(kc): zero the RoPE channels, then LayerNorm
        kf = tl.where(od[None, :] < RPD, 0.0, kc.to(tl.float32))
        kmu = tl.sum(kf, 1) / D
        kfc = kf - kmu[:, None]
        krstd = tl.rsqrt(tl.sum(kfc * kfc, 1) / D + EPS)
        kxn = kfc * krstd[:, None]
        pk = (kxn * ln_w + ln_b).to(in_dtype)  # [CL,D]

        # --- recompute the frozen routing, exactly as the forward ---
        smu = tl.sum(sk_pre, 1) / D
        sc = sk_pre - smu[:, None]
        srstd = tl.rsqrt(tl.sum(sc * sc, 1) / D + EPS)
        skn = (sc * srstd[:, None] * ln_w + ln_b).to(in_dtype)
        sim = tl.dot(pk, tl.trans(skn))
        sim = tl.where(om[None, :] < cur, sim, -float("inf"))
        score = tl.max(sim, 1)
        lt = score[None, :] < score[:, None]
        eq = (score[None, :] == score[:, None]) & (ocl[None, :] < ocl[:, None])
        rank = tl.sum((lt | eq).to(tl.int32), 1)
        is_append = rank < n_app
        is_merge = rank >= n_app
        app_pos = tl.cumsum(is_append.to(tl.int32), 0) - 1
        route_app = cur + app_pos
        oh_app = (is_append[:, None] & (route_app[:, None] == om[None, :])).to(in_dtype)
        post_sk = sk_pre + tl.dot(tl.trans(oh_app), pk)
        smu = tl.sum(post_sk, 1) / D
        sc = post_sk - smu[:, None]
        srstd = tl.rsqrt(tl.sum(sc * sc, 1) / D + EPS)
        skn = (sc * srstd[:, None] * ln_w + ln_b).to(in_dtype)
        sim = tl.dot(pk, tl.trans(skn))
        valid = (om[None, :] >= SINK) & (om[None, :] < cur + n_app)
        sim = tl.where(valid, sim, -float("inf"))
        route_mrg = tl.argmax(sim, 1)
        oh_mrg = (is_merge[:, None] & (route_mrg[:, None] == om[None, :])).to(in_dtype)
        oh = (oh_app + oh_mrg).to(tl.float32)  # [CL,M] one-hot per row

        # --- readout LayerNorm backward: BK[c] = LN(sk_c) ---
        rmu = tl.sum(sk_c, 1) / D
        rc = sk_c - rmu[:, None]
        rrstd = tl.rsqrt(tl.sum(rc * rc, 1) / D + EPS)
        rxn = rc * rrstd[:, None]
        dbk_c = tl.load(DBK + tbase + om[:, None] * D + od[None, :]).to(tl.float32)
        dxn = dbk_c * ln_w
        dsk_ro = rrstd[:, None] * (
            dxn
            - tl.sum(dxn, 1)[:, None] / D
            - rxn * (tl.sum(dxn * rxn, 1)[:, None] / D)
        )
        dlnw += tl.sum(dbk_c * rxn, 0)
        dlnb += tl.sum(dbk_c, 0)

        # --- total grad on the post-merge state of chunk c ---
        dsk_c = g_sk + dsk_ro
        dbv_c = tl.load(DBV + tbase + om[:, None] * D + od[None, :]).to(tl.float32)
        dsv_c = g_sv + tl.where(svlen_c[:, None] > 0.0, dbv_c / svlen_c[:, None], 0.0)

        # --- scatter back through OH: d pk = OH @ dsk, d vc = OH @ dsv ---
        # OH is one-hot per row, so this is an exact gather (ieee, not tf32).
        d_pk = tl.dot(oh, dsk_c, input_precision="ieee")  # [CL,D]
        d_vc = tl.dot(oh, dsv_c, input_precision="ieee")  # [CL,D]

        # --- prepare_state_k backward: d pk -> d kc ---
        dxn = d_pk * ln_w
        dkf = krstd[:, None] * (
            dxn
            - tl.sum(dxn, 1)[:, None] / D
            - kxn * (tl.sum(dxn * kxn, 1)[:, None] / D)
        )
        dlnw += tl.sum(d_pk * kxn, 0)
        dlnb += tl.sum(d_pk, 0)
        dkc = tl.where(od[None, :] < RPD, 0.0, dkf)
        koff = kbase + (row + ocl)[:, None] * D + od[None, :]
        tl.store(DK + koff, dkc.to(DK.dtype.element_ty))
        tl.store(DV + koff, d_vc.to(DV.dtype.element_ty))

        # carry: sk_c = sk_{c-1} + OH^T@pk  =>  d sk_{c-1} = d sk_c
        g_sk = dsk_c
        g_sv = dsv_c

    tl.atomic_add(DLNW + od, dlnw)
    tl.atomic_add(DLNB + od, dlnb)


# state-tiled variant for M > 128: at M=256 even two [M,D] fp32 tensors blow the
# ~99KB SRAM budget, so the carried state grads g_sk/g_sv live in HBM (G_SK/G_SV)
# and every state-spanning op loops the M dimension in MT-slot tiles. Same math
# as _kvm_merge_bwd_kernel, just tiled; three tile passes per chunk: SPLIT score,
# MERGE argmax (running max/argmax across tiles == the forward's full reduction),
# then the gradient scatter + carry.
@triton.jit
def _kvm_merge_bwd_tiled_kernel(
    K,
    V,
    DBK,
    DBV,
    SK_TRAJ,
    SVLEN_TRAJ,
    N_APP,
    CUR,
    LN_W,
    LN_B,
    G_SK,
    G_SV,
    DK,
    DV,
    DLNW,
    DLNB,
    n_q,
    T: tl.constexpr,
    CL: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    MT: tl.constexpr,
    RPD: tl.constexpr,
    SINK: tl.constexpr,
    EPS: tl.constexpr,
):
    bh = tl.program_id(0)
    od = tl.arange(0, D)
    ocl = tl.arange(0, CL)
    omt = tl.arange(0, MT)
    n_mt = M // MT
    in_dtype = K.dtype.element_ty
    ln_w = tl.load(LN_W + od).to(tl.float32)[None, :]
    ln_b = tl.load(LN_B + od).to(tl.float32)[None, :]
    kbase = bh * T * D
    gbase = bh * M * D

    dlnw = tl.zeros([D], tl.float32)
    dlnb = tl.zeros([D], tl.float32)

    for i in range(0, n_q):
        c = n_q - 1 - i
        row = c * CL
        n_app = tl.load(N_APP + c)
        cur = tl.load(CUR + c)
        pre0 = (c > 0).to(tl.float32)  # chunk 0 starts from zeros
        pbase = (bh * n_q + tl.maximum(c - 1, 0)) * M * D
        cbase = (bh * n_q + c) * M * D

        # recompute pk = prepare_state_k(kc)
        kc = tl.load(K + kbase + (row + ocl)[:, None] * D + od[None, :])
        kf = tl.where(od[None, :] < RPD, 0.0, kc.to(tl.float32))
        kmu = tl.sum(kf, 1) / D
        kfc = kf - kmu[:, None]
        krstd = tl.rsqrt(tl.sum(kfc * kfc, 1) / D + EPS)
        kxn = kfc * krstd[:, None]
        pk = (kxn * ln_w + ln_b).to(in_dtype)

        # --- PASS 1: SPLIT score = max over M (running across tiles) ---
        score = tl.full([CL], -float("inf"), tl.float32)
        for mt in range(0, n_mt):
            mo = mt * MT + omt
            spre = tl.load(SK_TRAJ + pbase + mo[:, None] * D + od[None, :]) * pre0
            smu = tl.sum(spre, 1) / D
            sc = spre - smu[:, None]
            srstd = tl.rsqrt(tl.sum(sc * sc, 1) / D + EPS)
            skn = (sc * srstd[:, None] * ln_w + ln_b).to(in_dtype)
            sim = tl.dot(pk, tl.trans(skn))  # [CL,MT]
            sim = tl.where(mo[None, :] < cur, sim, -float("inf"))
            score = tl.maximum(score, tl.max(sim, 1))
        lt = score[None, :] < score[:, None]
        eq = (score[None, :] == score[:, None]) & (ocl[None, :] < ocl[:, None])
        rank = tl.sum((lt | eq).to(tl.int32), 1)
        is_append = rank < n_app
        is_merge = rank >= n_app
        app_pos = tl.cumsum(is_append.to(tl.int32), 0) - 1
        route_app = cur + app_pos

        # --- PASS 2: MERGE argmax over M of sim(post-append state) ---
        best = tl.full([CL], -float("inf"), tl.float32)
        route_mrg = tl.zeros([CL], tl.int32)
        for mt in range(0, n_mt):
            mo = mt * MT + omt
            spre = tl.load(SK_TRAJ + pbase + mo[:, None] * D + od[None, :]) * pre0
            oh_app = (is_append[:, None] & (route_app[:, None] == mo[None, :])).to(
                in_dtype
            )
            post = spre + tl.dot(tl.trans(oh_app), pk)  # [MT,D]
            smu = tl.sum(post, 1) / D
            sc = post - smu[:, None]
            srstd = tl.rsqrt(tl.sum(sc * sc, 1) / D + EPS)
            skn = (sc * srstd[:, None] * ln_w + ln_b).to(in_dtype)
            sim = tl.dot(pk, tl.trans(skn))  # [CL,MT]
            valid = (mo[None, :] >= SINK) & (mo[None, :] < cur + n_app)
            sim = tl.where(valid, sim, -float("inf"))
            tmax = tl.max(sim, 1)
            targ = mt * MT + tl.argmax(sim, 1).to(tl.int32)
            better = tmax > best
            route_mrg = tl.where(better, targ, route_mrg)
            best = tl.maximum(best, tmax)

        # --- PASS 3: gradient scatter + state-grad carry ---
        d_pk = tl.zeros([CL, D], tl.float32)
        d_vc = tl.zeros([CL, D], tl.float32)
        for mt in range(0, n_mt):
            mo = mt * MT + omt
            tb = cbase + mo[:, None] * D + od[None, :]
            sk_c = tl.load(SK_TRAJ + tb)
            svlen_c = tl.load(SVLEN_TRAJ + (bh * n_q + c) * M + mo)
            rmu = tl.sum(sk_c, 1) / D
            rc = sk_c - rmu[:, None]
            rrstd = tl.rsqrt(tl.sum(rc * rc, 1) / D + EPS)
            rxn = rc * rrstd[:, None]
            dbk_c = tl.load(DBK + tb).to(tl.float32)
            dxn = dbk_c * ln_w
            dsk_ro = rrstd[:, None] * (
                dxn
                - tl.sum(dxn, 1)[:, None] / D
                - rxn * (tl.sum(dxn * rxn, 1)[:, None] / D)
            )
            dlnw += tl.sum(dbk_c * rxn, 0)
            dlnb += tl.sum(dbk_c, 0)
            g_sk = tl.load(G_SK + gbase + mo[:, None] * D + od[None, :])
            g_sv = tl.load(G_SV + gbase + mo[:, None] * D + od[None, :])
            dsk_c = g_sk + dsk_ro
            dbv_c = tl.load(DBV + tb).to(tl.float32)
            dsv_c = g_sv + tl.where(
                svlen_c[:, None] > 0.0, dbv_c / svlen_c[:, None], 0.0
            )
            oh_app = is_append[:, None] & (route_app[:, None] == mo[None, :])
            oh_mrg = is_merge[:, None] & (route_mrg[:, None] == mo[None, :])
            oh = (oh_app | oh_mrg).to(tl.float32)  # [CL,MT] one-hot per row
            d_pk += tl.dot(oh, dsk_c, input_precision="ieee")
            d_vc += tl.dot(oh, dsv_c, input_precision="ieee")
            tl.store(G_SK + gbase + mo[:, None] * D + od[None, :], dsk_c)
            tl.store(G_SV + gbase + mo[:, None] * D + od[None, :], dsv_c)

        # prepare_state_k backward: d_pk -> d_kc
        dxn = d_pk * ln_w
        dkf = krstd[:, None] * (
            dxn
            - tl.sum(dxn, 1)[:, None] / D
            - kxn * (tl.sum(dxn * kxn, 1)[:, None] / D)
        )
        dlnw += tl.sum(d_pk * kxn, 0)
        dlnb += tl.sum(d_pk, 0)
        dkc = tl.where(od[None, :] < RPD, 0.0, dkf)
        koff = kbase + (row + ocl)[:, None] * D + od[None, :]
        tl.store(DK + koff, dkc.to(DK.dtype.element_ty))
        tl.store(DV + koff, d_vc.to(DV.dtype.element_ty))

    tl.atomic_add(DLNW + od, dlnw)
    tl.atomic_add(DLNB + od, dlnb)


class _KVMPhase1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, k, v, ln_w, ln_b, n_app, cur_before, T, CL, D, M, n_q, RPD, SINK, eps
    ):
        BH = k.shape[0]
        k, v = k.contiguous(), v.contiguous()
        ln_w_c, ln_b_c = ln_w.contiguous(), ln_b.contiguous()
        buck_k = torch.empty(BH, n_q, M, D, device=k.device, dtype=k.dtype)
        buck_v = torch.empty(BH, n_q, M, D, device=k.device, dtype=k.dtype)
        save = any(ctx.needs_input_grad[:4])  # k, v, ln_w, ln_b
        if save:
            sk_traj = torch.empty(BH, n_q, M, D, device=k.device, dtype=torch.float32)
            svlen_traj = torch.empty(BH, n_q, M, device=k.device, dtype=torch.float32)
        else:
            sk_traj = svlen_traj = k  # unused dummy (SAVE_TRAJ=False)
        _kvm_merge_kernel[(BH,)](
            k,
            v,
            buck_k,
            buck_v,
            sk_traj,
            svlen_traj,
            n_app,
            cur_before,
            ln_w_c,
            ln_b_c,
            n_q,
            T=T,
            CL=CL,
            D=D,
            M=M,
            RPD=RPD,
            SINK=SINK,
            EPS=eps,
            SAVE_TRAJ=save,
            num_warps=4,
            num_stages=1,
        )
        if save:
            ctx.save_for_backward(
                k, v, ln_w, ln_b, sk_traj, svlen_traj, n_app, cur_before
            )
            ctx.dims = (T, CL, D, M, n_q, RPD, SINK, eps)
        return buck_k, buck_v

    @staticmethod
    def backward(ctx, dbuck_k, dbuck_v):
        k, v, ln_w, ln_b, sk_traj, svlen_traj, n_app, cur_before = ctx.saved_tensors
        T, CL, D, M, n_q, RPD, SINK, eps = ctx.dims
        BH = k.shape[0]
        dbuck_k = dbuck_k.contiguous()
        dbuck_v = dbuck_v.contiguous()
        ln_w_c, ln_b_c = ln_w.contiguous(), ln_b.contiguous()
        dK = torch.zeros(BH, T, D, device=k.device, dtype=k.dtype)
        dV = torch.zeros(BH, T, D, device=k.device, dtype=k.dtype)
        dLNW = torch.zeros(D, device=k.device, dtype=torch.float32)
        dLNB = torch.zeros(D, device=k.device, dtype=torch.float32)
        if M <= 128:  # SRAM-resident, fast path
            _kvm_merge_bwd_kernel[(BH,)](
                k,
                v,
                dbuck_k,
                dbuck_v,
                sk_traj,
                svlen_traj,
                n_app,
                cur_before,
                ln_w_c,
                ln_b_c,
                dK,
                dV,
                dLNW,
                dLNB,
                n_q,
                T=T,
                CL=CL,
                D=D,
                M=M,
                RPD=RPD,
                SINK=SINK,
                EPS=eps,
                num_warps=4,
                num_stages=1,
            )
        else:  # state-tiled, HBM-carried
            g_sk = torch.zeros(BH, M, D, device=k.device, dtype=torch.float32)
            g_sv = torch.zeros(BH, M, D, device=k.device, dtype=torch.float32)
            _kvm_merge_bwd_tiled_kernel[(BH,)](
                k,
                v,
                dbuck_k,
                dbuck_v,
                sk_traj,
                svlen_traj,
                n_app,
                cur_before,
                ln_w_c,
                ln_b_c,
                g_sk,
                g_sv,
                dK,
                dV,
                dLNW,
                dLNB,
                n_q,
                T=T,
                CL=CL,
                D=D,
                M=M,
                MT=32,
                RPD=RPD,
                SINK=SINK,
                EPS=eps,
                num_warps=4,
                num_stages=1,
            )
        return (
            dK.to(k.dtype),
            dV.to(v.dtype),
            dLNW.to(ln_w.dtype),
            dLNB.to(ln_b.dtype),
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


def kvm_merge_forward(attn, k, v):
    """KVM PHASE 1 via the Triton merge kernel, differentiable. Supports growing
    budgets (power_law / saturation / fixed). Grads flow to k / v and ln_s_k;
    the routing decisions are frozen (as in the naive KVM path). Returns
    (buck_k, buck_v, buck_bias)."""
    B, H, T, D = k.shape
    cl = attn.chunk_len
    cfg = attn.cfg
    assert cfg.state_budget_mode in (
        "fixed",
        "power_law",
        "saturation",
    ), f"unknown budget mode {cfg.state_budget_mode!r}"
    assert not (
        cfg.use_vlens or cfg.use_merge_gate or cfg.use_head_temps
    ), "Triton KVM v2: use_vlens / use_merge_gate / use_head_temps must be off"
    plan = _kvm_budget_plan(attn, T, k.device)
    n_q, M = plan["n_q"], plan["M"]
    assert M <= 256, (
        f"Triton KVM v2: final state budget M={M} exceeds 256. The merge kernel "
        f"carries the state in registers; M>256 would need a state-tiled kernel. "
        f"(M=128/256 run but Triton spills the carried state to local memory.)"
    )

    k_ = k.reshape(B * H, T, D)
    v_ = v.reshape(B * H, T, D)
    buck_k, buck_v = _KVMPhase1.apply(
        k_,
        v_,
        attn.ln_s_k.weight,
        attn.ln_s_k.bias,
        plan["n_app"],
        plan["cur_before"],
        T,
        cl,
        D,
        M,
        n_q,
        attn.rope_partial_dim,
        attn.sink_len,
        attn.ln_s_k.eps,
    )
    return buck_k, buck_v, plan["bias"]
