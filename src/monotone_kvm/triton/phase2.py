"""PHASE 2 -- the shared chunked-attention Triton kernels (forward + backward).

Both KVM and monotone-KVM decompose into PHASE 1 (build the compressed state)
and PHASE 2 (chunked attention). PHASE 2 is this kernel set; PHASE 1 lives in
`monotone_phase1.py` / `kvm_phase1.py`.

forward -- grid (n_query_chunks, B*H): load one query chunk, stream
  [ buckets(+sink) | sliding window ] through one online-softmax accumulator,
  write one output chunk + the logsumexp L.
backward -- standard FA2 three-kernel structure, no atomics:
  _phase2_bwd_preprocess : delta = rowsum(O * dO)
  _phase2_bwd_dq_dbuck   : dQ, dBK, dBV (bucket grads are per-chunk)
  _phase2_bwd_draw       : dRK, dRV, re-tiled over CL-aligned raw columns
"""

import torch
import triton
import triton.language as tl


# ======================================================================
# PHASE 2 forward kernel
# ======================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in (2, 4, 8)
        for s in (2, 3, 4)
    ],
    key=["n_qchunks", "T", "M", "D"],  # D affects SRAM -> must be in the key
)
@triton.jit
def _phase2_fwd_kernel(
    Q,
    BK,
    BV,
    BIAS,
    RK,
    RV,
    O,
    L,  # pointers (L: logsumexp out)
    scale,
    n_qchunks,
    T,  # runtime scalars
    FRONT: tl.constexpr,
    CL: tl.constexpr,
    M: tl.constexpr,
    WIN: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ci = tl.program_id(0)  # query chunk
    bh = tl.program_id(1)  # batch * head

    offs_cl = tl.arange(0, CL)
    offs_d = tl.arange(0, D)
    offs_m = tl.arange(0, M)

    # loads stay in the input dtype so the matmuls run on fp16/bf16 tensor
    # cores; the softmax accumulators are fp32. (input_precision="ieee" is
    # ignored for fp16/bf16 and gives accurate fp32 for fp32 inputs.)
    q = tl.load(
        Q + (bh * n_qchunks + ci) * CL * D + offs_cl[:, None] * D + offs_d[None, :]
    )

    m_i = tl.full([CL], -float("inf"), tl.float32)
    l_i = tl.zeros([CL], tl.float32)
    acc = tl.zeros([CL, D], tl.float32)

    # --- segment 1: compressed state (sink prepended), M rows ---
    base = (bh * n_qchunks + ci) * M * D
    bk = tl.load(BK + base + offs_m[:, None] * D + offs_d[None, :])
    bv = tl.load(BV + base + offs_m[:, None] * D + offs_d[None, :])
    bias = tl.load(BIAS + ci * M + offs_m).to(tl.float32)  # [M], -inf = pad
    qk = tl.dot(q, tl.trans(bk)) * scale + bias[None, :]
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.exp(qk - m_ij[:, None])
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(bv.dtype), bv)
    m_i = m_ij

    # --- segment 2: sliding window of raw tokens, causal ---
    # window = [win_begin, win_begin + WIN); win_begin derivable from ci.
    # causal mask is a pure offset relation: masked iff offs_n > offs_cl + (WIN-CL).
    win_begin = FRONT + (ci + 1) * CL - WIN
    rk_base = bh * T * D + win_begin * D
    for n0 in tl.static_range(0, WIN, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        wk = tl.load(RK + rk_base + offs_n[:, None] * D + offs_d[None, :])
        wv = tl.load(RV + rk_base + offs_n[:, None] * D + offs_d[None, :])
        qk = tl.dot(q, tl.trans(wk)) * scale  # [CL, BLOCK_N]
        cmask = offs_n[None, :] > (offs_cl[:, None] + (WIN - CL))
        qk = tl.where(cmask, -float("inf"), qk)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(wv.dtype), wv)
        m_i = m_ij

    o = acc / l_i[:, None]
    tl.store(L + (bh * n_qchunks + ci) * CL + offs_cl, m_i + tl.log(l_i))
    tl.store(
        O + (bh * n_qchunks + ci) * CL * D + offs_cl[:, None] * D + offs_d[None, :],
        o.to(O.dtype.element_ty),
    )


# ======================================================================
# PHASE 2 backward kernels
# ======================================================================
@triton.jit
def _phase2_bwd_preprocess(O, DO, DELTA, n_qchunks, CL: tl.constexpr, D: tl.constexpr):
    """delta[r] = sum_d O[r,d] * dO[r,d]  -- the softmax-jacobian row term."""
    ci = tl.program_id(0)
    bh = tl.program_id(1)
    offs_cl = tl.arange(0, CL)
    offs_d = tl.arange(0, D)
    base = (bh * n_qchunks + ci) * CL * D
    o = tl.load(O + base + offs_cl[:, None] * D + offs_d[None, :]).to(tl.float32)
    do = tl.load(DO + base + offs_cl[:, None] * D + offs_d[None, :]).to(tl.float32)
    tl.store(DELTA + (bh * n_qchunks + ci) * CL + offs_cl, tl.sum(o * do, axis=1))


# the state segment materialises [CL,M] + [M,D] fp32 tiles -- SRAM-heavy at
# large M (e.g. M=256), so num_stages is capped low and D is in the key.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s) for w in (2, 4, 8) for s in (1, 2)
    ],
    key=["n_qchunks", "T", "M", "D"],
)
@triton.jit
def _phase2_bwd_dq_dbuck(
    Q,
    BK,
    BV,
    BIAS,
    RK,
    RV,
    DO,
    L,
    DELTA,
    DQ,
    DBK,
    DBV,
    scale,
    n_qchunks,
    T,
    FRONT: tl.constexpr,
    CL: tl.constexpr,
    M: tl.constexpr,
    WIN: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # one program owns query chunk ci: dQ[ci] (state + window) and the
    # per-chunk bucket grads dBK[ci], dBV[ci]. The window grad is left to
    # _phase2_bwd_draw -- here the window only feeds dQ.
    ci = tl.program_id(0)
    bh = tl.program_id(1)
    offs_cl = tl.arange(0, CL)
    offs_d = tl.arange(0, D)

    qbase = (bh * n_qchunks + ci) * CL * D
    q = tl.load(Q + qbase + offs_cl[:, None] * D + offs_d[None, :])
    do = tl.load(DO + qbase + offs_cl[:, None] * D + offs_d[None, :])
    l_i = tl.load(L + (bh * n_qchunks + ci) * CL + offs_cl)  # [CL] fp32
    delta = tl.load(DELTA + (bh * n_qchunks + ci) * CL + offs_cl)  # [CL] fp32

    dq = tl.zeros([CL, D], tl.float32)

    # --- segment 1: compressed state -- tiled over M (BLOCK_M slots at a time)
    # so the [CL,M] + [M,D] fp32 tiles stay in SRAM even at large M (e.g. 256).
    base = (bh * n_qchunks + ci) * M * D
    for m0 in tl.static_range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        bk = tl.load(BK + base + offs_m[:, None] * D + offs_d[None, :])
        bv = tl.load(BV + base + offs_m[:, None] * D + offs_d[None, :])
        bias = tl.load(BIAS + ci * M + offs_m).to(tl.float32)
        qk = tl.dot(q, tl.trans(bk)) * scale + bias[None, :]  # [CL, BLOCK_M]
        p = tl.exp(qk - l_i[:, None])  # 0 at -inf bias
        dp = tl.dot(do, tl.trans(bv))  # [CL, BLOCK_M] fp32
        ds = p * (dp - delta[:, None])  # [CL, BLOCK_M] fp32
        dbv = tl.dot(tl.trans(p.to(do.dtype)), do)  # P^T @ dO -> [BLOCK_M, D]
        dbk = tl.dot(tl.trans(ds.to(q.dtype)), q) * scale  # dS^T @ Q -> [BLOCK_M, D]
        dq += tl.dot(ds.to(bk.dtype), bk) * scale  # dS @ BK -> [CL, D]
        tl.store(
            DBK + base + offs_m[:, None] * D + offs_d[None, :],
            dbk.to(DBK.dtype.element_ty),
        )
        tl.store(
            DBV + base + offs_m[:, None] * D + offs_d[None, :],
            dbv.to(DBV.dtype.element_ty),
        )

    # --- segment 2: sliding window (raw tokens), causal -> dQ only ---
    win_begin = FRONT + (ci + 1) * CL - WIN
    rk_base = bh * T * D + win_begin * D
    for n0 in tl.static_range(0, WIN, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        wk = tl.load(RK + rk_base + offs_n[:, None] * D + offs_d[None, :])
        wv = tl.load(RV + rk_base + offs_n[:, None] * D + offs_d[None, :])
        qk = tl.dot(q, tl.trans(wk)) * scale
        cmask = offs_n[None, :] > (offs_cl[:, None] + (WIN - CL))
        qk = tl.where(cmask, -float("inf"), qk)
        p = tl.exp(qk - l_i[:, None])
        dp = tl.dot(do, tl.trans(wv))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(wk.dtype), wk) * scale

    tl.store(
        DQ + qbase + offs_cl[:, None] * D + offs_d[None, :], dq.to(DQ.dtype.element_ty)
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in (2, 4, 8)
        for s in (2, 3, 4)
    ],
    key=["n_qchunks", "T", "D"],
)
@triton.jit
def _phase2_bwd_draw(
    Q,
    RK,
    RV,
    DO,
    L,
    DELTA,
    DRK,
    DRV,
    scale,
    n_qchunks,
    T,
    FRONT: tl.constexpr,
    CL: tl.constexpr,
    NB: tl.constexpr,
    WIN: tl.constexpr,
    D: tl.constexpr,
):
    # one program owns one CL-aligned raw column gcol = program_id(0) + 1
    # (FRONT == WIN => win_begin(ci) = (ci+1)*CL, so windows are CL-aligned and
    # column gcol is covered by query chunks ci in [gcol-NB, gcol-1]).
    col_pid = tl.program_id(0)
    bh = tl.program_id(1)
    gcol = col_pid + 1
    offs_cl = tl.arange(0, CL)
    offs_d = tl.arange(0, D)

    rbase = bh * T * D + gcol * CL * D
    wk = tl.load(RK + rbase + offs_cl[:, None] * D + offs_d[None, :])
    wv = tl.load(RV + rbase + offs_cl[:, None] * D + offs_d[None, :])

    dwk = tl.zeros([CL, D], tl.float32)
    dwv = tl.zeros([CL, D], tl.float32)

    ci_lo = tl.maximum(gcol - NB, 0)
    ci_hi = tl.minimum(gcol - 1, n_qchunks - 1)
    for ci in range(ci_lo, ci_hi + 1):
        qbase = (bh * n_qchunks + ci) * CL * D
        q = tl.load(Q + qbase + offs_cl[:, None] * D + offs_d[None, :])
        do = tl.load(DO + qbase + offs_cl[:, None] * D + offs_d[None, :])
        l_i = tl.load(L + (bh * n_qchunks + ci) * CL + offs_cl)
        delta = tl.load(DELTA + (bh * n_qchunks + ci) * CL + offs_cl)

        qk = tl.dot(q, tl.trans(wk)) * scale  # [CL, CL]
        # window-local position of this column's tokens for query chunk ci,
        # then the same offset causal relation the forward uses.
        nloc = (gcol - ci - 1) * CL + offs_cl[None, :]
        cmask = nloc > (offs_cl[:, None] + (WIN - CL))
        qk = tl.where(cmask, -float("inf"), qk)
        p = tl.exp(qk - l_i[:, None])  # [CL, CL]
        dwv += tl.dot(tl.trans(p.to(do.dtype)), do)  # P^T @ dO
        dp = tl.dot(do, tl.trans(wv))  # [CL, CL]
        ds = p * (dp - delta[:, None])
        dwk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale  # dS^T @ Q

    tl.store(
        DRK + rbase + offs_cl[:, None] * D + offs_d[None, :],
        dwk.to(DRK.dtype.element_ty),
    )
    tl.store(
        DRV + rbase + offs_cl[:, None] * D + offs_d[None, :],
        dwv.to(DRV.dtype.element_ty),
    )


# ======================================================================
# PHASE 2 autograd Function -- makes chunked_attention_forward differentiable
# ======================================================================
class _Phase2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, front, win, scale
    ):
        assert front == win, (
            "PHASE 2 backward assumes FRONT == WIN (CL-aligned windows); "
            f"got front={front}, win={win}"
        )
        q_chunks = q_chunks.contiguous()
        buck_k, buck_v = buck_k.contiguous(), buck_v.contiguous()
        buck_bias = buck_bias.contiguous()
        raw_k, raw_v = raw_k.contiguous(), raw_v.contiguous()
        BH, n_q, CL, D = q_chunks.shape
        M = buck_k.shape[2]
        T = raw_k.shape[1]
        o = torch.empty_like(q_chunks)
        L = torch.empty(BH, n_q, CL, device=q_chunks.device, dtype=torch.float32)
        _phase2_fwd_kernel[(n_q, BH)](
            q_chunks,
            buck_k,
            buck_v,
            buck_bias,
            raw_k,
            raw_v,
            o,
            L,
            scale,
            n_q,
            T,
            FRONT=front,
            CL=CL,
            M=M,
            WIN=win,
            D=D,
            BLOCK_N=CL,
        )
        ctx.save_for_backward(q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, o, L)
        ctx.front, ctx.win, ctx.scale = front, win, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, o, L = ctx.saved_tensors
        front, win, scale = ctx.front, ctx.win, ctx.scale
        BH, n_q, CL, D = q_chunks.shape
        M = buck_k.shape[2]
        T = raw_k.shape[1]
        NB = win // CL
        do = do.contiguous()

        delta = torch.empty(BH, n_q, CL, device=q_chunks.device, dtype=torch.float32)
        _phase2_bwd_preprocess[(n_q, BH)](o, do, delta, n_q, CL=CL, D=D)

        dq = torch.empty_like(q_chunks)
        dbk = torch.empty_like(buck_k)
        dbv = torch.empty_like(buck_v)
        _phase2_bwd_dq_dbuck[(n_q, BH)](
            q_chunks,
            buck_k,
            buck_v,
            buck_bias,
            raw_k,
            raw_v,
            do,
            L,
            delta,
            dq,
            dbk,
            dbv,
            scale,
            n_q,
            T,
            FRONT=front,
            CL=CL,
            M=M,
            WIN=win,
            D=D,
            BLOCK_N=CL,
            BLOCK_M=min(M, 128),
        )

        # raw columns [0, CL) are never inside a PHASE 2 window -> stay zero.
        drk = torch.zeros_like(raw_k)
        drv = torch.zeros_like(raw_v)
        n_cols = n_q + NB - 1
        _phase2_bwd_draw[(n_cols, BH)](
            q_chunks,
            raw_k,
            raw_v,
            do,
            L,
            delta,
            drk,
            drv,
            scale,
            n_q,
            T,
            FRONT=front,
            CL=CL,
            NB=NB,
            WIN=win,
            D=D,
        )

        return (dq, dbk, dbv, None, drk, drv, None, None, None)


def chunked_attention_forward(
    q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, front, win, scale
):
    """PHASE 2, differentiable. Shapes:
       q_chunks [BH, n_q, CL, D] | buck_{k,v} [BH, n_q, M, D] | buck_bias [n_q, M]
       raw_{k,v} [BH, T, D].  Returns o [BH, n_q, CL, D].
    Grads flow to q_chunks / buck_k / buck_v / raw_k / raw_v (buck_bias is a mask)."""
    return _Phase2.apply(
        q_chunks, buck_k, buck_v, buck_bias, raw_k, raw_v, front, win, scale
    )
