"""Parallel FlexAttention prefill for the monotone variant.

The monotone schedule is data-independent, so the whole chunk recurrence can be
replaced by a parallel pipeline:

    project  ->  dyadic-pyramid reduction  ->  one flex_attention call

The schedule is **token-based**: every bucket it produces is a dyadically
aligned contiguous interval of *tokens*, so the set of all possible bucket
summaries is exactly the token dyadic pyramid (raw per-token prepared K/V, then
token pairs, then quads, ...). It is built bottom up in ``log n`` parallel
reductions. Which pyramid nodes each query chunk reads is fixed by the schedule,
so it becomes a precomputed `BlockMask`.

This module operates on a `MonotoneKVMAttention` instance and shares all its
weights -- the recurrent `forward` stays as the decode path, `forward_flex`
(this module) is the parallel prefill / training path. Only the `flex_attention`
op itself is wrapped in `torch.compile`, per PyTorch's guidance; the BlockMask
and schedule tables are data-independent and cached per sequence length.

`chunk_len` here is ONLY the BSWA window block size; the pyramid and the
schedule are entirely token-based.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ..scheduler import intervals, simulate

# compile the flex_attention op only (not the surrounding model)
_flex_attention = torch.compile(flex_attention)


def _build_plan(attn, T, device):
    """Precompute everything that depends only on the schedule + sequence length:
    the token dyadic-pyramid layout, per-query-chunk visibility, the log-size
    bias, and the FlexAttention BlockMask. Pure integer work -- no tensor data.
    """
    cl = attn.chunk_len
    front = min(T, attn.bswa_len)
    SL = attn.sink_len
    n_q = (T - front) // cl
    if n_q == 0:
        return None  # whole sequence fits the warmup window; flex not needed

    # the schedule is token-based: it runs over the tokens that have exited the
    # window. trace[t-1] = the bucket-size partition of the first t tokens.
    n_state_tok = attn._bswa_begin(front + n_q * cl)
    trace = simulate(attn.scheduler, n_state_tok)

    # --- token dyadic pyramid: level L has n_state_tok // 2**L nodes ----------
    levels, node_off = [], []  # (level, n_nodes, span_tokens)
    m, span, L, off = n_state_tok, 1, 0, 0
    while True:
        levels.append((L, m, span))
        node_off.append(off)
        off += m
        if m == 1:
            break
        m, span, L = m // 2, span * 2, L + 1
    P = off
    node_span = torch.empty(P, dtype=torch.long)
    for L, nn, span in levels:
        node_span[node_off[L] : node_off[L] + nn] = span

    def node_id(t0, t1):
        span = t1 - t0
        assert span & (span - 1) == 0 and t0 % span == 0, (
            f"flex prefill needs dyadic buckets; schedule {attn.cfg.schedule!r} "
            f"produced non-dyadic interval [{t0}, {t1})"
        )
        return node_off[span.bit_length() - 1] + t0 // span

    # --- per-query-chunk visibility (which pyramid nodes / window) ------------
    node_visible = torch.zeros(n_q, P, dtype=torch.bool)
    window_begin = torch.zeros(n_q, dtype=torch.long)
    for ci in range(n_q):
        qe = min(T, front + (ci + 1) * cl)
        bswa_b = attn._bswa_begin(qe)
        window_begin[ci] = bswa_b
        for t0, t1 in intervals(trace[bswa_b - 1]):  # TOKEN intervals
            node_visible[ci, node_id(t0, t1)] = True

    KV = SL + P + T
    pyr_count = node_span.to(device)  # token count per pyramid node
    logsize = torch.zeros(KV, device=device)
    logsize[SL : SL + P] = pyr_count.float().log()  # +log|bucket| score bias
    node_visible = node_visible.to(device)
    window_begin = window_begin.to(device)
    Qn = T - front

    # --- FlexAttention BlockMask (data-independent -> built once, cached) -----
    def mask_mod(b, h, qi, kvi):
        c = qi // cl  # post-warmup query chunk
        aq = qi + front  # absolute query position
        is_sink = kvi < SL
        is_pyr = (kvi >= SL) & (kvi < SL + P)
        is_raw = kvi >= SL + P
        pyr_ok = node_visible[c, (kvi - SL).clamp(0, P - 1)]
        j = kvi - SL - P
        raw_ok = (j >= window_begin[c]) & (j <= aq)
        return is_sink | (is_pyr & pyr_ok) | (is_raw & raw_ok)

    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=Qn, KV_LEN=KV, device=str(device)
    )

    # score_mod built ONCE here (not per forward): a fresh closure each call
    # would churn torch.compile's guards and trigger recompiles.
    def score_mod(score, b, h, qi, kvi):
        return score + logsize[kvi]

    return {
        "cl": cl,
        "front": front,
        "SL": SL,
        "P": P,
        "n_state_tok": n_state_tok,
        "pyr_count": pyr_count,
        "block_mask": block_mask,
        "score_mod": score_mod,
    }


def _dyadic_pyramid(x_tok, n_state_tok):
    """Bottom-up dyadic pyramid over the first n_state_tok TOKENS.

    x_tok: [B, H, T, d]. Returns [B, H, P, d] of node sums, ordered level by
    level (level 0 = per-token raw, level 1 = token pairs, level 2 = quads, ...).
    """
    B, H, _, d = x_tok.shape
    lvl = x_tok[:, :, :n_state_tok]  # level 0: one token per node
    out = [lvl]
    m = n_state_tok
    while m > 1:
        m2 = m // 2
        lvl = lvl[:, :, : 2 * m2].view(B, H, m2, 2, d).sum(3)
        out.append(lvl)
        m = m2
    return torch.cat(out, dim=2)


def flex_forward(attn, x):
    """Parallel FlexAttention prefill for a `MonotoneKVMAttention` instance.

    Numerically equivalent to `attn.forward(x)` (the chunk recurrence), but with
    no Python loop: one token pyramid reduction + one flex_attention call.
    """
    B, T, _ = x.shape
    q, k, v, gate = attn.project_qkv(x)
    front = min(T, attn.bswa_len)

    # warmup region: while everything fits the BSWA window, plain causal attn
    warm = F.scaled_dot_product_attention(
        q[:, :, :front],
        k[:, :, :front] * attn._front_temp(),
        v[:, :, :front],
        is_causal=True,
    )
    if T <= front:
        return attn.c_proj(warm.transpose(1, 2).reshape(B, T, -1))

    cache = attn.__dict__.setdefault("_flex_plans", {})
    key = (T, str(x.device))
    plan = cache.get(key)
    if plan is None:
        plan = _build_plan(attn, T, x.device)
        cache[key] = plan

    SL, P = plan["SL"], plan["P"]
    n_state_tok = plan["n_state_tok"]

    # per-token prepared K / raw V -- exactly what each schedule bucket sums.
    # The merge gate (data-dependent token weighting) multiplies in here; it is
    # part of the *operation*, not the schedule.
    Kp = attn._prepare_state_k(k)
    Vp = v
    if attn.cfg.use_merge_gate:
        Kp, Vp = Kp * gate, Vp * gate

    # token dyadic pyramid: every possible bucket summary, in log(n) parallel steps
    pyr_k = _dyadic_pyramid(Kp, n_state_tok)
    pyr_v = _dyadic_pyramid(Vp, n_state_tok)

    # readout transforms (head temps baked into K_cat, value-mean into V_cat)
    state_t, front_t = attn._state_temp(), attn._front_temp()
    pyr_k = attn.ln_s_k(pyr_k) * state_t
    pyr_v = pyr_v / plan["pyr_count"].view(1, 1, P, 1)
    sink_k = attn.ln_s_k(attn._prepare_state_k(k[:, :, :SL])) * state_t
    sink_v = v[:, :, :SL]

    K_cat = torch.cat([sink_k, pyr_k, k * front_t], dim=2).contiguous()
    V_cat = torch.cat([sink_v, pyr_v, v], dim=2).contiguous()

    flex_out = _flex_attention(
        q[:, :, front:].contiguous(),
        K_cat,
        V_cat,
        score_mod=plan["score_mod"],
        block_mask=plan["block_mask"],
    )
    y = torch.cat([warm, flex_out], dim=2).transpose(1, 2).reshape(B, T, -1)
    return attn.c_proj(y)
