#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src_v3/crc.py — Conformal Risk Control for the arbitration threshold.

BORROWED MODULE: Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei,
Schuster — ICLR 2024, arXiv:2208.02814), the generalisation of split
conformal prediction to any bounded monotone risk.

WHY IT FITS (reports/08): v2's "no-harm" came from a heuristic u>0 filter and
a bootstrap gate. CRC upgrades that into a distribution-free finite-sample
guarantee: pick the adoption threshold tau on VAL so that

    E[ false-adoption rate on new disagreements ] <= alpha,

where a false adoption = "arbiter score > tau, we adopt the LLM label, and the
LLM is wrong" (i.e., exactly the harm event that killed v1 on GossipCop).

Setup. For each val disagreement i we have an arbiter score s_i and the harm
indicator h_i = 1[LLM wrong on i]. The per-sample loss at threshold tau,
l_i(tau) = 1[s_i > tau] * h_i, is in [0,1] and monotone non-increasing in tau,
so the CRC bound applies:

    tau_hat = inf{ tau :  (n/(n+1)) * Rhat(tau) + B/(n+1)  <= alpha },  B = 1.

Guarantee (exchangeability of val/test disagreements): E[R(tau_hat)] <= alpha.

Notes for the paper:
  * alpha is a *user-facing risk knob* — we report the whole alpha sweep, not
    one magic value. Small alpha reproduces v2's "structurally no-harm" mode
    (adopt almost nothing on GossipCop); larger alpha trades bounded harm for
    recovered gain.
  * The controlled quantity is the false-adoption RATE among disagreements.
    Multiply by the class-weighted disagreement mass to translate into a
    macro-F1 harm bound (see pipeline.py output field 'harm_bound_f1').
  * Exchangeability caveat: Weibo21 val/test is a temporal split — report the
    realised test risk next to alpha (it is an *experiment*, cf. reports/02 P1).
"""
from __future__ import annotations

import numpy as np


def crc_threshold(scores, harms, alpha, B=1.0):
    """Smallest (most permissive) tau with CRC-bounded false-adoption risk.

    scores: (n,) arbiter scores on VAL disagreements (higher = trust LLM more)
    harms:  (n,) 1 if adopting LLM on that sample would be wrong
    alpha:  target risk level
    Returns (tau, info). tau=+inf means even adopting nothing misses alpha
    (never happens with B=1 unless alpha < B/(n+1))."""
    scores = np.asarray(scores, dtype=np.float64)
    harms = np.asarray(harms, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf"), dict(n=0, note="no calibration disagreements")
    # candidate taus = observed scores (risk only changes there) + sentinel
    cand = np.unique(scores)
    best = float("inf")
    for tau in cand[::-1]:  # descending: stop at smallest feasible
        adopted = scores > tau
        rhat = float((harms * adopted).sum()) / n
        bound = (n / (n + 1)) * rhat + B / (n + 1)
        if bound <= alpha:
            best = float(tau)
        else:
            break
    info = dict(n=n, alpha=alpha,
                adopt_rate=float((scores > best).mean()) if np.isfinite(best) else 0.0)
    return best, info


def crc_sweep(scores_val, harms_val, scores_test=None, harms_test=None,
              alphas=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    """Alpha sweep; if test arrays given, also report realised test risk."""
    rows = []
    for a in alphas:
        tau, info = crc_threshold(scores_val, harms_val, a)
        row = dict(alpha=a, tau=None if not np.isfinite(tau) else tau,
                   val_adopt_rate=info.get("adopt_rate", 0.0))
        if scores_test is not None and np.isfinite(tau):
            ad = np.asarray(scores_test) > tau
            row["test_adopt_rate"] = float(ad.mean())
            row["test_realized_risk"] = float((np.asarray(harms_test) * ad).sum() / max(len(ad), 1))
        rows.append(row)
    return rows
