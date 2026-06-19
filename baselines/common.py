#!/usr/bin/env python
"""
baselines/common.py — shared plumbing for the three router baselines.

All three baselines (Hybrid LLM, RouteLLM, Mozannar-Sontag) consume the SAME
unified per-sample prediction files that step6/step7 use, and emit a routing
cost-quality curve in the SAME shape, so their curves overlay directly on
step6's Pareto plot.

FEATURE CONTRACT  (identical to src/step7_learned_router.py)
  small_*: [{"id","label","pred","prob", "emb":[..768..]?}]   RoBERTa (+ optional emb)
  large_*: [{"id","label","pred"}]                            GPT-5.4 direct judge
  - inner-joined on integer `id`
  - `prob` = P(label==1) from the SLM; `entropy(prob)` is the uncertainty scalar
  - if EVERY small row carries `emb`, features = [emb | prob | entropy]; else
    features = [prob | entropy] (scalar-only). Merge emb first with src/merge_emb.py.

TRAIN on val, EVAL on test. Never train on test.

CURVE CONVENTION  (identical to step6/step7)
  x = fraction of samples routed to the LLM (a cost proxy)
  y = macro-F1 (%) of the combined prediction (SLM by default, LLM where routed)
  Each baseline produces an *ordering* of test samples (route highest-priority
  first); curve_from_order() turns that ordering into the y-vs-x curve.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- #
# IO + features
# --------------------------------------------------------------------------- #
def load(path: str) -> dict:
    """Load a unified preds json keyed by integer id."""
    return {int(r["id"]): r for r in json.loads(Path(path).read_text())}


def entropy(p1: np.ndarray) -> np.ndarray:
    """Binary entropy of [p1, 1-p1]; higher = more uncertain."""
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def build(small_path: str, large_path: str):
    """Inner-join small/large preds; return the common feature/label bundle.

    Returns
    -------
    feats : (n, d)  [emb | prob | entropy]  or  [prob | entropy]  if no emb
    y     : (n,)    gold label
    sp    : (n,)    SLM prediction
    lp    : (n,)    LLM prediction
    prob  : (n,)    SLM P(label==1)
    has_emb : bool  whether emb was present on every row
    """
    S, L = load(small_path), load(large_path)
    ids = sorted(set(S) & set(L))
    if not ids:
        raise SystemExit(f"no overlapping ids between {small_path} and {large_path}")
    y = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"] for i in ids])
    lp = np.array([L[i]["pred"] for i in ids])
    prob = np.array([float(S[i].get("prob", S[i].get("prob_fake", 0.5))) for i in ids])
    scal = np.column_stack([prob, entropy(prob)])
    if all("emb" in S[i] for i in ids):
        emb = np.array([S[i]["emb"] for i in ids], float)
        feats = np.column_stack([emb, scal])
        has_emb = True
    else:
        feats, has_emb = scal, False
    return feats, y, sp, lp, prob, has_emb


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return f1_score(y, p, average="macro") * 100.0


def combine(sp: np.ndarray, lp: np.ndarray, route_mask: np.ndarray) -> np.ndarray:
    """route_mask[i]=True -> trust LLM for sample i, else SLM."""
    out = sp.copy()
    out[route_mask] = lp[route_mask]
    return out


def curve_from_order(order: np.ndarray, y, sp, lp, points: int = 21):
    """Route the first k samples in `order` to the LLM as k grows; return the
    fractions and macro-F1 curve. `order` lists sample indices, highest routing
    priority first (this is exactly step7's learned-router curve construction)."""
    n = len(y)
    fr = np.linspace(0, 1, points)
    curve = []
    for f in fr:
        k = int(round(f * n))
        mask = np.zeros(n, bool)
        mask[order[:k]] = True
        curve.append(macro_f1(y, combine(sp, lp, mask)))
    return fr, np.array(curve)


def apgr(fr: np.ndarray, curve: np.ndarray, f1_small: float, f1_large: float) -> float:
    """Average Performance Gap Recovered = normalized area under the
    call-performance curve (RouteLLM's APGR; == AUC of your Pareto curve).
    pgr(f) = (curve(f) - weak) / (strong - weak), averaged over the budget axis.
    Uses strong=max endpoint, weak=min endpoint so it's well-defined even when
    all-LLM < all-SLM (the 'LLM is not uniformly better' regime)."""
    strong, weak = max(f1_small, f1_large), min(f1_small, f1_large)
    if strong - weak < 1e-9:
        return float("nan")
    pgr = (curve - weak) / (strong - weak)
    return float(np.trapz(pgr, fr) / (fr[-1] - fr[0]))


def cpt(fr: np.ndarray, curve: np.ndarray, f1_small: float, f1_large: float,
        target_pct: float) -> float:
    """Call-Performance Threshold: smallest LLM-call fraction reaching
    target_pct% of the performance gap. NaN if never reached."""
    strong, weak = max(f1_small, f1_large), min(f1_small, f1_large)
    if strong - weak < 1e-9:
        return float("nan")
    pgr = (curve - weak) / (strong - weak) * 100.0
    hit = np.where(pgr >= target_pct)[0]
    return float(fr[hit[0]]) if len(hit) else float("nan")


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_curve(out_dir: str, fname: str, payload: dict):
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    p = odir / fname
    p.write_text(json.dumps(payload, indent=2))
    return p


def report_budget_points(fr, curve, f1_small, f1_large, name, tag):
    """Pretty-print the standard summary block shared by all baselines."""
    best_i = int(np.argmax(curve))
    print(f"\n[{name}] {tag}")
    print(f"  all-SLM={f1_small:.2f}  all-LLM={f1_large:.2f}")
    print(f"  peak={curve[best_i]:.2f} @ {fr[best_i]*100:.0f}% LLM budget")
    a = apgr(fr, curve, f1_small, f1_large)
    print(f"  APGR={a:.3f}  "
          f"CPT(50%)={cpt(fr, curve, f1_small, f1_large, 50)*100 if not np.isnan(cpt(fr,curve,f1_small,f1_large,50)) else float('nan'):.0f}%  "
          f"CPT(90%)={cpt(fr, curve, f1_small, f1_large, 90)*100 if not np.isnan(cpt(fr,curve,f1_small,f1_large,90)) else float('nan'):.0f}%")
    npts = len(fr)
    for f in (0.05, 0.1, 0.2, 0.3):
        i = int(round(f * (npts - 1)))
        print(f"    budget {int(f*100):>2}% -> {curve[i]:.2f}")
    return a
