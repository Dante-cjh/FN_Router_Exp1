#!/usr/bin/env python
"""
Hybrid LLM baseline  (Ding et al., ICLR 2024, arXiv:2404.14618)
Official code: https://github.com/m365-core/hybrid_llm_routing

WHAT THE PAPER DOES
  Learns a BERT-style router r:X->[0,1] that scores how "easy" a query is, i.e.
  how likely the SMALL model is good enough that we can keep it on the small
  model. High score -> keep small; below threshold -> route to large. Pure
  pre-generation routing (the router never calls an LLM). The label is built
  from a quality gap H(x) = q(S(x)) - q(L(x)) (q = BART score in the paper):
    r_det   : hard label  y = 1[q(S) >= q(L)]                       (BCE)
    r_prob  : soft label  y = Pr[H >= 0]   (10 samples / model)      (BCE)
    r_trans : soft label  y = Pr[H >= -t], relaxation t > 0 chosen by
              grid search to MAXIMISE average pairwise label spread (paper Eq.3)
              -- rebalances labels when L >> S and positives are scarce.

ADAPTATION TO THIS REPO (classification, frozen RoBERTa features)
  Our task is binary classification, so quality degrades to discrete correctness:
      H(x) = 1[SLM correct] - 1[LLM correct]  in {-1, 0, +1}.
  Predictions are deterministic, so the 10x sampling of r_prob is unnecessary and
  r_det is the natural port. We still implement r_trans's relaxation idea on the
  discrete gap, which becomes a meaningful knob:
      t = 0  -> label = 1[H >= 0]  = 1[SLM not worse than LLM]   (= r_det)
      t = 1  -> label = 1[H >= -1] = 1[SLM correct OR LLM wrong] (very permissive)
  The router is a logistic head on the frozen RoBERTa penultimate embedding +
  [prob, entropy] scalars -- the cheap pre-generation feature set, mirroring the
  paper's "encoder forward pass is negligible vs LLM decode" assumption.

  ROUTING SCORE -> CURVE.  The router scores P[keep-small]. We route to the LLM
  the samples with the LOWEST keep-small score first (most likely the SLM is NOT
  good enough), exactly the paper's threshold sweep, and reuse step6/step7's
  curve construction so the curve overlays on the Pareto plot.

  Paradigm note (for the paper): Hybrid LLM predicts ANSWERABILITY (is S good
  enough), not the net-gain band; it has no notion of rescue asymmetry or
  upgrade-harm. That contrast is the point of running it.

TRAIN on val, EVAL on test.

USAGE
  python -m baselines.hybrid_llm.hybrid_llm_router \
    --small_val  outputs/preds/weibo21_roberta_val.json \
    --large_val  outputs/preds/weibo21_gpt54_val.json \
    --small_test outputs/preds/weibo21_roberta.json \
    --large_test outputs/preds/weibo21_gpt54.json \
    --variant r_trans --name weibo21 \
    --out outputs/diagnostic/weibo21/baselines
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))  # baselines/ on path
from common import (build, macro_f1, curve_from_order, report_budget_points,
                    write_curve)  # noqa: E402


def keep_small_label(sp, lp, y, t):
    """y_easy = 1[H >= -t], H = 1[SLM right] - 1[LLM right]. t in {0,1}."""
    H = (sp == y).astype(int) - (lp == y).astype(int)
    return (H >= -t).astype(int)


def pick_t_by_spread(sp, lp, y, candidates=(0, 1)):
    """Paper Eq.3 in spirit: choose the relaxation t that maximises the average
    pairwise spread of labels (== 2 * p * (1-p) for binary labels, maximal at
    the most balanced split). Keeps the label signal from collapsing when one
    leg dominates."""
    best_t, best_spread = candidates[0], -1.0
    for t in candidates:
        lab = keep_small_label(sp, lp, y, t)
        p = lab.mean()
        spread = 2 * p * (1 - p)  # proportional to mean |y_i - y_j|
        if spread > best_spread:
            best_t, best_spread = t, spread
    return best_t


def main():
    ap = argparse.ArgumentParser()
    for k in ("small_val", "large_val", "small_test", "large_test", "name", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--variant", choices=["r_det", "r_trans"], default="r_trans")
    ap.add_argument("--t", type=int, default=-1,
                    help="relaxation for r_trans; -1 = auto-pick by label spread")
    ap.add_argument("--points", type=int, default=21)
    args = ap.parse_args()

    Xtr, ytr, sptr, lptr, _, he1 = build(args.small_val, args.large_val)
    Xte, yte, spte, lpte, _, he2 = build(args.small_test, args.large_test)
    if Xtr.shape[1] != Xte.shape[1]:
        raise SystemExit("val/test feature dims differ — merge emb consistently "
                         "(src/merge_emb.py) or not at all.")

    # ---- label construction ----
    if args.variant == "r_det":
        t = 0
    else:
        t = args.t if args.t >= 0 else pick_t_by_spread(sptr, lptr, ytr)
    ytr_easy = keep_small_label(sptr, lptr, ytr, t)

    if ytr_easy.sum() in (0, len(ytr_easy)):
        raise SystemExit(f"degenerate 'easy' label (all={ytr_easy.mean():.0f}); "
                         "try the other variant / a different t.")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr_easy)
    keep_small = clf.predict_proba(sc.transform(Xte))[:, 1]   # P[S is good enough]

    # ---- route LOWEST keep-small first (most likely S is NOT enough) ----
    order = np.argsort(keep_small)                            # ascending
    fr, curve = curve_from_order(order, yte, spte, lpte, args.points)
    f1_small, f1_large = macro_f1(yte, spte), macro_f1(yte, lpte)

    tag = (f"Hybrid LLM [{args.variant}{'' if args.variant=='r_det' else f', t={t}'}] "
           f"feats={'emb+scalar' if he2 else 'scalar-only'}")
    a = report_budget_points(fr, curve, f1_small, f1_large, args.name, tag)

    p = write_curve(args.out, f"routing_hybridllm_{args.variant}.json", {
        "method": f"HybridLLM/{args.variant}", "name": args.name,
        "variant": args.variant, "relaxation_t": int(t),
        "n": int(len(yte)), "fractions": fr.tolist(),
        "all_small_f1": f1_small, "all_large_f1": f1_large,
        "router_f1": curve.tolist(), "apgr": a,
        "feats": "emb+scalar" if he2 else "scalar-only",
    })
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
