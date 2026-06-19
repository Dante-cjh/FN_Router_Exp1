#!/usr/bin/env python
"""
Mozannar & Sontag (2020) NAIVE learning-to-defer baseline
Paper: "Consistent Estimators for Learning to Defer to an Expert", ICML 2020,
       PMLR v119:7076-7087  (arXiv:2006.01862)
Official code: https://github.com/clinicalml/learn-to-defer

WHAT THE PAPER DOES
  Jointly learns a classifier h:X->Y and a deferrer r:X->{0,1} by augmenting the
  label space to Y u {⊥}. With K classes there are K+1 score functions g_y, g_⊥.
  The CONSISTENT surrogate (their L_CE, alpha=1) is:

    L_CE = -log softmax(g)[y]  -  1[m == y] * log softmax(g)[⊥]

  where m is the EXPERT's prediction. The first term is ordinary cross-entropy on
  the true label; the second pushes mass to ⊥ ONLY when the expert is correct.
  Test rule: predict h(x)=argmax_{y in Y} g_y(x); defer when max_y g_y(x) <= g_⊥(x).
  Consistency holds ONLY at alpha=1 (this naive version). No class weighting, no
  profitability / regime gate -- exactly the intended ablation.

ADAPTATION TO THIS REPO
  Expert m = LLM (GPT-5.4) prediction. K=2 classes -> a 3-logit head on the
  frozen RoBERTa embedding + [prob, entropy] scalars. We train the naive
  L_CE (alpha=1).

  ROUTING -> CURVE. The naive deferrer is r(x)=1[g_⊥ >= max_y g_y]; to trace a
  cost-quality curve we sweep a threshold tau on the margin (g_⊥ - max_y g_y),
  deferring the highest-margin samples first (tau-> -inf defers all, +inf defers
  none; tau=0 is the naive argmax deferrer). IMPORTANT: on NON-deferred samples
  the system prediction is the L2D classifier h(x) (the jointly-learned head),
  NOT the original RoBERTa pred -- this is faithful to the method (the system is
  the pair (h, r)). So the 0%-routed endpoint is macro-F1 of h, which may differ
  slightly from the standalone RoBERTa number.

  Ablation narrative (for the paper): naive L_CE has no class weighting (hurts
  macro-F1 on imbalanced GossipCop), no calibration, is not realizable-consistent
  (Mozannar et al. 2023), and has no profitability gate (it defers whenever the
  expert is "probably right", ignoring whether the SLM was already right). Each
  missing piece maps onto one component of your upgraded method.

TRAIN on val, EVAL on test.

USAGE
  python -m baselines.mozannar_sontag.l2d_router \
    --small_val  outputs/preds/weibo21_roberta_val.json \
    --large_val  outputs/preds/weibo21_gpt54_val.json \
    --small_test outputs/preds/weibo21_roberta.json \
    --large_test outputs/preds/weibo21_gpt54.json \
    --name weibo21 --out outputs/diagnostic/weibo21/baselines
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (build, macro_f1, curve_from_order, report_budget_points,
                    write_curve)  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise SystemExit("Mozannar-Sontag baseline needs PyTorch (already required "
                     f"for step3 RoBERTa training). Import failed: {e}")

K = 2  # number of true classes (real/fake); the head has K+1 logits


class DeferNet(nn.Module):
    """K+1 logit head (K classes + ⊥) on frozen features."""
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                  nn.Linear(hidden, K + 1))

    def forward(self, x):
        return self.net(x)                    # (n, K+1); index K is ⊥


def l_ce(logits, y, m, alpha=1.0):
    """Naive consistent surrogate (alpha=1). m = expert (LLM) prediction.
    L = -(alpha*1[m=y] + 1[m!=y]) * log p[y]  -  1[m=y] * log p[⊥]
    At alpha=1 the first factor is 1 -> standard CE on the true label."""
    logp = F.log_softmax(logits, dim=1)
    expert_right = (m == y).float()
    w = alpha * expert_right + (1.0 - expert_right)
    ce_true = -w * logp[torch.arange(len(y)), y]
    defer = -expert_right * logp[:, K]
    return (ce_true + defer).mean()


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    for k in ("small_val", "large_val", "small_test", "large_test", "name", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="keep at 1.0 for the consistent naive estimator; "
                         "any other value is a non-consistent variant.")
    ap.add_argument("--slm_floor", choices=["frozen", "h"], default="frozen",
                    help="non-deferred prediction used for the PRIMARY curve. "
                         "'frozen' (default) = frozen RoBERTa pred -> comparable "
                         "to Hybrid LLM/RouteLLM/step7 (isolates deferral). "
                         "'h' = jointly-learned classifier -> method-faithful but "
                         "folds in the val head-refit gain. Both are saved.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--points", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xtr, ytr, sptr, lptr, _, he1 = build(args.small_val, args.large_val)
    Xte, yte, spte, lpte, _, he2 = build(args.small_test, args.large_test)
    if Xtr.shape[1] != Xte.shape[1]:
        raise SystemExit("val/test feature dims differ — merge emb consistently "
                         "(src/merge_emb.py) or not at all.")

    Xtr, Xte = standardize(Xtr, Xte)
    mtr = lptr  # expert = LLM prediction

    torch.manual_seed(args.seed)
    model = DeferNet(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    mt = torch.tensor(mtr, dtype=torch.long)
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = l_ce(model(Xt), yt, mt, alpha=args.alpha)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        g = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
    class_logits = g[:, :K]
    h_pred = class_logits.argmax(1)                 # L2D classifier prediction
    defer_margin = g[:, K] - class_logits.max(1)    # g_⊥ - max_y g_y

    # most defer-worthy first (defer = route to LLM)
    order = np.argsort(-defer_margin)

    # TWO curves on the SAME defer ordering:
    #   h-floor      : non-deferred prediction = the jointly-learned h(x).
    #                  Faithful to the (h, r) system, but folds in the gain from
    #                  REFITTING a head on val (see diagnose_slm_floor.py): on
    #                  weibo21 that head jumps 76->~88 purely from train/test
    #                  shift, which is NOT a deferral gain.
    #   frozen-floor : non-deferred prediction = the frozen RoBERTa pred (sp),
    #                  exactly the floor Hybrid LLM / RouteLLM / step7 use. This
    #                  ISOLATES the deferral contribution and is the apples-to-
    #                  apples curve for the comparison plot.
    fr, curve_h = curve_from_order(order, yte, h_pred, lpte, args.points)
    _,  curve_fz = curve_from_order(order, yte, spte, lpte, args.points)

    # endpoints / reference numbers
    f1_h = macro_f1(yte, h_pred)            # h at 0% routed (h-floor endpoint)
    f1_large = macro_f1(yte, lpte)
    f1_slm = macro_f1(yte, spte)           # frozen RoBERTa (frozen-floor endpoint)
    naive_defer = defer_margin >= 0        # tau = 0 naive deferrer
    naive_pred = h_pred.copy(); naive_pred[naive_defer] = lpte[naive_defer]
    f1_naive = macro_f1(yte, naive_pred)
    naive_frac = float(naive_defer.mean())

    primary = curve_fz if args.slm_floor == "frozen" else curve_h
    floor_f1 = f1_slm if args.slm_floor == "frozen" else f1_h

    tag = (f"Mozannar-Sontag naive L_CE (alpha={args.alpha}, floor={args.slm_floor}) "
           f"feats={'emb+scalar' if he2 else 'scalar-only'}")
    a = report_budget_points(fr, primary, floor_f1, f1_large, args.name, tag)
    print(f"  L2D classifier h macro-F1 = {f1_h:.2f}   "
          f"frozen RoBERTa = {f1_slm:.2f}  "
          f"(gap {f1_h-f1_slm:+.2f} = val head-refit, NOT deferral; "
          f"see diagnose_slm_floor.py)")
    print(f"  naive argmax deferrer (tau=0): defers {naive_frac*100:.0f}%% -> "
          f"macro-F1 (h-floor) {f1_naive:.2f}")
    print(f"  curves: frozen-floor peak={max(curve_fz):.2f}  "
          f"h-floor peak={max(curve_h):.2f}  (primary = {args.slm_floor})")

    p = write_curve(args.out, "routing_mozannar_sontag.json", {
        "method": "MozannarSontag/naive_LCE", "name": args.name,
        "alpha": args.alpha, "slm_floor": args.slm_floor,
        "n": int(len(yte)), "fractions": fr.tolist(),
        # `all_small_f1`/`router_f1` follow the chosen floor so the comparison
        # plot picks up the comparable (frozen) curve by default.
        "all_small_f1": floor_f1,
        "router_f1": primary.tolist(), "apgr": a,
        "all_large_f1": f1_large,
        # both curves + both floors kept for transparency / ablation:
        "router_f1_frozen_floor": curve_fz.tolist(),
        "router_f1_h_floor": curve_h.tolist(),
        "frozen_roberta_f1": f1_slm, "h_classifier_f1": f1_h,
        "naive_defer_fraction": naive_frac, "naive_defer_f1": f1_naive,
        "feats": "emb+scalar" if he2 else "scalar-only",
    })
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
