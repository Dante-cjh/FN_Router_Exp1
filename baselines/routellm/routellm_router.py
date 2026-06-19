#!/usr/bin/env python
"""
RouteLLM baseline  (Ong et al., 2024, arXiv:2406.18665)
Official code: https://github.com/lm-sys/RouteLLM

WHAT THE PAPER DOES
  Binary route between a STRONG (expensive) and WEAK (cheap) model. A win-rate
  model P_theta(win_strong | q) estimates the probability the strong model gives
  the better answer; route weak if P < alpha, else strong. Sweep alpha -> the
  cost-quality curve. Pure pre-generation routing. Reported with APGR (area under
  the call-performance curve) and CPT(x%) (calls needed to recover x% of the gap)
  -- the same summary statistics as your Pareto AUC.

  Of the four official routers, the two cheap-to-port ones are:
    - BERT classifier   : contextual embedding + classification head -> P(win_strong)
    - Matrix factorisation (the paper's BEST router): a bilinear score
        s(M, q) = <v_M, W q> + b_M ,  P(win_strong) = sigmoid(s(strong,q) - s(weak,q))

ADAPTATION TO THIS REPO
  Preference labels become "which model was right":
      win_strong = 1   if  LLM right AND SLM wrong      (strong strictly wins)
      win_strong = 0   if  SLM right AND LLM wrong      (strong strictly loses)
      win_strong = 0.5 if  both right OR both wrong     (tie)
  Trained with soft-target BCE. Both variants run on the frozen RoBERTa
  embedding + [prob, entropy] scalars. The BERT variant consumes the features
  directly; the MF variant treats SLM/LLM as the two "models" with learned
  embeddings and a learned projection of the query features.

  ROUTING -> CURVE. Route to the LLM the highest-P(win_strong) samples first
  (== sweeping alpha downward), then reuse step6/step7's curve construction.

  Paradigm note (for the paper): RouteLLM predicts a WIN-RATE (relative quality)
  and, like Hybrid LLM, assumes upgrading is beneficial; it never models a
  harm-leg or asks whether the gain beats the base-rate / rescue asymmetry.

TRAIN on val, EVAL on test.

USAGE
  python -m baselines.routellm.routellm_router \
    --small_val  outputs/preds/weibo21_roberta_val.json \
    --large_val  outputs/preds/weibo21_gpt54_val.json \
    --small_test outputs/preds/weibo21_roberta.json \
    --large_test outputs/preds/weibo21_gpt54.json \
    --variant mf --name weibo21 \
    --out outputs/diagnostic/weibo21/baselines
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
except Exception as e:  # pragma: no cover
    raise SystemExit("RouteLLM baseline needs PyTorch (already required for "
                     f"step3 RoBERTa training). Import failed: {e}")


def win_label(sp, lp, y):
    """Soft win_strong target in {0, 0.5, 1}."""
    s_right, l_right = (sp == y), (lp == y)
    lab = np.full(len(y), 0.5)
    lab[l_right & ~s_right] = 1.0      # strong strictly wins
    lab[s_right & ~l_right] = 0.0      # strong strictly loses
    return lab.astype(np.float32)


class BertHead(nn.Module):
    """Linear win-rate head on frozen features (RouteLLM 'bert' variant port)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)            # logit of P(win_strong)


class MF(nn.Module):
    """Matrix-factorisation router (RouteLLM 'mf' variant port).
    Two model embeddings v_S, v_L (rank r); query projected to rank r.
    logit = s(strong,q) - s(weak,q) = <v_L - v_S, Wq> + (b_L - b_S)."""
    def __init__(self, d, rank=32):
        super().__init__()
        self.proj = nn.Linear(d, rank, bias=False)
        self.vS = nn.Parameter(torch.randn(rank) * 0.01)
        self.vL = nn.Parameter(torch.randn(rank) * 0.01)
        self.bS = nn.Parameter(torch.zeros(1))
        self.bL = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.proj(x)
        sL = (q * self.vL).sum(-1) + self.bL
        sS = (q * self.vS).sum(-1) + self.bS
        return sL - sS                            # logit of P(win_strong)


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_router(model, Xtr, ytr, epochs=200, lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    for k in ("small_val", "large_val", "small_test", "large_test", "name", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--variant", choices=["bert", "mf"], default="mf")
    ap.add_argument("--rank", type=int, default=32, help="MF latent rank")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--points", type=int, default=21)
    args = ap.parse_args()

    Xtr, ytr, sptr, lptr, _, he1 = build(args.small_val, args.large_val)
    Xte, yte, spte, lpte, _, he2 = build(args.small_test, args.large_test)
    if Xtr.shape[1] != Xte.shape[1]:
        raise SystemExit("val/test feature dims differ — merge emb consistently "
                         "(src/merge_emb.py) or not at all.")

    Xtr, Xte = standardize(Xtr, Xte)
    wtr = win_label(sptr, lptr, ytr)

    d = Xtr.shape[1]
    model = MF(d, args.rank) if args.variant == "mf" else BertHead(d)
    model = train_router(model, Xtr, wtr, epochs=args.epochs)

    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy()

    # route highest P(win_strong) first (== sweeping alpha down)
    order = np.argsort(-score)
    fr, curve = curve_from_order(order, yte, spte, lpte, args.points)
    f1_small, f1_large = macro_f1(yte, spte), macro_f1(yte, lpte)

    tag = (f"RouteLLM [{args.variant}{'' if args.variant=='bert' else f', rank={args.rank}'}] "
           f"feats={'emb+scalar' if he2 else 'scalar-only'}")
    a = report_budget_points(fr, curve, f1_small, f1_large, args.name, tag)

    p = write_curve(args.out, f"routing_routellm_{args.variant}.json", {
        "method": f"RouteLLM/{args.variant}", "name": args.name,
        "variant": args.variant, "rank": args.rank if args.variant == "mf" else None,
        "n": int(len(yte)), "fractions": fr.tolist(),
        "all_small_f1": f1_small, "all_large_f1": f1_large,
        "router_f1": curve.tolist(), "apgr": a,
        "feats": "emb+scalar" if he2 else "scalar-only",
    })
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
