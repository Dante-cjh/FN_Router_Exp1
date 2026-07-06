#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src_v3/pipeline.py — RA^3 end-to-end: pre-route -> call -> ARBITRATE.

The Round-3 main experiment (reports/08 §6 M3). Decision structure:

  Stage-1 (pre-generation, SLM features only): net-utility dual head (v2 asset)
          decides WHETHER to spend an LLM call:   u(x) = w[1-sp]*p_gain - w[sp]*p_harm,
          route u>0 by descending u up to budget b (budget = upper bound).
          NOTE the gain-term weight uses w[1-sp], a purely pre-hoc quantity
          (on a realised gain the LLM's label is exactly 1-sp), so Stage-1
          never peeks at the LLM output.
  Stage-2 (post-generation, on ROUTED samples only, zero extra LLM cost):
          the rationale-aware arbiter (src_v3/arbiter.py) decides WHOSE label
          to keep. Policies compared at every budget:
            swallow   adopt LLM label on all routed samples  (v1/v2 rule)
            arb_f1    adopt only if arbiter score > tau tuned on val macro-F1
            arb_crc   adopt only if score > tau chosen by Conformal Risk
                      Control at level alpha (distribution-free harm bound)

Main claims this script tests:
  (1) same budget, arb_* >= swallow on both datasets (arbitration is a free
      Pareto improvement given the call was already paid for);
  (2) GossipCop curve turns from monotonically losing into rising;
  (3) arb_crc keeps min(curve) >= all-SLM floor with a certificate, replacing
      v2's heuristic gate.

USAGE (from Router_Exp1/):
  python -m src_v3.pipeline --name weibo21   --lang zh --features emb
  python -m src_v3.pipeline --name gossipcop --lang en --features emb --alpha 0.2
Output: outputs_v3/pipeline/<name>__<features>/pipeline.json (+ .png if matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src_v3.arbiter import arbitrated_pred, fit_arbiter, tune_tau_valf1
from src_v3.common import (build_split, bootstrap_delta, class_weights_macro_f1,
                           macro_f1)
from src_v3.crc import crc_threshold

BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]


# ----------------------------------------------------------------- stage-1
def stage1_features(split):
    return np.hstack([split["emb"],
                      np.column_stack([split["prob"], split["ent"]])])


def fit_head(X, y, seed=0):
    base = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=seed))
    if len(np.unique(y)) < 2:
        return None
    return CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X, y)


def net_utility(val, te, seed=0):
    """v2 net-utility, pre-hoc: p_gain/p_harm heads (calibrated) + class weights."""
    Xv, Xt = stage1_features(val), stage1_features(te)
    gain_v = ((val["lp"] == val["y"]) & (val["sp"] != val["y"])).astype(int)
    harm_v = ((val["sp"] == val["y"]) & (val["lp"] != val["y"])).astype(int)
    hg, hh = fit_head(Xv, gain_v, seed), fit_head(Xv, harm_v, seed)
    pg = hg.predict_proba(Xt)[:, 1] if hg else np.zeros(len(te["y"]))
    ph = hh.predict_proba(Xt)[:, 1] if hh else np.zeros(len(te["y"]))
    w = class_weights_macro_f1(val["y"], val["sp"])
    w_gain = np.array([w[1 - s] for s in te["sp"]])   # LLM label on a gain == 1-sp
    w_harm = np.array([w[s] for s in te["sp"]])
    return w_gain * pg - w_harm * ph


def routed_sets(u, budgets, n, relax=False):
    """strict: v2 semantics (only u>0, budget = upper bound).
    relax:  rank by u but NO u>0 filter — spend the budget. Rationale
    (smoke-test finding): with a post-hoc arbiter bounding the downside,
    the conservative pre-hoc gate wastes rescueable samples; arbitration
    should let the pre-router route MORE, not less."""
    order = np.argsort(-u)
    eligible = order if relax else order[u[order] > 0]
    return {b: eligible[: min(int(round(b * n)), len(eligible))] for b in budgets}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--features", default="emb",
                    choices=["frugal", "dict", "emb", "enc", "full"])
    ap.add_argument("--scope", default="disagree", choices=["disagree", "all"])
    ap.add_argument("--enc", default=None)
    ap.add_argument("--alpha", type=float, default=0.2, help="CRC risk level")
    ap.add_argument("--stage1", default="netutil", choices=["netutil", "entropy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out_root", default="outputs_v3/pipeline")
    args = ap.parse_args()

    val = build_split(args.name, args.lang, "val", args.enc)
    te = build_split(args.name, args.lang, "test", args.enc)
    n = len(te["y"])

    # ---- stage-1 ranking (pre-hoc) ----
    if args.stage1 == "netutil":
        u = net_utility(val, te, args.seed)
    else:                                   # entropy baseline: route most-uncertain
        u = te["ent"].copy()
    sel = routed_sets(u, BUDGETS, n)
    sel_rx = routed_sets(u, BUDGETS, n, relax=True)

    # ---- stage-2 arbiter (trained on val; scores are post-call quantities) ----
    score = fit_arbiter(val, args.features, args.scope, "logreg", 0.5, args.seed)
    s_val, s_te = score(val), score(te)
    mv = val["sp"] != val["lp"]
    tau_f1, _ = tune_tau_valf1(val, s_val)
    tau_crc, crc_info = crc_threshold(
        s_val[mv], (val["lp"][mv] != val["y"][mv]).astype(int), args.alpha)

    # ---- curves ----
    def combined(sel_idx, policy, tau=None):
        pred = te["sp"].copy()
        if policy.startswith("swallow"):
            pred[sel_idx] = te["lp"][sel_idx]
        else:
            m = np.zeros(n, bool)
            m[sel_idx] = True
            adopt = m & (te["sp"] != te["lp"]) & (s_te > tau)
            pred[adopt] = te["lp"][adopt]
        return pred

    curves, deltas = {}, {}
    for pol, tau, ss in [("swallow", None, sel), ("arb_f1", tau_f1, sel),
                         ("arb_crc", tau_crc, sel),
                         ("arb_relax", tau_f1, sel_rx),      # bold pre-route, arbiter as safety net
                         ("swallow_relax", None, sel_rx)]:   # control: bold WITHOUT arbiter
        curves[pol] = {}
        for b in BUDGETS:
            curves[pol][b] = round(macro_f1(te["y"], combined(ss[b], pol, tau)), 2)
    for b in [0.1, 0.2, 0.5, 1.0]:          # bootstrap the key claims at same cost
        deltas[str(b)] = dict(
            arb_vs_swallow=bootstrap_delta(
                te["y"], combined(sel[b], "arb_f1", tau_f1),
                combined(sel[b], "swallow"), args.n_boot),
            arbrelax_vs_swallow=bootstrap_delta(
                te["y"], combined(sel_rx[b], "arb_relax", tau_f1),
                combined(sel[b], "swallow"), args.n_boot))

    all_slm = round(macro_f1(te["y"], te["sp"]), 2)
    orc = te["sp"].copy()
    m_or = (te["lp"] == te["y"]) & (te["sp"] != te["y"])
    orc[m_or] = te["lp"][m_or]
    res = dict(
        name=args.name, stage1=args.stage1, features=args.features,
        alpha=args.alpha, tau_valF1=tau_f1,
        tau_crc=(None if not np.isfinite(tau_crc) else tau_crc),
        crc_info=crc_info,
        endpoints=dict(all_slm=all_slm,
                       all_llm=round(macro_f1(te["y"], te["lp"]), 2),
                       oracle_full=round(macro_f1(te["y"], orc), 2)),
        curves=curves,
        peak={p: dict(f1=max(c.values()), budget=max(c, key=c.get))
              for p, c in curves.items()},
        floor_check={p: dict(min_f1=min(c.values()),
                             no_harm=bool(min(c.values()) >= all_slm - 1e-9))
                     for p, c in curves.items()},
        boot_arbf1_minus_swallow=deltas,
        n_routed_eligible=int((u > 0).sum()) if args.stage1 == "netutil" else None,
    )

    tag = f"{args.name}__{args.features}"
    out = f"{args.out_root}/{tag}"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/pipeline.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [b * 100 for b in BUDGETS]
        plt.figure(figsize=(7, 4.5))
        for pol, style in [("swallow", "--"), ("arb_f1", "-"), ("arb_crc", "-."),
                           ("arb_relax", "-"), ("swallow_relax", ":")]:
            plt.plot(xs, [curves[pol][b] for b in BUDGETS], style, marker="o",
                     label=pol, lw=1.8, ms=3.5)
        plt.axhline(all_slm, color="gray", lw=1, label=f"all-SLM {all_slm}")
        plt.axhline(res["endpoints"]["all_llm"], color="brown", lw=1,
                    label=f"all-LLM {res['endpoints']['all_llm']}")
        plt.xlabel("% routed to LLM (cost)")
        plt.ylabel("macro-F1")
        plt.title(f"RA$^3$ {args.name} (stage1={args.stage1}, feat={args.features})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out}/pareto.png", dpi=160)
        print(f"[pipeline] wrote {out}/pareto.png")
    except Exception as e:  # plotting is optional
        print(f"[pipeline] plot skipped: {e}")
    print(f"[pipeline] wrote {out}/pipeline.json")


if __name__ == "__main__":
    main()
