#!/usr/bin/env python
"""
src_v2/uncertainty_diagnostic.py — Round-2 experiment, step B (the go/kill test).

Takes the MC-Dropout uncertainties from step A and asks the three questions from
reports/04_uncertainty_routing_plan.md §3 that decide whether the whole
uncertainty-decoupling route is worth building:

  (a) DECOUPLING USEFUL?  Does U_epi separate the gain band (LLM-rescues:
      LLM-right & SLM-wrong) from the both-wrong band BETTER than U_tot
      (= the signal the old confidence router already used)?  AUC comparison.
  (b) ALEATORIC GATES WASTE?  Is U_ale systematically HIGHER on the both-wrong
      band than on the gain band (so "don't route high-U_ale" saves budget)?
  (c) DOES IT CASH OUT?  A quick cost-quality Pareto using U_epi as the routing
      score (plain, and gated by an epi>ale rule) — does it stay >= all-SLM in
      the low-budget regime where the old confidence curve fell below?

Bands (from v1 unified preds, inner-joined on id):
  both_correct | only_slm (= HARM band) | only_llm (= GAIN band) | both_wrong

ISOLATION: reads v1 preds (outputs/preds/*) + step-A uncertainties
(outputs_v2/uncertainty/*), writes only under outputs_v2/diagnostic/.

USAGE:
  python -m src_v2.uncertainty_diagnostic \
      --name weibo21 \
      --unc  outputs_v2/uncertainty/weibo21_test.json \
      --small outputs/preds/weibo21_roberta.json \
      --large outputs/preds/weibo21_gpt54.json \
      --out  outputs_v2/diagnostic/weibo21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)


def load_rows(path):
    return {int(r["id"]): r for r in json.loads(Path(path).read_text())}


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro") * 100.0


def safe_auc(label, score):
    """ROC-AUC that tolerates degenerate single-class subsets."""
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    if label.min() == label.max():
        return float("nan")
    return float(roc_auc_score(label, score))


def binary_entropy_from_prob(p1, eps=1e-12):
    p1 = np.clip(p1, eps, 1 - eps)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def pareto_curve(score, sp, lp, y, eligible=None, points=21):
    """Route top-k samples (highest score) to the LLM; adopt LLM label there.

    `eligible` (bool mask) optionally restricts which samples may EVER be routed
    (e.g. an epi>ale gate); ineligible samples keep the SLM label at any budget.
    Returns (fractions, macro_f1_list) where fraction = k / N (budget proxy).
    """
    n = len(y)
    order = np.argsort(-score)
    if eligible is not None:
        order = order[eligible[order]]            # keep only eligible, in score order
    fr = np.linspace(0, 1, points)
    out = []
    for f in fr:
        k = int(round(f * n))
        mask = np.zeros(n, bool)
        mask[order[:k]] = True
        comb = sp.copy()
        comb[mask] = lp[mask]
        out.append(macro_f1(y, comb))
    return fr.tolist(), out


def band_stats(arr, bands):
    return {b: {"n": int(m.sum()),
                "mean": float(arr[m].mean()) if m.any() else float("nan"),
                "median": float(np.median(arr[m])) if m.any() else float("nan")}
            for b, m in bands.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--unc", required=True, help="step-A uncertainty json (test split)")
    ap.add_argument("--small", required=True, help="v1 SLM preds (RoBERTa)")
    ap.add_argument("--large", required=True, help="v1 LLM preds (GPT-5.4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=21)
    args = ap.parse_args()

    U = load_rows(args.unc)
    S = load_rows(args.small)
    L = load_rows(args.large)
    ids = sorted(set(U) & set(S) & set(L))
    if not ids:
        raise SystemExit("no overlapping ids across unc/small/large — check files")
    n = len(ids)

    y = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"] for i in ids])
    lp = np.array([L[i]["pred"] for i in ids])
    s_prob = np.array([float(S[i].get("prob", 0.5)) for i in ids])
    U_tot = np.array([U[i]["U_tot"] for i in ids])
    U_ale = np.array([U[i]["U_ale"] for i in ids])
    U_epi = np.array([U[i]["U_epi"] for i in ids])
    conf_single = binary_entropy_from_prob(s_prob)   # the OLD confidence signal

    s_ok, l_ok = (sp == y), (lp == y)
    bands = {
        "both_correct": s_ok & l_ok,
        "only_slm_HARM": s_ok & ~l_ok,
        "only_llm_GAIN": ~s_ok & l_ok,
        "both_wrong": ~s_ok & ~l_ok,
    }
    gain = bands["only_llm_GAIN"]
    bothwrong = bands["both_wrong"]

    # ---------- (a) decoupling usefulness ----------
    # gain vs both_wrong (the crux: tell "LLM can rescue" from "nobody can")
    gw = gain | bothwrong
    auc_gw = {
        "U_epi":       safe_auc(gain[gw], U_epi[gw]),
        "U_tot":       safe_auc(gain[gw], U_tot[gw]),
        "conf_single": safe_auc(gain[gw], conf_single[gw]),
    }
    # gain vs everything else (deployment view: rank gain to the top)
    auc_all = {
        "U_epi":       safe_auc(gain, U_epi),
        "U_tot":       safe_auc(gain, U_tot),
        "conf_single": safe_auc(gain, conf_single),
    }

    # ---------- (b) does aleatoric flag the irreducible band? ----------
    ale_stats = band_stats(U_ale, bands)
    epi_stats = band_stats(U_epi, bands)
    tot_stats = band_stats(U_tot, bands)

    # ---------- (c) cash-out Pareto ----------
    all_small = macro_f1(y, sp)
    all_large = macro_f1(y, lp)
    fr, pe_epi = pareto_curve(U_epi, sp, lp, y, points=args.points)
    _,  pe_tot = pareto_curve(U_tot, sp, lp, y, points=args.points)
    _,  pe_cnf = pareto_curve(conf_single, sp, lp, y, points=args.points)
    # parameter-free gate: only route where reducible uncertainty dominates
    eligible = U_epi > U_ale
    _,  pe_gate = pareto_curve(U_epi, sp, lp, y, eligible=eligible, points=args.points)

    def at(curve, f):
        i = int(round(f * (args.points - 1)))
        return curve[i]

    low_budget = [0.05, 0.1, 0.2]
    epi_floor_ok = all(at(pe_gate, f) >= all_small - 1e-9 for f in low_budget)

    # ---------- verdict ----------
    go_a = (not np.isnan(auc_gw["U_epi"])) and auc_gw["U_epi"] > auc_gw["U_tot"]
    go_b = ale_stats["both_wrong"]["mean"] > ale_stats["only_llm_GAIN"]["mean"]
    go_c = epi_floor_ok
    verdict = ("GO — decoupling helps on all three checks" if (go_a and go_b and go_c)
               else "PARTIAL — see per-check flags below"
               if (go_a or go_b or go_c)
               else "KILL — decoupling adds nothing here; fall back to net-utility dual head")

    result = {
        "name": args.name, "n": n,
        "all_small_f1": all_small, "all_large_f1": all_large,
        "band_sizes": {b: int(m.sum()) for b, m in bands.items()},
        "check_a_auc_gain_vs_bothwrong": auc_gw,
        "check_a_auc_gain_vs_rest": auc_all,
        "check_b_U_ale_by_band": ale_stats,
        "U_epi_by_band": epi_stats,
        "U_tot_by_band": tot_stats,
        "check_c_pareto": {
            "fractions": fr,
            "route_by_U_epi": pe_epi,
            "route_by_U_tot": pe_tot,
            "route_by_conf_single": pe_cnf,
            "route_by_U_epi_gated_epi_gt_ale": pe_gate,
            "n_eligible_epi_gt_ale": int(eligible.sum()),
        },
        "flags": {"go_a_epi_beats_tot": bool(go_a),
                  "go_b_ale_flags_bothwrong": bool(go_b),
                  "go_c_gated_keeps_floor": bool(go_c)},
        "verdict": verdict,
    }

    odir = Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "uncertainty_bands.json").write_text(json.dumps(result, indent=2))

    # ---------- console readout ----------
    print(f"\n===== [{args.name}] uncertainty-decoupling diagnostic (n={n}) =====")
    print(f"  bands: " + "  ".join(f"{b}={int(m.sum())}" for b, m in bands.items()))
    print(f"  (a) AUC gain-vs-bothwrong:  U_epi={auc_gw['U_epi']:.3f}  "
          f"U_tot={auc_gw['U_tot']:.3f}  conf_single={auc_gw['conf_single']:.3f}")
    print(f"      AUC gain-vs-rest:       U_epi={auc_all['U_epi']:.3f}  "
          f"U_tot={auc_all['U_tot']:.3f}  conf_single={auc_all['conf_single']:.3f}")
    print(f"  (b) mean U_ale  gain={ale_stats['only_llm_GAIN']['mean']:.4f}  "
          f"both_wrong={ale_stats['both_wrong']['mean']:.4f}  "
          f"(want both_wrong > gain)")
    print(f"  (c) all-SLM={all_small:.2f}  all-LLM={all_large:.2f}")
    for f in low_budget:
        print(f"      budget {int(f*100):>2}%  epi={at(pe_epi,f):.2f}  "
              f"epi_gated={at(pe_gate,f):.2f}  tot={at(pe_tot,f):.2f}  "
              f"conf={at(pe_cnf,f):.2f}")
    print(f"  flags: a(epi>tot)={go_a}  b(ale flags bothwrong)={go_b}  "
          f"c(gated>=floor)={go_c}")
    print(f"  -> {verdict}\n")

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # boxplot of U_epi and U_ale by band
        order = ["both_correct", "only_slm_HARM", "only_llm_GAIN", "both_wrong"]
        labels = ["both\ncorrect", "only-SLM\n(harm)", "only-LLM\n(gain)", "both\nwrong"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].boxplot([U_epi[bands[b]] for b in order], labels=labels, showfliers=False)
        ax[0].set_title(f"U_epi by band — {args.name}"); ax[0].set_ylabel("U_epi (nats)")
        ax[0].grid(alpha=.3)
        ax[1].boxplot([U_ale[bands[b]] for b in order], labels=labels, showfliers=False)
        ax[1].set_title(f"U_ale by band — {args.name}"); ax[1].set_ylabel("U_ale (nats)")
        ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(odir / "uncertainty_by_band.png", dpi=150)
        plt.close(fig)

        # cash-out Pareto
        plt.figure(figsize=(6, 4.2))
        plt.plot(fr, pe_epi, "-o", ms=3, label="route by U_epi")
        plt.plot(fr, pe_gate, "-s", ms=3, label="U_epi, gated (epi>ale)")
        plt.plot(fr, pe_tot, "--", label="route by U_tot")
        plt.plot(fr, pe_cnf, ":", label="conf (single-pass entropy)")
        plt.axhline(all_small, color="gray", lw=1, label="all-SLM floor")
        plt.scatter([0, 1], [all_small, all_large], zorder=5)
        plt.xlabel("fraction routed to LLM (cost proxy)"); plt.ylabel("macro-F1")
        plt.title(f"Uncertainty routing Pareto — {args.name}")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(odir / "uncertainty_pareto.png", dpi=150); plt.close()
        print(f"  wrote figures -> {odir}/uncertainty_by_band.png, uncertainty_pareto.png")
    except Exception as e:
        print(f"  (figures skipped: {e})")

    print(f"  wrote {odir/'uncertainty_bands.json'}")


if __name__ == "__main__":
    main()
