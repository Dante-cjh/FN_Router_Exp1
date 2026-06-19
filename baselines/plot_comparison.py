#!/usr/bin/env python
"""
plot_comparison.py — overlay ALL routing policies on one Pareto figure per
dataset, and emit a summary table (markdown + CSV).

Pulls curves from the existing pipeline outputs:
  step6  outputs/diagnostic/<ds>_llm_slm/routing_pareto.json
         -> random / conf_threshold (Chow) / oracle_frontier + endpoints
  step7  outputs/diagnostic/<ds>/routing_learned.json
         -> learned_f1  (YOUR method)
  baselines  outputs/diagnostic/<ds>/baselines/routing_*.json
         -> Hybrid LLM, RouteLLM-mf, RouteLLM-bert, Mozannar-Sontag (frozen floor)

All curves share the same 21-pt fraction grid (x = % routed to LLM); anything on
a different grid is linearly interpolated onto the step6 grid.

USAGE  (run from Router_Exp1/ root, after the baselines have been run)
  python -m baselines.plot_comparison \
    --datasets weibo21 gossipcop \
    --diag_root outputs/diagnostic \
    --out_dir   outputs/diagnostic/baseline_comparison

Outputs per dataset: <out_dir>/<ds>_pareto.png
Combined:            <out_dir>/comparison_all.png
Tables:              <out_dir>/summary.md , <out_dir>/summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[0]))
from common import apgr, cpt  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# label -> (source file rel to diag, key holding the curve, plot style dict)
# style: color, linestyle, linewidth, zorder, marker
def sources(ds):
    s6 = f"{ds}_llm_slm/routing_pareto.json"
    s7 = f"{ds}/routing_learned.json"
    bl = f"{ds}/baselines"
    return [
        # label,              file,                         curve-key,            style
        ("Oracle (upper bnd)", s6,                          "oracle_frontier_f1", dict(color="#9aa0a6", ls="--", lw=1.6, z=1)),
        ("Random",             s6,                          "random_f1",          dict(color="#cfcfcf", ls="-",  lw=1.2, z=1)),
        ("Chow (conf-thresh)", s6,                          "conf_threshold_f1",  dict(color="#8c6d31", ls="-",  lw=1.6, z=2)),
        ("Ours (step7 learned)", s7,                        "learned_f1",         dict(color="#111111", ls="-",  lw=2.8, z=6, marker="o", ms=3)),
        ("Hybrid LLM",         f"{bl}/routing_hybridllm_r_det.json", "router_f1", dict(color="#1f77b4", ls="-",  lw=1.8, z=3)),
        ("RouteLLM-mf",        f"{bl}/routing_routellm_mf.json",     "router_f1", dict(color="#2ca02c", ls="-",  lw=1.8, z=3)),
        ("RouteLLM-bert",      f"{bl}/routing_routellm_bert.json",   "router_f1", dict(color="#d62728", ls="-",  lw=1.8, z=4)),
        ("Mozannar-Sontag (frozen)", f"{bl}/routing_mozannar_sontag.json", "router_f1", dict(color="#9467bd", ls="-", lw=1.8, z=3)),
        # faint reference: M&S read on its own refit head (NOT comparable)
        ("M-S (h-floor, ref)", f"{bl}/routing_mozannar_sontag.json", "router_f1_h_floor", dict(color="#9467bd", ls=":", lw=1.2, z=2)),
    ]


def load_curve(diag_root, rel, key, grid):
    p = Path(diag_root) / rel
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if key not in d:
        return None
    fr = np.array(d["fractions"], float)
    c = np.array(d[key], float)
    # interpolate onto the common grid if needed
    if len(fr) != len(grid) or not np.allclose(fr, grid):
        c = np.interp(grid, fr, c)
    return c, d.get("all_small_f1"), d.get("all_large_f1")


def get_endpoints(diag_root, ds):
    p = Path(diag_root) / f"{ds}_llm_slm/routing_pareto.json"
    d = json.loads(p.read_text())
    return np.array(d["fractions"], float), d["all_small_f1"], d["all_large_f1"]


def budget_at(grid, curve, f):
    i = int(np.argmin(np.abs(grid - f)))
    return curve[i]


def plot_dataset(ax, diag_root, ds):
    grid, f1_small, f1_large = get_endpoints(diag_root, ds)
    rows = []
    for label, rel, key, st in sources(ds):
        got = load_curve(diag_root, rel, key, grid)
        if got is None:
            print(f"[plot] {ds}: missing {rel}::{key} — skipped")
            continue
        curve, _, _ = got
        ax.plot(grid * 100, curve, label=label, color=st["color"], ls=st["ls"],
                lw=st["lw"], zorder=st["z"],
                marker=st.get("marker"), markersize=st.get("ms", 0))
        a = apgr(grid, curve, f1_small, f1_large)
        rows.append({
            "dataset": ds, "method": label,
            "peak": round(float(np.max(curve)), 2),
            "peak_budget_%": int(round(grid[int(np.argmax(curve))] * 100)),
            "@5%": round(float(budget_at(grid, curve, 0.05)), 2),
            "@10%": round(float(budget_at(grid, curve, 0.10)), 2),
            "@20%": round(float(budget_at(grid, curve, 0.20)), 2),
            "@30%": round(float(budget_at(grid, curve, 0.30)), 2),
            "APGR": round(float(a), 3) if a == a else None,
        })
    # endpoint guide lines
    ax.axhline(f1_small, color="#888", ls=":", lw=1.0, zorder=0)
    ax.axhline(f1_large, color="#888", ls="-.", lw=1.0, zorder=0)
    ax.text(101, f1_small, f"all-SLM {f1_small:.1f}", va="center", fontsize=7, color="#555")
    ax.text(101, f1_large, f"all-LLM {f1_large:.1f}", va="center", fontsize=7, color="#555")
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.set_xlabel("% routed to LLM (cost proxy)")
    ax.set_ylabel("macro-F1")
    ax.grid(True, alpha=0.25)
    return rows


def write_tables(out_dir, rows):
    cols = ["dataset", "method", "peak", "peak_budget_%", "@5%", "@10%", "@20%", "@30%", "APGR"]
    csvp = Path(out_dir) / "summary.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    mdp = Path(out_dir) / "summary.md"
    lines = ["# Routing baseline comparison\n",
             "x = % routed to LLM, y = macro-F1. **Ours = step7 learned router.** "
             "Mozannar-Sontag shown on the *frozen* floor (comparable); its h-floor "
             "row in the figure is a non-comparable reference (folds in the val "
             "head-refit gain — see diagnose_slm_floor.py).\n"]
    cur = None
    for r in rows:
        if r["dataset"] != cur:
            cur = r["dataset"]
            lines.append(f"\n## {cur}\n")
            lines.append("| method | peak | @budget | @5% | @10% | @20% | @30% | APGR |")
            lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(f"| {r['method']} | {r['peak']} | {r['peak_budget_%']}% | "
                     f"{r['@5%']} | {r['@10%']} | {r['@20%']} | {r['@30%']} | {r['APGR']} |")
    Path(mdp).write_text("\n".join(lines) + "\n")
    return csvp, mdp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["weibo21", "gossipcop"])
    ap.add_argument("--diag_root", default="outputs/diagnostic")
    ap.add_argument("--out_dir", default="outputs/diagnostic/baseline_comparison")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_rows = []
    # per-dataset figures
    for ds in args.datasets:
        fig, ax = plt.subplots(figsize=(7, 5))
        rows = plot_dataset(ax, args.diag_root, ds)
        all_rows += rows
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        fig.tight_layout()
        outp = Path(args.out_dir) / f"{ds}_pareto.png"
        fig.savefig(outp, dpi=150); plt.close(fig)
        print(f"[plot] wrote {outp}")

    # combined side-by-side
    n = len(args.datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, ds in zip(axes[0], args.datasets):
        plot_dataset(ax, args.diag_root, ds)
    axes[0][-1].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    outp = Path(args.out_dir) / "comparison_all.png"
    fig.savefig(outp, dpi=150); plt.close(fig)
    print(f"[plot] wrote {outp}")

    csvp, mdp = write_tables(args.out_dir, all_rows)
    print(f"[plot] wrote {csvp}\n[plot] wrote {mdp}")


if __name__ == "__main__":
    main()
