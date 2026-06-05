"""Step 5 -- The diagnostic: error overlap + oracle router upper bound.

Answers the central question for the collaborative-routing project:

    Is there enough *complementarity* between the small model (RoBERTa), the
    small+LLM-advisor system (ARG), and the large model (GPT-5.4 direct) for a
    router to beat the single best model?

For each model we have a unified prediction file:
    [{"id":..., "label": <0/1>, "pred": <0/1, -1 means wrong-by-default>}, ...]

Metrics produced
----------------
* Per model: accuracy, macro-F1.
* Pairwise error Jaccard: |E_i ∩ E_j| / |E_i ∪ E_j|  (1 = identical mistakes,
  0 = disjoint mistakes -> maximal routing headroom).
* Oracle router upper bound: accuracy of a perfect router that, per sample,
  picks any model that is correct = fraction of samples where >=1 model is right.
  Computed for every pair and for the full set.
* Routing gain = oracle - best single-model accuracy.

Verdict rule (from the project brief)
-------------------------------------
    gain >= 5.0 pts  -> routing direction is well-founded
    gain <  1.0 pts  -> pivot
    otherwise        -> weak / inconclusive

Outputs: a markdown report, figures (PNG), and a machine-readable summary.json.

Usage:
    python -m src.step5_diagnostic \
        --dataset GossipCop \
        --pred RoBERTa=outputs/preds/gossipcop_roberta.json \
        --pred ARG=outputs/preds/gossipcop_arg.json \
        --pred GPT-5.4=outputs/preds/gossipcop_gpt54.json \
        --outdir outputs/diagnostic/gossipcop
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_pred_file(path: str) -> Dict[str, int]:
    """Return {str(id): correct(0/1)} plus the raw label/pred maps."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    label_map, pred_map = {}, {}
    for r in rows:
        i = str(r["id"])
        label_map[i] = int(r["label"])
        pred_map[i] = int(r["pred"])
    return label_map, pred_map


def align(models: Dict[str, str]):
    """Load all models, intersect ids, return ids, labels, and correctness."""
    labels_per, preds_per = {}, {}
    for name, path in models.items():
        lm, pm = load_pred_file(path)
        labels_per[name] = lm
        preds_per[name] = pm

    id_sets = [set(pm.keys()) for pm in preds_per.values()]
    common = set.intersection(*id_sets) if id_sets else set()
    sizes = {n: len(pm) for n, pm in preds_per.items()}
    print(f"[diag] per-model sizes: {sizes} | common ids: {len(common)}")
    if not common:
        raise SystemExit("No overlapping ids across prediction files.")

    # Sanity: gold labels must agree across files on common ids.
    ids = sorted(common, key=lambda x: (len(x), x))
    ref = next(iter(labels_per.values()))
    mismatches = 0
    for i in ids:
        gold_vals = {labels_per[n][i] for n in models}
        if len(gold_vals) > 1:
            mismatches += 1
    if mismatches:
        print(f"[diag] WARNING: {mismatches} ids have disagreeing gold labels "
              "across files (check you used the same test split).")

    labels = np.array([ref[i] for i in ids], dtype=int)
    correct = {}  # name -> bool array (pred == gold)
    for n in models:
        correct[n] = np.array([preds_per[n][i] == labels_per[n][i] for i in ids],
                              dtype=bool)
    return ids, labels, correct, sizes


def macro_f1(correct_unused, labels, preds):
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average="macro")


def compute(models: Dict[str, str]):
    ids, labels, correct, sizes = align(models)
    names = list(models.keys())
    n = len(ids)

    # Per-model accuracy + macro-F1 (rebuild preds from correctness+labels).
    acc, f1 = {}, {}
    # We need raw preds for F1; reload to keep it simple.
    raw_preds = {}
    for name, path in models.items():
        _, pm = load_pred_file(path)
        raw_preds[name] = np.array([pm[i] for i in ids], dtype=int)
    from sklearn.metrics import f1_score
    for name in names:
        acc[name] = float(correct[name].mean())
        f1[name] = float(f1_score(labels, raw_preds[name], average="macro"))

    best_single = max(acc, key=acc.get)
    best_single_acc = acc[best_single]

    # Pairwise error Jaccard + pairwise oracle.
    pair_jaccard, pair_oracle, pair_breakdown = {}, {}, {}
    for a, b in itertools.combinations(names, 2):
        Ea = ~correct[a]
        Eb = ~correct[b]
        inter = np.logical_and(Ea, Eb).sum()
        union = np.logical_or(Ea, Eb).sum()
        pair_jaccard[(a, b)] = float(inter / union) if union else 0.0
        oracle = float(np.logical_or(correct[a], correct[b]).mean())
        pair_oracle[(a, b)] = oracle
        pair_breakdown[(a, b)] = {
            "both_correct": int(np.logical_and(correct[a], correct[b]).sum()),
            f"only_{a}": int(np.logical_and(correct[a], ~correct[b]).sum()),
            f"only_{b}": int(np.logical_and(~correct[a], correct[b]).sum()),
            "both_wrong": int(np.logical_and(~correct[a], ~correct[b]).sum()),
            "pair_best_single_acc": float(max(acc[a], acc[b])),
            "oracle_gain_pts": (oracle - max(acc[a], acc[b])) * 100,
        }

    # Full oracle (any model correct).
    any_correct = np.zeros(n, dtype=bool)
    for name in names:
        any_correct |= correct[name]
    full_oracle = float(any_correct.mean())
    full_gain_pts = (full_oracle - best_single_acc) * 100

    # Verdict per project rule.
    if full_gain_pts >= 5.0:
        verdict = "FOUNDED — routing direction has a learning-theoretic basis (gain ≥ 5 pts)."
    elif full_gain_pts < 1.0:
        verdict = "PIVOT — complementarity is negligible (gain < 1 pt)."
    else:
        verdict = "WEAK — gain in the 1–5 pt grey zone; routing is plausible but not compelling."

    summary = {
        "n_test": n,
        "models": names,
        "per_model_sizes": sizes,
        "accuracy": acc,
        "macro_f1": f1,
        "best_single_model": best_single,
        "best_single_acc": best_single_acc,
        "pair_error_jaccard": {f"{a}|{b}": v for (a, b), v in pair_jaccard.items()},
        "pair_oracle_acc": {f"{a}|{b}": v for (a, b), v in pair_oracle.items()},
        "pair_breakdown": {f"{a}|{b}": v for (a, b), v in pair_breakdown.items()},
        "full_oracle_acc": full_oracle,
        "full_oracle_gain_pts": full_gain_pts,
        "verdict": verdict,
    }
    return summary, names, acc, pair_jaccard, pair_oracle, full_oracle, best_single_acc


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_accuracy_bars(names, acc, pair_oracle, full_oracle, best_single_acc,
                       dataset, path):
    labels = list(names) + [f"oracle:{a}+{b}" for (a, b) in pair_oracle] + ["oracle:ALL"]
    vals = [acc[n] for n in names] + [pair_oracle[k] for k in pair_oracle] + [full_oracle]
    colors = ["#4C78A8"] * len(names) + ["#F58518"] * len(pair_oracle) + ["#54A24B"]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.5))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(best_single_acc, ls="--", color="grey",
               label=f"best single = {best_single_acc:.3f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(min(vals) - 0.05, 1.02)
    ax.set_title(f"{dataset}: single models vs oracle routers")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_jaccard_heatmap(names, pair_jaccard, dataset, path):
    k = len(names)
    M = np.full((k, k), np.nan)
    for i in range(k):
        M[i, i] = 1.0
    for (a, b), v in pair_jaccard.items():
        i, j = names.index(a), names.index(b)
        M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(1.6 * k + 1, 1.6 * k))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    for i in range(k):
        for j in range(k):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=10)
    ax.set_title(f"{dataset}: error-set Jaccard\n(low = complementary mistakes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(summary, dataset, fig_paths, path):
    s = summary
    lines = []
    lines.append(f"# Collaborative-Routing Diagnostic — {dataset}\n")
    lines.append(f"- Test samples (common ids): **{s['n_test']}**")
    lines.append(f"- Models: {', '.join(s['models'])}\n")

    lines.append("## Verdict\n")
    lines.append(f"> **{s['verdict']}**\n")
    lines.append(f"- Best single model: **{s['best_single_model']}** "
                 f"(acc = {s['best_single_acc']*100:.2f}%)")
    lines.append(f"- Oracle router (any model correct): "
                 f"**{s['full_oracle_acc']*100:.2f}%**")
    lines.append(f"- **Routing headroom = {s['full_oracle_gain_pts']:+.2f} "
                 f"percentage points**\n")
    lines.append("Decision rule: ≥ 5 pts → well-founded; < 1 pt → pivot; "
                 "1–5 pts → grey zone.\n")

    lines.append("## Per-model performance\n")
    lines.append("| Model | Accuracy | Macro-F1 | Test size |")
    lines.append("|---|---|---|---|")
    for n in s["models"]:
        lines.append(f"| {n} | {s['accuracy'][n]*100:.2f}% | "
                     f"{s['macro_f1'][n]*100:.2f}% | {s['per_model_sizes'][n]} |")
    lines.append("")

    lines.append("## Pairwise error overlap & oracle\n")
    lines.append("| Pair | Error Jaccard | Pair oracle acc | Pair best single | Pair gain (pts) |")
    lines.append("|---|---|---|---|---|")
    for key in s["pair_oracle_acc"]:
        j = s["pair_error_jaccard"][key]
        o = s["pair_oracle_acc"][key]
        bd = s["pair_breakdown"][key]
        lines.append(f"| {key} | {j:.3f} | {o*100:.2f}% | "
                     f"{bd['pair_best_single_acc']*100:.2f}% | "
                     f"{bd['oracle_gain_pts']:+.2f} |")
    lines.append("")
    lines.append("*Error Jaccard = |both wrong| / |at least one wrong|. "
                 "Lower means the two models fail on different samples — exactly "
                 "the complementarity a router can exploit.*\n")

    lines.append("## Where the routing gain comes from (per pair)\n")
    for key, bd in s["pair_breakdown"].items():
        a, b = key.split("|")
        lines.append(f"**{a} vs {b}** (n={s['n_test']}): both correct "
                     f"{bd['both_correct']}, only {a} {bd[f'only_{a}']}, "
                     f"only {b} {bd[f'only_{b}']}, both wrong {bd['both_wrong']}.")
    lines.append("")

    lines.append("## Figures\n")
    for fp in fig_paths:
        rel = os.path.basename(fp)
        lines.append(f"![{rel}]({rel})\n")

    lines.append("## How to read this\n")
    lines.append("- **World A (routing useless):** error Jaccard ≈ 1 and oracle "
                 "gain ≈ 0 — models make the *same* mistakes.")
    lines.append("- **World B (routing gold):** error Jaccard ≈ 0 and oracle gain "
                 "is large — disjoint mistakes, a perfect router approaches 100%.")
    lines.append("- The real datasets sit between these; this report locates "
                 "exactly where, and whether the gap justifies the routing thesis.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    metavar="NAME=PATH",
                    help="repeatable, e.g. --pred RoBERTa=outputs/preds/x.json")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    models = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"--pred must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = path

    os.makedirs(args.outdir, exist_ok=True)
    (summary, names, acc, pair_jaccard, pair_oracle,
     full_oracle, best_single_acc) = compute(models)

    fig1 = os.path.join(args.outdir, "accuracy_vs_oracle.png")
    fig2 = os.path.join(args.outdir, "error_jaccard_heatmap.png")
    plot_accuracy_bars(names, acc, pair_oracle, full_oracle, best_single_acc,
                       args.dataset, fig1)
    plot_jaccard_heatmap(names, pair_jaccard, args.dataset, fig2)

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_report(summary, args.dataset, [fig1, fig2],
                 os.path.join(args.outdir, "report.md"))

    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"dataset           : {args.dataset}")
    print(f"best single model : {summary['best_single_model']} "
          f"({best_single_acc*100:.2f}%)")
    print(f"oracle (ALL)      : {full_oracle*100:.2f}%")
    print(f"routing headroom  : {summary['full_oracle_gain_pts']:+.2f} pts")
    print(f"verdict           : {summary['verdict']}")
    print(f"outputs           : {args.outdir}/  (report.md, summary.json, *.png)")


if __name__ == "__main__":
    main()
