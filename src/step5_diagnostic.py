"""Step 5 -- The diagnostic: error overlap + oracle router upper bound.

Answers the central question for the collaborative-routing project:

    Is there enough *complementarity* between the small model (RoBERTa), the
    small+LLM-advisor system (ARG), and the large model (GPT-5.4 direct) for a
    router to beat the single best model?

For each model we have a unified prediction file:
    [{"id":..., "label": <0/1>, "pred": <0/1, -1 means wrong-by-default>}, ...]

Metrics produced
----------------
* Per model: accuracy AND macro-F1.
* Pairwise error Jaccard: |E_i ∩ E_j| / |E_i ∪ E_j|  (1 = identical mistakes,
  0 = disjoint mistakes -> maximal routing headroom).
* Oracle router upper bound, reported in BOTH metrics:
    - oracle accuracy = fraction of samples where >=1 model is right.
    - oracle macro-F1 = macro-F1 of the oracle's per-sample predictions, where
      the oracle outputs the gold label whenever some model is correct and the
      flipped label otherwise. This makes the oracle "at least one correct"
      accounting *per class*, so a majority-class consensus can no longer inflate
      the headroom on imbalanced datasets (e.g. GossipCop).
* Routing headroom = oracle - best single model, in BOTH metrics.

PRIMARY metric = macro-F1.
On imbalanced data (GossipCop: real ≫ fake) accuracy is dominated by the
majority class, so both the per-model numbers and the "≥1 correct" oracle look
inflated. The verdict below is therefore driven by the **macro-F1** headroom;
the accuracy headroom is reported alongside it for reference only.

Verdict rule (from the project brief, applied to macro-F1)
----------------------------------------------------------
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

from sklearn.metrics import f1_score  # noqa: E402


def load_pred_file(path: str):
    """Return ({str(id): gold}, {str(id): pred})."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    label_map, pred_map = {}, {}
    for r in rows:
        i = str(r["id"])
        label_map[i] = int(r["label"])
        pred_map[i] = int(r["pred"])
    return label_map, pred_map


def align(models: Dict[str, str], strict_split: bool = False):
    """Load all models, intersect ids, return ids, labels, preds, correctness.

    If ``strict_split`` is set, disagreeing gold labels across files on the same
    id are a hard error rather than a warning. Disagreement is the fingerprint of
    two different test splits being joined (e.g. an official-ARG 1258 test paired
    with a RoBERTa run trained on the rebuilt prepare_data split), so for a
    publishable diagnostic you want this to fail loudly.
    """
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
        msg = (f"{mismatches} ids have disagreeing gold labels across files "
               "-- the prediction files almost certainly come from DIFFERENT "
               "test splits (check that RoBERTa was trained/evaluated on the "
               "same official ARG split that produced the GPT-5.4 preds).")
        if strict_split:
            raise SystemExit(f"[diag] FATAL: {msg}")
        print(f"[diag] WARNING: {msg}")
    else:
        print(f"[diag] split check OK: gold labels agree on all "
              f"{len(ids)} common ids.")

    labels = np.array([ref[i] for i in ids], dtype=int)
    raw_preds, correct = {}, {}
    for n in models:
        raw_preds[n] = np.array([preds_per[n][i] for i in ids], dtype=int)
        correct[n] = raw_preds[n] == labels
    return ids, labels, raw_preds, correct, sizes


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))


def oracle_predictions(labels: np.ndarray, correct_list: List[np.ndarray]):
    """Per-sample oracle: output gold where any model is correct, else flip.

    Returns (oracle_preds, any_correct). Binary {0,1} assumed (real=0, fake=1).
    Computing macro-F1 on these predictions yields a *per-class* oracle ceiling:
    a sample where everyone misses the minority (fake) class stays a fake-class
    error, so a majority-class consensus cannot launder the headroom.
    """
    any_correct = np.zeros(len(labels), dtype=bool)
    for c in correct_list:
        any_correct = np.logical_or(any_correct, c)
    preds = np.where(any_correct, labels, 1 - labels)
    return preds, any_correct


def compute(models: Dict[str, str], strict_split: bool = False):
    ids, labels, raw_preds, correct, sizes = align(models, strict_split)
    names = list(models.keys())
    n = len(ids)

    # Per-model accuracy + macro-F1.
    acc, f1 = {}, {}
    for name in names:
        acc[name] = float(correct[name].mean())
        f1[name] = macro_f1(labels, raw_preds[name])

    # PRIMARY ranking is by macro-F1; accuracy kept for reference.
    best_single_f1_model = max(f1, key=f1.get)
    best_single_f1 = f1[best_single_f1_model]
    best_single_acc_model = max(acc, key=acc.get)
    best_single_acc = acc[best_single_acc_model]

    # Pairwise: error Jaccard + oracle (acc & macro-F1) + breakdown.
    pair_jaccard, pair_oracle_acc, pair_oracle_f1, pair_breakdown = {}, {}, {}, {}
    for a, b in itertools.combinations(names, 2):
        Ea, Eb = ~correct[a], ~correct[b]
        inter = np.logical_and(Ea, Eb).sum()
        union = np.logical_or(Ea, Eb).sum()
        pair_jaccard[(a, b)] = float(inter / union) if union else 0.0

        o_preds, any_c = oracle_predictions(labels, [correct[a], correct[b]])
        o_acc = float(any_c.mean())
        o_f1 = macro_f1(labels, o_preds)
        pair_oracle_acc[(a, b)] = o_acc
        pair_oracle_f1[(a, b)] = o_f1

        pair_best_acc = max(acc[a], acc[b])
        pair_best_f1 = max(f1[a], f1[b])
        pair_breakdown[(a, b)] = {
            "both_correct": int(np.logical_and(correct[a], correct[b]).sum()),
            f"only_{a}": int(np.logical_and(correct[a], ~correct[b]).sum()),
            f"only_{b}": int(np.logical_and(~correct[a], correct[b]).sum()),
            "both_wrong": int(np.logical_and(~correct[a], ~correct[b]).sum()),
            "pair_best_single_acc": pair_best_acc,
            "pair_best_single_f1": pair_best_f1,
            "oracle_acc_gain_pts": (o_acc - pair_best_acc) * 100,
            "oracle_f1_gain_pts": (o_f1 - pair_best_f1) * 100,
        }

    # Full oracle over all models, both metrics.
    o_preds_full, any_correct = oracle_predictions(
        labels, [correct[name] for name in names])
    full_oracle_acc = float(any_correct.mean())
    full_oracle_f1 = macro_f1(labels, o_preds_full)
    full_gain_acc_pts = (full_oracle_acc - best_single_acc) * 100
    full_gain_f1_pts = (full_oracle_f1 - best_single_f1) * 100

    # Verdict on the PRIMARY (macro-F1) headroom.
    g = full_gain_f1_pts
    if g >= 5.0:
        verdict = ("FOUNDED — routing direction has a learning-theoretic basis "
                   "(macro-F1 gain ≥ 5 pts).")
    elif g < 1.0:
        verdict = ("PIVOT — complementarity is negligible in macro-F1 "
                   "(gain < 1 pt).")
    else:
        verdict = ("WEAK — macro-F1 gain in the 1–5 pt grey zone; routing is "
                   "plausible but not compelling.")

    summary = {
        "n_test": n,
        "models": names,
        "per_model_sizes": sizes,
        "primary_metric": "macro_f1",
        "label_balance": {
            "n_real_0": int((labels == 0).sum()),
            "n_fake_1": int((labels == 1).sum()),
            "fake_ratio": float((labels == 1).mean()),
        },
        "accuracy": acc,
        "macro_f1": f1,
        "best_single_model": best_single_f1_model,       # by macro-F1 (primary)
        "best_single_f1": best_single_f1,
        "best_single_acc_model": best_single_acc_model,
        "best_single_acc": best_single_acc,
        "pair_error_jaccard": {f"{a}|{b}": v for (a, b), v in pair_jaccard.items()},
        "pair_oracle_acc": {f"{a}|{b}": v for (a, b), v in pair_oracle_acc.items()},
        "pair_oracle_f1": {f"{a}|{b}": v for (a, b), v in pair_oracle_f1.items()},
        "pair_breakdown": {f"{a}|{b}": v for (a, b), v in pair_breakdown.items()},
        "full_oracle_acc": full_oracle_acc,
        "full_oracle_f1": full_oracle_f1,
        "full_oracle_gain_acc_pts": full_gain_acc_pts,
        "full_oracle_gain_f1_pts": full_gain_f1_pts,
        # primary headroom alias (macro-F1)
        "full_oracle_gain_pts": full_gain_f1_pts,
        "verdict": verdict,
    }
    return (summary, names, acc, f1, pair_jaccard, pair_oracle_acc,
            pair_oracle_f1, pair_breakdown, full_oracle_acc, full_oracle_f1,
            best_single_acc, best_single_f1)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_metric_bars(names, acc, f1, pair_oracle_acc, pair_oracle_f1,
                     full_oracle_acc, full_oracle_f1, best_single_acc,
                     best_single_f1, dataset, path):
    """Grouped acc vs macro-F1 bars: single models + pair oracles + full oracle.

    The acc–F1 gap on each bar is the visual tell for majority-class inflation:
    where the orange (macro-F1) bar sits far below the blue (accuracy) bar, the
    model — or the oracle — is mostly riding the majority class.
    """
    cats = list(names) + [f"oracle:{a}+{b}" for (a, b) in pair_oracle_acc] \
        + ["oracle:ALL"]
    acc_vals = [acc[n] for n in names] \
        + [pair_oracle_acc[k] for k in pair_oracle_acc] + [full_oracle_acc]
    f1_vals = [f1[n] for n in names] \
        + [pair_oracle_f1[k] for k in pair_oracle_f1] + [full_oracle_f1]

    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.25), 4.8))
    b1 = ax.bar(x - w / 2, acc_vals, w, label="accuracy", color="#4C78A8")
    b2 = ax.bar(x + w / 2, f1_vals, w, label="macro-F1 (primary)",
                color="#F58518")
    ax.axhline(best_single_acc, ls=":", color="#4C78A8", lw=1.2,
               label=f"best single acc = {best_single_acc:.3f}")
    ax.axhline(best_single_f1, ls="--", color="#F58518", lw=1.4,
               label=f"best single F1 = {best_single_f1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(min(min(acc_vals), min(f1_vals)) - 0.06, 1.03)
    ax.set_title(f"{dataset}: single models vs oracle routers "
                 f"(accuracy vs macro-F1)")
    for bars, vals in ((b1, acc_vals), (b2, f1_vals)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.legend(fontsize=8, loc="lower left")
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


def plot_complementarity(pair_breakdown, n_test, dataset, path):
    """MOTIVATION figure: bidirectional complementarity per model pair.

    For each pair we draw a stacked bar of the four outcome buckets. The two
    middle bands — only-A-correct and only-B-correct — are the heart of the
    routing thesis: a non-trivial only-A band means the small model BEATS the
    LLM on a real slice of data, so the optimal policy is not "always send to
    the LLM". (Weibo21: only-RoBERTa=220 / only-GPT=364.)
    """
    # pair_breakdown keys may be ("A","B") tuples; normalize to (a, b).
    pairs = [k if isinstance(k, tuple) else tuple(k.split("|"))
             for k in pair_breakdown.keys()]
    raw_keys = list(pair_breakdown.keys())
    npairs = len(pairs)
    fig, ax = plt.subplots(figsize=(max(5, npairs * 2.4), 5))

    # bucket colors: both_correct (grey), only_A (green), only_B (blue),
    # both_wrong (red)
    colA, colB = "#54A24B", "#4C78A8"
    x = np.arange(npairs)
    w = 0.6
    for xi, (key, (a, b)) in enumerate(zip(raw_keys, pairs)):
        bd = pair_breakdown[key]
        both_c = bd["both_correct"]
        only_a = bd[f"only_{a}"]
        only_b = bd[f"only_{b}"]
        both_w = bd["both_wrong"]
        bottom = 0
        segs = [
            (both_c, "#BFBFBF", "both correct"),
            (only_a, colA, f"only {a}"),
            (only_b, colB, f"only {b}"),
            (both_w, "#E45756", "both wrong"),
        ]
        for val, col, lab in segs:
            ax.bar(xi, val, w, bottom=bottom, color=col,
                   label=lab if xi == 0 else None, edgecolor="white")
            if val > 0:
                pct = 100.0 * val / n_test
                ax.text(xi, bottom + val / 2, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if col in (colA, colB, "#E45756")
                        else "black")
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\nvs\n{b}" for (a, b) in pairs], fontsize=9)
    ax.set_ylabel(f"# test samples (n={n_test})")
    ax.set_title(f"{dataset}: bidirectional complementarity\n"
                 "(both middle bands non-zero ⇒ routing, not always-LLM)")
    ax.legend(fontsize=8, loc="upper right")
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
    lb = s["label_balance"]
    lines.append(f"- Label balance: real(0) = {lb['n_real_0']}, "
                 f"fake(1) = {lb['n_fake_1']} "
                 f"(fake ratio = {lb['fake_ratio']*100:.1f}%)")
    lines.append(f"- Models: {', '.join(s['models'])}")
    lines.append(f"- **Primary metric: macro-F1** "
                 f"(accuracy reported for reference)\n")

    lines.append("## Verdict\n")
    lines.append(f"> **{s['verdict']}**\n")
    lines.append(f"- Best single model (by macro-F1): **{s['best_single_model']}** "
                 f"(macro-F1 = {s['best_single_f1']*100:.2f}%, "
                 f"acc = {s['accuracy'][s['best_single_model']]*100:.2f}%)")
    lines.append(f"- Oracle router (any model correct), macro-F1: "
                 f"**{s['full_oracle_f1']*100:.2f}%** | "
                 f"accuracy: {s['full_oracle_acc']*100:.2f}%")
    lines.append(f"- **Routing headroom (macro-F1) = "
                 f"{s['full_oracle_gain_f1_pts']:+.2f} pts** "
                 f"(accuracy headroom = {s['full_oracle_gain_acc_pts']:+.2f} "
                 f"pts, inflated by the majority class)\n")
    lines.append("Decision rule (on macro-F1): ≥ 5 pts → well-founded; "
                 "< 1 pt → pivot; 1–5 pts → grey zone.\n")
    lines.append("> The oracle macro-F1 is computed on per-sample oracle "
                 "predictions (gold when any model is right, flipped otherwise), "
                 "so the *at-least-one-correct* credit is counted **per class**. "
                 "This stops a majority-class consensus from inflating the "
                 "headroom on imbalanced data.\n")

    lines.append("## Per-model performance\n")
    lines.append("| Model | Macro-F1 (primary) | Accuracy | acc−F1 gap | Test size |")
    lines.append("|---|---|---|---|---|")
    for n in s["models"]:
        gap = (s["accuracy"][n] - s["macro_f1"][n]) * 100
        lines.append(f"| {n} | {s['macro_f1'][n]*100:.2f}% | "
                     f"{s['accuracy'][n]*100:.2f}% | {gap:+.2f} pts | "
                     f"{s['per_model_sizes'][n]} |")
    lines.append("")
    lines.append("*A large positive acc−F1 gap means the model is largely "
                 "riding the majority (real) class and is weak on the minority "
                 "(fake) class.*\n")

    lines.append("## Pairwise error overlap & oracle\n")
    lines.append("| Pair | Error Jaccard | Oracle macro-F1 | Oracle acc | "
                 "Pair best F1 | F1 gain (pts) | acc gain (pts) |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in s["pair_oracle_f1"]:
        j = s["pair_error_jaccard"][key]
        of1 = s["pair_oracle_f1"][key]
        oacc = s["pair_oracle_acc"][key]
        bd = s["pair_breakdown"][key]
        lines.append(f"| {key} | {j:.3f} | {of1*100:.2f}% | {oacc*100:.2f}% | "
                     f"{bd['pair_best_single_f1']*100:.2f}% | "
                     f"{bd['oracle_f1_gain_pts']:+.2f} | "
                     f"{bd['oracle_acc_gain_pts']:+.2f} |")
    lines.append("")
    lines.append("*Error Jaccard = |both wrong| / |at least one wrong|. "
                 "Lower means the two models fail on different samples — exactly "
                 "the complementarity a router can exploit.*\n")

    lines.append("## Where the routing gain comes from — bidirectional "
                 "complementarity\n")
    lines.append("This is the crux of the routing argument: the thesis needs "
                 "**both** directions to be non-empty — samples where the small "
                 "model wins AND samples where the LLM wins. If only one band is "
                 "populated, the optimal policy collapses to \"always use that "
                 "model\" and there is nothing to route.\n")
    lines.append("| Pair | both correct | only-1st correct | only-2nd correct | "
                 "both wrong |")
    lines.append("|---|---|---|---|---|")
    for key, bd in s["pair_breakdown"].items():
        a, b = key.split("|")
        n = s["n_test"]
        bc, oa, ob, bw = (bd["both_correct"], bd[f"only_{a}"],
                          bd[f"only_{b}"], bd["both_wrong"])
        lines.append(
            f"| {a} vs {b} | {bc} ({bc/n*100:.1f}%) | "
            f"only {a}: {oa} ({oa/n*100:.1f}%) | "
            f"only {b}: {ob} ({ob/n*100:.1f}%) | {bw} ({bw/n*100:.1f}%) |")
    lines.append("")

    lines.append("## Figures\n")
    for fp in fig_paths:
        rel = os.path.basename(fp)
        lines.append(f"![{rel}]({rel})\n")

    lines.append("## How to read this\n")
    lines.append("- **World A (routing useless):** error Jaccard ≈ 1 and oracle "
                 "macro-F1 gain ≈ 0 — models make the *same* mistakes.")
    lines.append("- **World B (routing gold):** error Jaccard ≈ 0 and oracle "
                 "macro-F1 gain is large — disjoint mistakes, a perfect router "
                 "approaches 100%.")
    lines.append("- **Watch the acc vs macro-F1 split.** On imbalanced data "
                 "(GossipCop) the accuracy headroom overstates the case; the "
                 "macro-F1 headroom is the honest number and drives the verdict.")
    lines.append("- The real datasets sit between these worlds; this report "
                 "locates exactly where, and whether the gap justifies the "
                 "routing thesis.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    metavar="NAME=PATH",
                    help="repeatable, e.g. --pred RoBERTa=outputs/preds/x.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--require", action="append", default=[],
                    help="model name(s) that MUST be present; error if missing. "
                         "By default missing prediction files are skipped.")
    ap.add_argument("--strict-split", action="store_true",
                    help="fail (instead of warn) if gold labels disagree across "
                         "prediction files — guards against mixing test splits.")
    args = ap.parse_args()

    models = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"--pred must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = path

    # Tolerate not-yet-produced prediction files (e.g. ARG before you've trained
    # and dumped it). Skip them with a warning instead of crashing.
    present, missing = {}, []
    for name, path in models.items():
        if os.path.exists(path):
            present[name] = path
        else:
            missing.append((name, path))
    for name, path in missing:
        if name in args.require:
            raise SystemExit(f"[diag] required model {name!r} missing: {path}")
        print(f"[diag] SKIP {name!r}: prediction file not found ({path})")
    if len(present) < 2:
        raise SystemExit(
            f"[diag] need >=2 models to compare, found {len(present)} "
            f"({list(present)}). Produce more prediction files first — e.g. ARG "
            "via src/arg_integration/README.md.")
    if missing:
        print(f"[diag] running on {len(present)} models: {list(present)}")
    models = present

    os.makedirs(args.outdir, exist_ok=True)
    (summary, names, acc, f1, pair_jaccard, pair_oracle_acc, pair_oracle_f1,
     pair_breakdown, full_oracle_acc, full_oracle_f1, best_single_acc,
     best_single_f1) = compute(models, strict_split=args.strict_split)

    fig1 = os.path.join(args.outdir, "accuracy_vs_oracle.png")
    fig2 = os.path.join(args.outdir, "error_jaccard_heatmap.png")
    fig3 = os.path.join(args.outdir, "complementarity.png")
    plot_metric_bars(names, acc, f1, pair_oracle_acc, pair_oracle_f1,
                     full_oracle_acc, full_oracle_f1, best_single_acc,
                     best_single_f1, args.dataset, fig1)
    plot_jaccard_heatmap(names, pair_jaccard, args.dataset, fig2)
    plot_complementarity(pair_breakdown, summary["n_test"], args.dataset, fig3)

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_report(summary, args.dataset, [fig1, fig3, fig2],
                 os.path.join(args.outdir, "report.md"))

    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"dataset             : {args.dataset}")
    print(f"primary metric      : macro-F1")
    print(f"best single (F1)    : {summary['best_single_model']} "
          f"({best_single_f1*100:.2f}%)")
    print(f"oracle macro-F1     : {full_oracle_f1*100:.2f}%  "
          f"(acc {full_oracle_acc*100:.2f}%)")
    print(f"headroom (macro-F1) : {summary['full_oracle_gain_f1_pts']:+.2f} pts")
    print(f"headroom (accuracy) : {summary['full_oracle_gain_acc_pts']:+.2f} pts")
    print(f"verdict             : {summary['verdict']}")
    print(f"outputs             : {args.outdir}/  (report.md, summary.json, *.png)")


if __name__ == "__main__":
    main()
