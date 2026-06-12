#!/usr/bin/env python
"""
step7_learned_router.py — Learned, pre-generation, gain-predicting router.

The crux test: can a CHEAP model (no LLM call) predict the band where the LLM
helps, i.e. (large correct AND small wrong)? If yes, pre-generation routing is
viable. The script prints the gain-label test AUC/AP FIRST as the go/no-go, then
produces the learned-router cost-quality curve to overlay on step6's Pareto plot.

TRAIN on val, EVAL on test. Never train on test.

INPUT FILES (unified per-sample preds, inner-joined on `id`)
  small_*: [{"id","label","pred","prob", "emb": [..]? }]   RoBERTa (+ optional penultimate emb)
  large_*: [{"id","label","pred"}]                          GPT-5.4 direct judge

USAGE
  python step7_learned_router.py \
    --small_val outputs/preds/weibo21_roberta_val.json \
    --large_val outputs/preds/weibo21_gpt_val.json \
    --small_test outputs/preds/weibo21_roberta.json \
    --large_test outputs/preds/weibo21_gpt.json \
    --name weibo21 --out outputs/diagnostic/weibo21
"""
import json, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def load(p):
    return {int(r["id"]): r for r in json.loads(Path(p).read_text())}


def entropy(p1):
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def build(small_path, large_path):
    S, L = load(small_path), load(large_path)
    ids = sorted(set(S) & set(L))
    y  = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"]  for i in ids])
    lp = np.array([L[i]["pred"]  for i in ids])
    prob = np.array([float(S[i].get("prob", 0.5)) for i in ids])
    scal = np.column_stack([prob, entropy(prob)])          # always-available scalars
    if all("emb" in S[i] for i in ids):
        emb = np.array([S[i]["emb"] for i in ids], float)
        feats = np.column_stack([emb, scal])
        has_emb = True
    else:
        feats, has_emb = scal, False
    gain = ((lp == y) & (sp != y)).astype(int)             # the band the LLM rescues
    return feats, gain, y, sp, lp, has_emb


def macro_f1(y, p):
    return f1_score(y, p, average="macro") * 100.0


def main():
    ap = argparse.ArgumentParser()
    for k in ("small_val", "large_val", "small_test", "large_test", "name", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--points", type=int, default=21)
    args = ap.parse_args()

    Xtr, gtr, *_ , he1 = build(args.small_val, args.large_val)
    Xte, gte, yte, spte, lpte, he2 = build(args.small_test, args.large_test)
    if Xtr.shape[1] != Xte.shape[1]:
        raise SystemExit("val/test feature dims differ — dump emb consistently or not at all.")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), gtr)
    score = clf.predict_proba(sc.transform(Xte))[:, 1]     # predicted P(LLM rescues)

    # ---- THE go/no-go number ----
    auc = roc_auc_score(gte, score) if gte.sum() else float("nan")
    appr = average_precision_score(gte, score) if gte.sum() else float("nan")
    base = gte.mean()
    print(f"\n[{args.name}] gain-label learnability (test)  feats={'emb+scalar' if he2 else 'scalar-only'}")
    print(f"  positives (LLM-rescues) = {int(gte.sum())}/{len(gte)} ({base*100:.1f}%)")
    print(f"  AUC = {auc:.3f}   AP = {appr:.3f}  (AP baseline = {base:.3f})")
    verdict = ("VIABLE — pre-generation routing can find the band"
               if auc > 0.62 else
               "MARGINAL — band barely predictable; consider cascade-with-abstention"
               if auc > 0.55 else
               "NOT learnable from cheap features — LLM wins are not pre-predictable here")
    print(f"  -> {verdict}\n")

    # ---- learned-router cost-quality curve (route highest predicted-gain first) ----
    order = np.argsort(-score)
    n, fr = len(yte), np.linspace(0, 1, args.points)
    learned = []
    for f in fr:
        k = int(round(f * n))
        mask = np.zeros(n, bool); mask[order[:k]] = True
        comb = spte.copy(); comb[mask] = lpte[mask]
        learned.append(macro_f1(yte, comb))
    all_small, all_large = macro_f1(yte, spte), macro_f1(yte, lpte)
    best_i = int(np.argmax(learned))
    print(f"  all-SLM={all_small:.2f}  all-LLM={all_large:.2f}")
    print(f"  learned-router peak={learned[best_i]:.2f} @ {fr[best_i]*100:.0f}% LLM budget")
    for f in (0.05, 0.1, 0.2, 0.3):
        i = int(round(f * (args.points - 1)))
        print(f"    budget {int(f*100):>2}% -> learned {learned[i]:.2f}")

    odir = Path(args.out); odir.mkdir(parents=True, exist_ok=True)
    (odir / "routing_learned.json").write_text(json.dumps({
        "name": args.name, "n": n, "fractions": fr.tolist(),
        "gain_auc": auc, "gain_ap": appr, "gain_base_rate": base,
        "all_small_f1": all_small, "all_large_f1": all_large,
        "learned_f1": learned,
    }, indent=2))
    print(f"  wrote {odir/'routing_learned.json'} (overlay learned_f1 on step6's plot)")


if __name__ == "__main__":
    main()
