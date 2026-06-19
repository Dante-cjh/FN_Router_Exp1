#!/usr/bin/env python
"""
diagnose_slm_floor.py — why does Mozannar-Sontag's classifier h jump on weibo21?

FINDING (from the first baseline run): on weibo21 the M&S head h reached 89.92
macro-F1 while the frozen RoBERTa pred was 76.30 — a +13.6 gap that does NOT
appear on gossipcop. This script pins down the cause so you can decide how to
report it.

It is NOT a storage bug: the json `pred` and the npz `emb`/`pred` come from the
same forward pass (verified: 100% agreement, identical prob). The real cause is
that REFITTING a linear head on the *val* embeddings generalises to test much
better than the frozen head trained on *train* — i.e. train->test distribution
shift that the val split tracks. A fresh logistic regression on val emb recovers
~88 on weibo21 test from the SAME embeddings the frozen model scores 76 on.

Consequence: any method that refits a classifier on val (M&S's h) gets that
head-refit gain "for free", which is not a deferral/routing gain. Hybrid LLM,
RouteLLM and step7 keep the frozen pred as their floor, so for a fair comparison
M&S should be read on the frozen floor (l2d_router.py --slm_floor frozen).

USAGE
  python -m baselines.diagnose_slm_floor \
    --datasets weibo21:outputs/preds/weibo21_roberta_val.json:outputs/preds/weibo21_roberta.json \
               gossipcop:outputs/preds/gossipcop_roberta_val.json:outputs/preds/gossipcop_roberta.json

(or just run with no args to use those two defaults)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def load(p):
    return {int(r["id"]): r for r in json.loads(Path(p).read_text())}


def mf1(y, p):
    return f1_score(y, p, average="macro") * 100.0


def diagnose(name, val_path, test_path):
    V, T = load(val_path), load(test_path)
    vi, ti = sorted(V), sorted(T)
    if not all("emb" in V[i] for i in vi) or not all("emb" in T[i] for i in ti):
        print(f"[{name}] SKIP: emb missing — run src/merge_emb.py first.")
        return
    Xv = np.array([V[i]["emb"] for i in vi]); yv = np.array([V[i]["label"] for i in vi])
    Xt = np.array([T[i]["emb"] for i in ti]); yt = np.array([T[i]["label"] for i in ti])
    frozen = np.array([T[i]["pred"] for i in ti])

    sc = StandardScaler().fit(Xv)
    # plain refit (matches M&S head capacity roughly) and a class-balanced refit
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xv), yv)
    lrb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xv), yv)
    refit = lr.predict(sc.transform(Xt))
    refit_bal = lrb.predict(sc.transform(Xt))

    f1_frozen = mf1(yt, frozen)
    f1_refit = mf1(yt, refit)
    f1_refit_bal = mf1(yt, refit_bal)
    agree = (refit == frozen).mean() * 100
    bal = yt.mean()

    print(f"\n[{name}]  test n={len(ti)}  fake-rate={bal:.2f}")
    print(f"  frozen RoBERTa head (train-fit)         macro-F1 = {f1_frozen:.2f}")
    print(f"  refit head on VAL emb (plain)           macro-F1 = {f1_refit:.2f}  "
          f"({f1_refit - f1_frozen:+.2f})")
    print(f"  refit head on VAL emb (class-balanced)  macro-F1 = {f1_refit_bal:.2f}  "
          f"({f1_refit_bal - f1_frozen:+.2f})")
    print(f"  refit vs frozen agreement = {agree:.1f}%")
    if f1_refit - f1_frozen >= 3.0:
        print(f"  -> HEAD-REFIT GAIN is large: the frozen SLM leaves ~{f1_refit-f1_frozen:.0f} "
              f"pts on the table (train/test shift). Read M&S on --slm_floor frozen, "
              f"and consider a val-refit/calibrated head as a separate SLM upgrade.")
    else:
        print(f"  -> head-refit gain is small; frozen floor is fine, M&S is comparable as-is.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[
        "weibo21:outputs/preds/weibo21_roberta_val.json:outputs/preds/weibo21_roberta.json",
        "gossipcop:outputs/preds/gossipcop_roberta_val.json:outputs/preds/gossipcop_roberta.json",
    ], help="each entry NAME:VAL_PATH:TEST_PATH")
    args = ap.parse_args()
    for spec in args.datasets:
        name, vp, tp = spec.split(":")
        diagnose(name, vp, tp)


if __name__ == "__main__":
    main()
