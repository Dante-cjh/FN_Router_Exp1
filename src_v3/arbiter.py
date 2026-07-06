#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src_v3/arbiter.py — Round-3 core new module: the rationale-aware ARBITER.

WHAT IT IS (reports/08 §4/§5): v1/v2 died on the "swallow the LLM label once
routed" rule. The arbiter replaces it: on samples where SLM and LLM DISAGREE,
predict P(the LLM is the right one | SLM confidence, disagreement direction,
LLM rationale, text) and adopt the LLM label only when that probability clears
a threshold (tuned on val, or CRC-certified via src_v3/crc.py).

WHY IT CAN WORK (error analysis, reports/08 §2): the LLM's dominant failure
mode — hallucinated corroboration — is *legible in its own rationale*
(memory_match-style argument => predicts real regardless of claim truth),
while the SLM's failure mode (style-overfit false 'fake') is legible in its
confidence + the text. Neither is visible to SLM-side uncertainty, which is
exactly why U_epi/U_ale ablated to zero (reports/08 §3).

FEATURE TIERS (each is an ablation row AND a baseline mapping):
  frugal   [prob, ent, sp, lp, rlen]           ~ FrugalGPT post-hoc scorer
                                                  (Chen et al., 2023, arXiv:2305.05176)
                                                  — no rationale CONTENT
  dict     + 4-dim rationale argument-mode lexicon (reports/08 taxonomy)
  emb      + SLM news emb(768)                  = the pilot that hit 87.27/77.38
  enc      + SBERT interaction on frozen (news, rationale) encodings
                                                  ~ ARG-style rationale evaluation
                                                  (Hu et al., AAAI 2024), transferable zh<->en
  full     everything

TRAIN SCOPE:
  disagree  binary "LLM right" on val disagreements (pilot; n=530/187)
  all       two heads P(SLM right|x), P(LLM right|x,rat) on ALL val samples,
            arbitration score = difference (more data, indirect target)

USAGE (from Router_Exp1/):
  python -m src_v3.arbiter --name weibo21 --lang zh --features emb
  python -m src_v3.arbiter --name gossipcop --lang en --features enc --enc outputs_v3/enc
  # zh->en zero-shot transfer of the arbitration principle (M5 teaser):
  python -m src_v3.arbiter --name gossipcop --lang en --features enc --enc outputs_v3/enc \
      --train_name weibo21 --train_lang zh
Output: outputs_v3/arbiter/<name>__<features>[__from_<train_name>]/arbiter.json + scores.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src_v3.common import (build_split, interaction_block, macro_f1,
                           bootstrap_delta)
from src_v3.crc import crc_sweep

TIERS = ["frugal", "dict", "emb", "enc", "full"]


def feature_matrix(split, tier, transfer=False):
    blocks = [np.column_stack([split["prob"], split["ent"],
                               split["sp"], split["lp"], split["rlen"]])]
    if tier in ("dict", "emb", "enc", "full"):
        blocks.append(split["rdict"])
    if tier in ("emb", "full") and not transfer:
        blocks.append(split["emb"])          # backbone-specific: drop on transfer
    if tier in ("enc", "full"):
        if "e_news" not in split:
            raise SystemExit("tier needs encodings; run src_v3.encode_rationales "
                             "and pass --enc outputs_v3/enc")
        blocks.append(interaction_block(split["e_news"], split["e_rat"]))
    return np.hstack(blocks)


def make_model(kind, C, seed):
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000, random_state=seed))
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, C=C, class_weight="balanced", random_state=seed))


def fit_arbiter(tr, tier, scope="disagree", kind="logreg", C=0.5, seed=0,
                transfer=False):
    """Returns score_fn(split)->(n,) score, higher = trust the LLM."""
    Xtr = feature_matrix(tr, tier, transfer)
    if scope == "disagree":
        m = tr["sp"] != tr["lp"]
        clf = make_model(kind, C, seed)
        clf.fit(Xtr[m], (tr["lp"][m] == tr["y"][m]).astype(int))
        return lambda sp_: clf.predict_proba(feature_matrix(sp_, tier, transfer))[:, 1]
    # scope == "all": two heads on every val sample
    cS, cL = make_model(kind, C, seed), make_model(kind, C, seed + 1)
    cS.fit(Xtr, (tr["sp"] == tr["y"]).astype(int))
    cL.fit(Xtr, (tr["lp"] == tr["y"]).astype(int))

    def score(sp_):
        X = feature_matrix(sp_, tier, transfer)
        return cL.predict_proba(X)[:, 1] - cS.predict_proba(X)[:, 1] + 0.5
    return score


def arbitrated_pred(split, scores, tau):
    """Agreements keep the shared label; disagreements adopt LLM iff score>tau."""
    pred = split["sp"].copy()
    m = (split["sp"] != split["lp"]) & (scores > tau)
    pred[m] = split["lp"][m]
    return pred


def tune_tau_valf1(val, scores_val, grid=None):
    if grid is None:  # quantile grid over DISAGREEMENT scores (robust to skew)
        m = val["sp"] != val["lp"]
        qs = np.percentile(scores_val[m], np.arange(5, 100, 5)) if m.any() else []
        grid = np.unique(np.round(qs, 6))
    best = (0.5, -1.0)
    for t in grid:
        f = macro_f1(val["y"], arbitrated_pred(val, scores_val, t))
        if f > best[1]:
            best = (float(t), f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--features", default="emb", choices=TIERS)
    ap.add_argument("--scope", default="disagree", choices=["disagree", "all"])
    ap.add_argument("--model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc", default=None, help="outputs_v3/enc (needed for enc/full)")
    ap.add_argument("--train_name", default=None, help="transfer: train on this dataset")
    ap.add_argument("--train_lang", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out_root", default="outputs_v3/arbiter")
    args = ap.parse_args()

    transfer = args.train_name is not None and args.train_name != args.name
    tr_name = args.train_name or args.name
    tr_lang = args.train_lang or args.lang
    if transfer and args.features in ("emb", "full"):
        print("[warn] SLM emb is backbone-specific; it is DROPPED in transfer mode")

    tr = build_split(tr_name, tr_lang, "val", args.enc)
    val = build_split(args.name, args.lang, "val", args.enc)   # for tau tuning/CRC
    te = build_split(args.name, args.lang, "test", args.enc)

    score = fit_arbiter(tr, args.features, args.scope, args.model, args.C,
                        args.seed, transfer)
    s_val, s_te = score(val), score(te)

    mv, mt = val["sp"] != val["lp"], te["sp"] != te["lp"]
    y_arb_t = (te["lp"][mt] == te["y"][mt]).astype(int)
    auc = float(roc_auc_score(y_arb_t, s_te[mt])) if len(set(y_arb_t)) > 1 else None

    tau_f1, val_f1 = tune_tau_valf1(val, s_val)
    pred_arb = arbitrated_pred(te, s_te, tau_f1)
    pred_swallow = te["lp"]                       # full-coverage swallow == all-LLM
    res = dict(
        name=args.name, features=args.features, scope=args.scope,
        model=args.model, transfer_from=(tr_name if transfer else None),
        n_disagree_train=int((tr["sp"] != tr["lp"]).sum()),
        n_disagree_test=int(mt.sum()),
        arb_auc_test_disagree=auc,
        tau_valF1=tau_f1, val_full_arb_f1=round(val_f1, 2),
        endpoints=dict(all_slm=round(macro_f1(te["y"], te["sp"]), 2),
                       all_llm=round(macro_f1(te["y"], te["lp"]), 2)),
        full_arbitration_f1=round(macro_f1(te["y"], pred_arb), 2),
        adopt_rate_test=float((s_te[mt] > tau_f1).mean()),
        delta_vs_all_slm=bootstrap_delta(te["y"], pred_arb, te["sp"], args.n_boot),
        delta_vs_all_llm=bootstrap_delta(te["y"], pred_arb, pred_swallow, args.n_boot),
        crc_sweep=crc_sweep(s_val[mv], (val["lp"][mv] != val["y"][mv]).astype(int),
                            s_te[mt], (te["lp"][mt] != te["y"][mt]).astype(int)),
    )
    # F1 realised at each CRC alpha (the headline safety-vs-gain table)
    for row in res["crc_sweep"]:
        if row["tau"] is not None:
            row["full_arb_f1"] = round(
                macro_f1(te["y"], arbitrated_pred(te, s_te, row["tau"])), 2)

    tag = f"{args.name}__{args.features}" + (f"__from_{tr_name}" if transfer else "")
    out = f"{args.out_root}/{tag}"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/arbiter.json", "w"), ensure_ascii=False, indent=2)
    np.savez_compressed(f"{out}/scores.npz",
                        val_ids=np.array(val["ids"]), s_val=s_val,
                        test_ids=np.array(te["ids"]), s_test=s_te,
                        tau_valF1=tau_f1)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[arbiter] wrote {out}/arbiter.json")


if __name__ == "__main__":
    main()
