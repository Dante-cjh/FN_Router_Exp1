#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src_v3/common.py — Round-3 shared utilities.

Round-3 main method = RA^3 (Rationale-Aware Arbitrated Routing with Risk control):
  Stage-1  net-utility pre-router  (v2 asset, decides WHETHER to spend)
  Stage-2  rationale-aware arbiter (decides WHOM to trust; ARG/SBERT-style)
  CRC      conformal risk control  (decides HOW conservative; distribution-free)

ISOLATION: reads v1 preds (outputs/preds/*) + optional v3 encodings
(outputs_v3/enc/*); writes only under outputs_v3/.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- io
def load_rows(path):
    return {int(r["id"]): r for r in json.loads(Path(path).read_text())}


def preds_paths(name, split):
    suf = "" if split == "test" else "_val"
    return (f"outputs/preds/{name}_roberta{suf}.json",
            f"outputs/preds/{name}_gpt54{suf}.json")


def binary_entropy(p1, eps=1e-12):
    p1 = np.clip(np.asarray(p1, dtype=np.float64), eps, 1 - eps)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def get_emb(rows_by_id, ids, small_path):
    """SLM penultimate emb: inline 'emb' field, else *_emb.npz sidecar."""
    if all("emb" in rows_by_id[i] for i in ids):
        return np.array([rows_by_id[i]["emb"] for i in ids], dtype=np.float32)
    base, _ = os.path.splitext(small_path)
    npz = base + "_emb.npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        idx = {int(k): j for j, k in enumerate(z["ids"])}
        return np.array([z["emb"][idx[i]] for i in ids], dtype=np.float32)
    raise SystemExit(f"no emb inline and no sidecar {npz}")


# ----------------------------------------------------------------- rationale dictionary
# Argument-mode lexicon (same taxonomy as src_v2/error_analysis.py, reports/08 §2):
#   memory_match  = "I remember this event" corroboration  -> hallucination-prone
#   plausibility  = style/plausibility endorsement
#   skepticism    = source/verifiability doubt             -> fake-leaning
PAT = {
    "en": {
        "memory_match": r"\b(match(es|ed)?|widely reported|indeed|accurate(ly)?|confirmed|was (a )?real|actually (happened|occurred)|documented|verifiable)\b",
        "plausibility": r"\b(plausible|reads like|style|tone|typical|natural|consistent with|realistic|genuine)\b",
        "skepticism":   r"\b(lack(s|ing)?|no (reliable|credible|verifiable)|unverified|rumor|fabricat|implausible|sensational|clickbait|exaggerat|hoax|satir|unsubstantiated|misleading)\b",
    },
    "zh": {
        "memory_match": r"(属实|对应|确实|相符|吻合|一致|真实事件|确有|已证实|报道过)",
        "plausibility": r"(合理|符合常识|符合.{0,4}风格|通顺|自然|可信度较高)",
        "skepticism":   r"(缺乏|谣|夸大|未经证实|移花接木|不实|捏造|失实|误导|耸动|猎奇|拼接|冒用|无可靠|无权威|难以核实|可信度(很|较)?低)",
    },
}
CATS = ["memory_match", "plausibility", "skepticism"]


def rationale_onehot(rat, lang):
    hit = [1.0 if re.search(PAT[lang][c], rat or "", re.I) else 0.0 for c in CATS]
    return hit + [1.0 if not any(hit) else 0.0]  # + "other"


# ----------------------------------------------------------------- split assembly
def build_split(name, lang, split, enc_prefix=None):
    """Merged per-sample view of one split.

    Returns dict of aligned arrays: ids, y, sp(SLM pred), lp(LLM pred),
    prob(SLM prob_fake), ent, emb(768), rat(list[str]), rdict(n,4),
    rlen(n,), and — if enc_prefix given — e_news / e_rat (frozen
    multilingual sentence-encoder vectors from src_v3/encode_rationales.py).
    """
    sp_path, lp_path = preds_paths(name, split)
    S, L = load_rows(sp_path), load_rows(lp_path)
    ids = sorted(set(S) & set(L))
    y = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"] for i in ids])
    lp = np.array([L[i]["pred"] for i in ids])
    prob = np.array([float(S[i].get("prob_fake", S[i].get("prob", 0.5))) for i in ids])
    rat = [L[i].get("rationale", "") for i in ids]
    out = dict(
        name=name, lang=lang, split=split, ids=ids, y=y, sp=sp, lp=lp,
        prob=prob, ent=binary_entropy(prob), emb=get_emb(S, ids, sp_path),
        rat=rat, rdict=np.array([rationale_onehot(r, lang) for r in rat]),
        rlen=np.array([len(r or "") / 100.0 for r in rat]),
    )
    if enc_prefix:
        z = np.load(f"{enc_prefix}/{name}_{split}.npz", allow_pickle=True)
        idx = {int(k): j for j, k in enumerate(z["ids"])}
        out["e_news"] = np.array([z["e_news"][idx[i]] for i in ids], dtype=np.float32)
        out["e_rat"] = np.array([z["e_rat"][idx[i]] for i in ids], dtype=np.float32)
    return out


def interaction_block(e_news, e_rat):
    """SBERT-style (Reimers & Gurevych, EMNLP'19) pair-interaction features:
    [u, v, |u-v|, u*v, cos(u,v)] — the classic cross-pair block, here between
    the NEWS text and the LLM RATIONALE. Motivation (reports/08 F-L1): in
    hallucinated corroboration the rationale talks about the *context kernel*
    rather than the claim itself; interaction geometry is a cheap proxy."""
    nu = e_news / (np.linalg.norm(e_news, axis=1, keepdims=True) + 1e-9)
    nv = e_rat / (np.linalg.norm(e_rat, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(nu * nv, axis=1, keepdims=True)
    return np.hstack([e_news, e_rat, np.abs(e_news - e_rat), e_news * e_rat, cos])


# ----------------------------------------------------------------- metrics
def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f.append(2 * pr * rc / max(pr + rc, 1e-9))
    return 100.0 * float(np.mean(f))


def bootstrap_delta(y, pred_a, pred_b, n_boot=1000, seed=0):
    """CI for macroF1(pred_a) − macroF1(pred_b) by resampling test indices.
    Directly addresses reports/08 §4 caveat (i): small disagreement counts."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        deltas[b] = macro_f1(y[ix], pred_a[ix]) - macro_f1(y[ix], pred_b[ix])
    return dict(mean=float(deltas.mean()),
                lo=float(np.percentile(deltas, 2.5)),
                hi=float(np.percentile(deltas, 97.5)))


def class_weights_macro_f1(y, sp, cap=300, seed=0):
    """Marginal macro-F1 value of correcting one wrong gold-c sample (v2 asset,
    finite difference on val), normalised to mean 1. Handles fake/real imbalance."""
    rng = np.random.default_rng(seed)
    base = macro_f1(y, sp)
    w = {}
    for c in (0, 1):
        wrong = np.where((y == c) & (sp != c))[0]
        if len(wrong) == 0:
            w[c] = 1.0
            continue
        pick = rng.choice(wrong, min(cap, len(wrong)), replace=False)
        deltas = []
        for j in pick:
            p2 = sp.copy()
            p2[j] = y[j]
            deltas.append(macro_f1(y, p2) - base)
        w[c] = float(np.mean(deltas))
    m = np.mean([w[0], w[1]])
    return {c: (w[c] / m if m > 0 else 1.0) for c in (0, 1)}
