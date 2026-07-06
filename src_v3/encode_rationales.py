#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src_v3/encode_rationales.py — frozen multilingual encodings of (news, rationale).

BORROWED MODULES:
  * ARG (Hu et al., AAAI 2024, arXiv:2309.12247) treats the LLM rationale as an
    object to be *judged* (rationale-usefulness evaluation) rather than trusted.
    We keep that idea but replace their trained cross-attention with frozen
    sentence embeddings + SBERT-style interaction features (cheap, and our
    disagreement-training sets are tiny: 187/530 samples).
  * SBERT pair-interaction block (Reimers & Gurevych, EMNLP 2019).

A single MULTILINGUAL encoder is used for both datasets so that the arbiter
can be transferred zh<->en (reports/08 M5: hallucinated corroboration is a
property of the LLM, not of the dataset — the transfer test makes that claim
falsifiable).

Runs on the GPU box (or CPU, it is small). Needs: pip install sentence-transformers

USAGE (from Router_Exp1/):
  python -m src_v3.encode_rationales --name weibo21  --lang zh --splits val test
  python -m src_v3.encode_rationales --name gossipcop --lang en --splits val test
Output: outputs_v3/enc/<name>_<split>.npz  {ids, e_news, e_rat}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src_v3.common import load_rows, preds_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="multilingual ST model; try BAAI/bge-m3 for a stronger (slower) one")
    ap.add_argument("--device", default=None, help="cuda / cpu / None=auto")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=1500, help="truncate news content")
    ap.add_argument("--out", default="outputs_v3/enc")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # deferred import
    enc = SentenceTransformer(args.model, device=args.device)
    os.makedirs(args.out, exist_ok=True)

    data = {d["source_id"]: d for d in json.load(open(f"data/{args.lang}/{'test' if 'test' in args.splits else 'val'}.json"))}
    for split in args.splits:
        data = {d["source_id"]: d for d in json.load(open(f"data/{args.lang}/{split}.json"))}
        sp_path, lp_path = preds_paths(args.name, split)
        S, L = load_rows(sp_path), load_rows(lp_path)
        ids = sorted(set(S) & set(L))
        news = [str(data[i]["content"])[: args.max_chars] for i in ids]
        rats = [L[i].get("rationale", "") or "" for i in ids]
        e_news = enc.encode(news, batch_size=args.batch, show_progress_bar=True,
                            normalize_embeddings=False)
        e_rat = enc.encode(rats, batch_size=args.batch, show_progress_bar=True,
                           normalize_embeddings=False)
        out = f"{args.out}/{args.name}_{split}.npz"
        np.savez_compressed(out, ids=np.array(ids), e_news=e_news, e_rat=e_rat,
                            model=args.model)
        print(f"[enc] {out}  n={len(ids)} dim={e_news.shape[1]}")


if __name__ == "__main__":
    main()
