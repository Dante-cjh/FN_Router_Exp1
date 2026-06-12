"""merge_emb.py -- merge a step3/step3b `*_emb.npz` sidecar into its unified
preds json so step7_learned_router.py's `build()` picks up the `emb` feature
path (it checks `"emb" in row` on the JSON dict; step3/3b never write that key
into the json itself, only into the npz sidecar).

Writes the merged json IN PLACE by default (adds an `"emb": [...]` list of
floats to every row whose id is found in the npz). Rows missing from the npz
are left untouched (and would break step7's `all("emb" in S[i] ...)` check --
there shouldn't be any, since step3/3b dump both from the same loader pass).

Usage:
    python -m src.merge_emb \
        --preds outputs/preds/weibo21_roberta.json \
        --emb   outputs/preds/weibo21_roberta_emb.npz

    python -m src.merge_emb \
        --preds outputs/preds/weibo21_roberta_val.json \
        --emb   outputs/preds/weibo21_roberta_val_emb.npz

Repeat for gossipcop_roberta(.json/_val.json) + their _emb.npz files.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.io_utils import load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="unified preds json (e.g. *_roberta.json)")
    ap.add_argument("--emb", required=True, help="sidecar *_emb.npz (arrays ids, emb)")
    ap.add_argument("--output", default=None, help="defaults to overwriting --preds in place")
    args = ap.parse_args()

    rows = load_json(args.preds)
    npz = np.load(args.emb)
    ids = npz["ids"]
    emb = npz["emb"]
    id2emb = {int(i): emb[k].tolist() for k, i in enumerate(ids)}

    missing = 0
    for r in rows:
        e = id2emb.get(int(r["id"]))
        if e is None:
            missing += 1
            continue
        r["emb"] = e

    out_path = args.output or args.preds
    save_json(rows, out_path)
    print(f"[merge_emb] {len(rows)} rows, emb dim={emb.shape[1]}, "
          f"{len(rows) - missing} merged, {missing} missing -> {out_path}")
    if missing:
        print("[merge_emb] WARNING: some rows have no emb; "
              "step7's has_emb check requires ALL rows to have it.")


if __name__ == "__main__":
    main()
