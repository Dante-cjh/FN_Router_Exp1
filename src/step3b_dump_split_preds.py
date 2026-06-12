"""Step 3b -- dump RoBERTa predictions (+ penultimate emb) for an ARBITRARY split,
using an already-trained checkpoint from step3.

step3 only ever dumps TEST predictions. step7 (learned router) needs the SAME
unified format for the VAL split too (to train the gain-predictor without
touching test). This script reuses the trained `best.pt` checkpoint and just
runs inference + dumps:

    [{"id":..., "label": <gold 0/1>, "pred": <0/1>,
      "prob": <P(fake)>, "prob_fake": <P(fake), alias>}, ...]

plus a sidecar `*_emb.npz` (arrays `ids`,`emb`,`prob`,`label`,`pred`).

Usage (on the 4090 server, same env as step3):
    python -m src.step3b_dump_split_preds \
        --input data/zh/val.json --language zh \
        --model_name hfl/chinese-roberta-wwm-ext \
        --ckpt_dir outputs/ckpt/weibo21_roberta \
        --output outputs/preds/weibo21_roberta_val.json

    python -m src.step3b_dump_split_preds \
        --input data/en/val.json --language en \
        --model_name roberta-base \
        --ckpt_dir outputs/ckpt/gossipcop_roberta \
        --output outputs/preds/gossipcop_roberta_val.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.io_utils import save_json  # noqa: E402
from src.step3_train_roberta import (  # noqa: E402
    DEFAULT_MODEL, NewsDataset, evaluate, read_split, set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="split json (e.g. data/zh/val.json)")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--model_name", default=None, help="HF model id; defaults per-language")
    ap.add_argument("--ckpt_dir", required=True, help="dir containing best.pt from step3")
    ap.add_argument("--output", required=True, help="unified preds json for this split")
    ap.add_argument("--emb_output", default=None,
                    help="where to save penultimate-layer embeddings (.npz). "
                         "Defaults to --output path with `_emb.npz`. Pass 'none' to skip.")
    ap.add_argument("--max_len", type=int, default=170)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3759)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(args.seed)
    model_name = args.model_name or DEFAULT_MODEL[args.language]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step3b] model={model_name} device={device} ckpt_dir={args.ckpt_dir}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[step3b] loaded {ckpt_path}")

    ids, x, y = read_split(args.input)
    loader = DataLoader(NewsDataset(x, y, tok, args.max_len), batch_size=args.batch_size)

    result = evaluate(model, loader, device, return_emb=True)
    print(f"[step3b] {args.input}: macroF1={result['macro_f1']:.4f} acc={result['acc']:.4f} "
          f"n={len(ids)}")

    out = [{"id": ids[i], "label": int(result["labels"][i]),
            "pred": int(result["preds"][i]),
            "prob": float(result["probs"][i]),
            "prob_fake": float(result["probs"][i])}
           for i in range(len(ids))]
    save_json(out, args.output)
    print(f"[step3b] wrote {len(out)} preds -> {args.output}")

    emb_out = args.emb_output
    if emb_out is None:
        base, _ = os.path.splitext(args.output)
        emb_out = base + "_emb.npz"
    if str(emb_out).lower() != "none":
        os.makedirs(os.path.dirname(emb_out) or ".", exist_ok=True)
        np.savez_compressed(
            emb_out,
            ids=np.array(ids),
            emb=result["emb"].astype(np.float32),
            prob=result["probs"].astype(np.float32),
            label=result["labels"].astype(np.int64),
            pred=result["preds"].astype(np.int64),
        )
        print(f"[step3b] wrote {result['emb'].shape} embeddings -> {emb_out}")


if __name__ == "__main__":
    main()
