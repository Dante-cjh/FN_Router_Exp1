"""Dump per-sample ARG test predictions in the unified diagnostic format.

WHY THIS FILE EXISTS
--------------------
The official ARG repo trains and prints aggregate metrics but never saves
aligned per-sample (id, gold, pred) for the test split (its predict() returns
an empty `id` list). The diagnostic needs per-sample predictions keyed by
source_id, so this script re-runs the trained ARG model on test.json and dumps:

    [{"id": <source_id>, "label": <gold 0/1>, "pred": <0/1>,
      "prob_fake": <float>}, ...]

HOW TO USE  (run from INSIDE the cloned ARG repo, after training ARG)
--------------------------------------------------------------------
1) Train ARG first (this produces ./param_model/ARG_en-arg/1/parameter_bert.pkl):
       bash run_en.sh        # or run_zh.sh
2) Copy this file into the ARG repo root (next to main.py), then:
       python dump_arg_predictions.py \
           --root_path /path/to/en-data \
           --bert_path /path/to/bert-base-uncased \
           --ckpt ./param_model/ARG_en-arg/1/parameter_bert.pkl \
           --language en \
           --output gossipcop_arg.json
3) Move gossipcop_arg.json into Router_Exp1/outputs/preds/ for step5.

It imports ARG's own modules (models.arg, utils.*), so the prediction is
produced by the exact trained network -- no re-implementation.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

# These imports resolve against the ARG repo this file is dropped into.
from models.arg import ARGModel
from utils.dataloader import get_dataloader
from utils.utils import data2gpu


def build_config(args):
    return {
        "use_cuda": torch.cuda.is_available(),
        "batchsize": args.batchsize,
        "max_len": args.max_len,
        "emb_dim": 768,
        "co_attention_dim": 300,
        "bert_path": args.bert_path,
        "language": args.language,
        "data_type": "rationale",
        "model": {
            "mlp": {"dims": [384], "dropout": 0.2},
            "llm_judgment_predictor_weight": 1.0,
            "rationale_usefulness_evaluator_weight": 1.5,
            "kd_loss_weight": 1.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", required=True,
                    help="dir containing test.json (ARG rationale format)")
    ap.add_argument("--test_name", default="test.json")
    ap.add_argument("--bert_path", required=True)
    ap.add_argument("--ckpt", required=True, help="trained parameter_bert.pkl")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=170)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = build_config(args)

    test_path = os.path.join(args.root_path, args.test_name)
    loader = get_dataloader(test_path, config["max_len"], config["batchsize"],
                            shuffle=False, bert_path=config["bert_path"],
                            data_type="rationale", language=args.language)

    model = ARGModel(config)
    if config["use_cuda"]:
        model = model.cuda()
    state = torch.load(args.ckpt, map_location="cuda" if config["use_cuda"] else "cpu")
    model.load_state_dict(state)
    model.eval()

    ids, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = data2gpu(batch, config["use_cuda"], data_type="rationale")
            res = model(**{**config, **bd})
            p = res["classify_pred"].detach().cpu().numpy().tolist()
            y = bd["label"].detach().cpu().numpy().tolist()
            i = bd["id"].detach().cpu().numpy().tolist()
            probs.extend(p)
            labels.extend(y)
            ids.extend(i)

    out = []
    for iid, y, p in zip(ids, labels, probs):
        out.append({"id": iid, "label": int(y),
                    "pred": int(p >= 0.5), "prob_fake": float(p)})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    acc = sum(1 for r in out if r["pred"] == r["label"]) / max(len(out), 1)
    print(f"[dump_arg] wrote {len(out)} preds -> {args.output} | acc={acc:.4f}")


if __name__ == "__main__":
    main()
