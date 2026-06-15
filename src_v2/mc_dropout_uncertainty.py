#!/usr/bin/env python
"""
src_v2/mc_dropout_uncertainty.py — Round-2 experiment, step A.

Offline epistemic/aleatoric uncertainty DECOUPLING for the SLM, via MC Dropout.
This is the cheap (zero-API) pilot from reports/04_uncertainty_routing_plan.md §3.

Idea (BALD / mutual-information decomposition, Houlsby et al. / Gal):
  reuse the ALREADY-TRAINED RoBERTa checkpoint (outputs/ckpt/<ds>_roberta/best.pt),
  keep dropout ON at inference, run T stochastic forward passes, and split the
  predictive uncertainty into:

    p_bar(y|x) = (1/T) Σ_t p(y|x, θ_t)                      # MC predictive dist
    U_tot(x)   = H[p_bar]                                   # total / predictive
    U_ale(x)   = (1/T) Σ_t H[p(y|x, θ_t)]                   # aleatoric (irreducible)
    U_epi(x)   = U_tot − U_ale  =  I(y; θ | x)              # epistemic (reducible)

U_tot is exactly the signal the old confidence-threshold router used (single-pass
entropy ≈ U_tot). The hypothesis under test is that U_epi is a BETTER routing
signal and U_ale flags the irreducible (both-models-wrong) band.

ISOLATION: this is round-2 code. It only READS the v1 checkpoints/data and writes
everything under outputs_v2/. It imports stable infra (tokenizer/dataset/data
readers) from the v1 `src` package; all new experiment logic lives here.

USAGE (on the GPU server, same env as step3):
  python -m src_v2.mc_dropout_uncertainty \
      --input data/zh/test.json --language zh \
      --model_name hfl/chinese-roberta-wwm-ext \
      --ckpt_dir outputs/ckpt/weibo21_roberta \
      --name weibo21 --split test \
      --out outputs_v2/uncertainty --T 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# --- repo root on path so we can reuse the v1 infra (NOT v1 experiment logic) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)

from src.common.io_utils import save_json  # noqa: E402
from src.step3_train_roberta import (  # noqa: E402
    DEFAULT_MODEL, NewsDataset, read_split, set_seed,
)


def _softmax_t(logits):
    """Row-wise softmax for a (B, 2) torch tensor -> numpy (B, 2)."""
    import torch
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def enable_mc_dropout(model, dropout_p=None):
    """Put the model in eval mode but RE-ENABLE every Dropout layer.

    RoBERTa uses LayerNorm (deterministic), so the only stochasticity left is the
    dropout layers — this is the standard MC-Dropout recipe.

    If ``dropout_p`` is given, every Dropout layer's rate is OVERRIDDEN to that
    value (calibration experiment A1: amplify inference dropout to fight the
    epistemic-uncertainty collapse of an over-confident fine-tuned model).
    """
    model.eval()
    n = 0
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            if dropout_p is not None:
                m.p = float(dropout_p)
            m.train()
            n += 1
    return n


def entropy_rows(p, eps=1e-12):
    """Shannon entropy (nats) of each row of a (..., C) probability array."""
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def mc_forward(model, loader, device, T, temperature=1.0, dropout_p=None):
    """Return per-pass class probabilities, shape (T, N, 2).

    ``temperature`` (calibration experiment A2) divides the logits before the
    softmax so two differently-calibrated backbones can be put on the same scale
    before the U_ale / U_epi magnitudes are compared across datasets.
    """
    import torch
    n_drop = enable_mc_dropout(model, dropout_p)
    print(f"[mc] enabled {n_drop} dropout layers (p_override={dropout_p}); "
          f"T={T} temperature={temperature}")
    passes = []
    with torch.no_grad():
        for t in range(T):
            probs_t = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                probs_t.append(_softmax_t(logits / temperature))
            passes.append(np.concatenate(probs_t, 0))
            if (t + 1) % 5 == 0 or t == T - 1:
                print(f"[mc]   pass {t + 1}/{T}")
    return np.stack(passes, 0)  # (T, N, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="split json (e.g. data/zh/test.json)")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--model_name", default=None, help="HF id; defaults per-language")
    ap.add_argument("--ckpt_dir", required=True, help="dir with best.pt from v1 step3")
    ap.add_argument("--name", required=True, help="dataset tag, e.g. weibo21 / gossipcop")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", default="outputs_v2/uncertainty", help="output dir")
    ap.add_argument("--T", type=int, default=30, help="number of MC-Dropout passes")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="divide logits by this before softmax (A2 calibration)")
    ap.add_argument("--dropout_p", type=float, default=None,
                    help="override every Dropout rate at inference (A1 calibration)")
    ap.add_argument("--tag", default="",
                    help="suffix added to output filename to keep configs separate, "
                         "e.g. p0.3 / temp / p0.3_temp")
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
    print(f"[mc] model={model_name} device={device} ckpt={args.ckpt_dir} "
          f"name={args.name} split={args.split}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)
    state = torch.load(os.path.join(args.ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(state)

    ids, x, y = read_split(args.input)
    loader = DataLoader(NewsDataset(x, y, tok, args.max_len),
                        batch_size=args.batch_size)

    probs = mc_forward(model, loader, device, args.T,
                       temperature=args.temperature,
                       dropout_p=args.dropout_p)               # (T, N, 2)
    p_bar = probs.mean(0)                                       # (N, 2)
    U_tot = entropy_rows(p_bar)                                 # (N,)
    U_ale = entropy_rows(probs).mean(0)                         # (N,) mean over T
    U_epi = U_tot - U_ale                                       # (N,)
    pred_mc = p_bar.argmax(1).astype(int)
    prob_mc = p_bar[:, 1].astype(float)
    y = np.array(y, dtype=int)

    # sanity: epistemic is a (small) non-negative quantity up to MC noise
    neg = int((U_epi < -1e-6).sum())
    print(f"[mc] N={len(ids)}  U_epi<0 (MC noise) = {neg}  "
          f"mean U_tot={U_tot.mean():.4f} U_ale={U_ale.mean():.4f} "
          f"U_epi={U_epi.mean():.4f}")
    acc_mc = float((pred_mc == y).mean())
    print(f"[mc] MC-mean accuracy = {acc_mc:.4f} (sanity vs deterministic head)")

    odir = args.out
    os.makedirs(odir, exist_ok=True)
    rows = [{
        "id": ids[i],
        "label": int(y[i]),
        "pred_mc": int(pred_mc[i]),
        "prob_mc": float(prob_mc[i]),
        "U_tot": float(U_tot[i]),
        "U_ale": float(U_ale[i]),
        "U_epi": float(U_epi[i]),
    } for i in range(len(ids))]
    suffix = f"__{args.tag}" if args.tag else ""
    out_json = os.path.join(odir, f"{args.name}_{args.split}{suffix}.json")
    save_json(rows, out_json)
    np.savez_compressed(
        os.path.join(odir, f"{args.name}_{args.split}{suffix}.npz"),
        ids=np.array(ids), label=y, pred_mc=pred_mc, prob_mc=prob_mc,
        U_tot=U_tot, U_ale=U_ale, U_epi=U_epi, T=args.T,
        temperature=args.temperature,
        dropout_p=(args.dropout_p if args.dropout_p is not None else -1.0),
    )
    print(f"[mc] wrote {len(rows)} rows -> {out_json} (+ .npz)")


if __name__ == "__main__":
    main()
