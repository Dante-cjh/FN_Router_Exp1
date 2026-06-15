#!/usr/bin/env python
"""
src_v2/fit_temperature.py — Round-2 calibration experiment A2.

Fit a single temperature T* (Guo et al. 2017) on the VAL split for an
already-trained RoBERTa checkpoint, by minimising validation NLL of
softmax(logits / T). Reports ECE before/after so we can see how mis-calibrated
each backbone was. The fitted T* is then fed to mc_dropout_uncertainty.py
(--temperature) so the two differently-calibrated backbones (zh vs en) sit on a
comparable scale before we trust the cross-dataset U_ale / reducible-fraction
numbers from reports/05.

ISOLATION: reads v1 ckpt + data only; writes outputs_v2/calibration/.

USAGE:
  python -m src_v2.fit_temperature \
      --val data/zh/val.json --language zh \
      --model_name hfl/chinese-roberta-wwm-ext \
      --ckpt_dir outputs/ckpt/weibo21_roberta \
      --name weibo21 --out outputs_v2/calibration
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)

from src.step3_train_roberta import (  # noqa: E402
    DEFAULT_MODEL, NewsDataset, read_split, set_seed,
)


def collect_logits(model, loader, device):
    import torch
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            logits_all.append(out.logits.detach().cpu())
            labels_all.append(batch["labels"])
    import torch as T
    return T.cat(logits_all, 0), T.cat(labels_all, 0)


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error on the predicted (max-prob) class."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def softmax_np(logits, t=1.0):
    z = (logits / t)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="outputs_v2/calibration")
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
    print(f"[temp] model={model_name} device={device} ckpt={args.ckpt_dir}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best.pt"),
                                     map_location=device))

    ids, x, y = read_split(args.val)
    loader = DataLoader(NewsDataset(x, y, tok, args.max_len),
                        batch_size=args.batch_size)
    logits, labels = collect_logits(model, loader, device)

    # --- fit T by LBFGS on validation NLL ---
    logT = torch.zeros(1, requires_grad=True, device="cpu")  # T = exp(logT) > 0
    lg = logits.float()
    lb = labels.long()
    nll = torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nll(lg / logT.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    Tstar = float(logT.exp().item())

    np_logits = lg.numpy()
    np_labels = lb.numpy()
    p_before = softmax_np(np_logits, 1.0)
    p_after = softmax_np(np_logits, Tstar)

    def nll_np(p):
        return float(-np.log(np.clip(p[np.arange(len(np_labels)), np_labels],
                                     1e-12, 1)).mean())

    result = {
        "name": args.name, "n_val": int(len(np_labels)),
        "temperature": Tstar,
        "nll_before": nll_np(p_before), "nll_after": nll_np(p_after),
        "ece_before": ece(p_before, np_labels), "ece_after": ece(p_after, np_labels),
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.name}_temperature.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[temp] T*={Tstar:.4f}  NLL {result['nll_before']:.4f}->{result['nll_after']:.4f}"
          f"  ECE {result['ece_before']:.4f}->{result['ece_after']:.4f}")
    print(f"[temp] wrote {out_path}")
    print(f"[temp] feed this to: mc_dropout_uncertainty.py --temperature {Tstar:.4f} --tag temp")


if __name__ == "__main__":
    main()
