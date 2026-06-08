"""Step 3 -- RoBERTa SLM baseline (the "small model" in the diagnostic).

A standalone fine-tuned text classifier on the news `content` only -- no LLM
rationales. This is the content-only small model that ARG augments with an
LLM advisor, so it is the right "small" leg of the diagnostic.

  - English (GossipCop): roberta-base
  - Chinese (Weibo21)  : hfl/chinese-roberta-wwm-ext

Reads the ARG-format train/val/test json (only `content`, `label`, id are
used). Early-stops on validation macro-F1 (same selection metric as ARG),
then dumps unified per-sample test predictions:

    [{"id":..., "label": <gold 0/1>, "pred": <0/1>,
      "prob": <P(fake)>, "prob_fake": <P(fake), alias>}, ...]

plus a sidecar `*_emb.npz` (arrays `ids`,`emb`,`prob`,`label`,`pred`) holding the
penultimate-layer [CLS] embeddings — features for the Step-C learned router.

Usage (on the 4090 server):
    python -m src.step3_train_roberta \
        --train data/en/train.json --val data/en/val.json --test data/en/test.json \
        --model_name roberta-base --language en \
        --output outputs/preds/gossipcop_roberta.json \
        --ckpt_dir outputs/ckpt/gossipcop_roberta
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.io_utils import (  # noqa: E402
    get_item_content, get_item_id, load_json, normalize_gold_label, save_json,
)

DEFAULT_MODEL = {"en": "roberta-base", "zh": "hfl/chinese-roberta-wwm-ext"}


def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split(path):
    data = load_json(path)
    ids, texts, labels = [], [], []
    for item in data:
        ids.append(get_item_id(item))
        texts.append(get_item_content(item))
        labels.append(normalize_gold_label(item["label"]))
    return ids, texts, labels


class NewsDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len,
                             padding="max_length", return_tensors="pt")
        import torch
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def evaluate(model, loader, device, return_emb=False):
    """Run the model over a loader.

    When ``return_emb`` is set we also pull the penultimate-layer [CLS] / <s>
    representation (last hidden state at token 0, 768-d) for every sample. That
    embedding is the natural feature vector for the Step-C learned router, so we
    persist it once here at test time instead of re-running the encoder later.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_logits, all_labels, all_emb = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=return_emb)
            all_logits.append(out.logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            if return_emb:
                # last hidden state, [CLS]/<s> token -> (B, hidden)
                cls = out.hidden_states[-1][:, 0, :]
                all_emb.append(cls.detach().cpu().numpy())
    logits = np.concatenate(all_logits, 0)
    labels = np.concatenate(all_labels, 0)
    probs = _softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    result = {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "acc": accuracy_score(labels, preds),
        "preds": preds,
        "probs": probs,
        "labels": labels,
    }
    if return_emb:
        result["emb"] = np.concatenate(all_emb, 0)
    return result


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--model_name", default=None,
                    help="HF model id; defaults per-language")
    ap.add_argument("--output", required=True, help="unified test preds json")
    ap.add_argument("--emb_output", default=None,
                    help="where to save penultimate-layer test embeddings "
                         "(.npz with arrays `ids`,`emb`). Defaults to the "
                         "--output path with `_emb.npz`. Pass 'none' to skip.")
    ap.add_argument("--ckpt_dir", default="outputs/ckpt/roberta")
    ap.add_argument("--max_len", type=int, default=170)  # matches ARG
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--early_stop", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=3759)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(args.seed)
    model_name = args.model_name or DEFAULT_MODEL[args.language]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step3] model={model_name} device={device}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    tr_ids, tr_x, tr_y = read_split(args.train)
    va_ids, va_x, va_y = read_split(args.val)
    te_ids, te_x, te_y = read_split(args.test)

    tr_loader = DataLoader(NewsDataset(tr_x, tr_y, tok, args.max_len),
                           batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(NewsDataset(va_x, va_y, tok, args.max_len),
                           batch_size=args.batch_size)
    te_loader = DataLoader(NewsDataset(te_x, te_y, tok, args.max_len),
                           batch_size=args.batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    total_steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, 0, total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    best_f1, best_epoch = -1.0, -1

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in tr_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = loss_fn(logits, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
        val = evaluate(model, va_loader, device)
        print(f"[step3] epoch {epoch} train_loss={running/len(tr_loader):.4f} "
              f"val_macroF1={val['macro_f1']:.4f} val_acc={val['acc']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save(model.state_dict(), best_path)
        elif epoch - best_epoch >= args.early_stop:
            print(f"[step3] early stop (best epoch {best_epoch}, "
                  f"val macroF1={best_f1:.4f})")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, te_loader, device, return_emb=True)
    print(f"[step3] TEST macroF1={test['macro_f1']:.4f} acc={test['acc']:.4f}")

    # `prob` == P(label=1) == P(fake); `prob_fake` kept as a backwards-compatible
    # alias so older consumers (step5) keep working.
    out = [{"id": te_ids[i], "label": int(test["labels"][i]),
            "pred": int(test["preds"][i]),
            "prob": float(test["probs"][i]),
            "prob_fake": float(test["probs"][i])}
           for i in range(len(te_ids))]
    save_json(out, args.output)
    print(f"[step3] wrote {len(out)} preds -> {args.output}")

    # Persist penultimate-layer embeddings for the Step-C learned router.
    emb_out = args.emb_output
    if emb_out is None:
        base, _ = os.path.splitext(args.output)
        emb_out = base + "_emb.npz"
    if str(emb_out).lower() != "none":
        os.makedirs(os.path.dirname(emb_out) or ".", exist_ok=True)
        np.savez_compressed(
            emb_out,
            ids=np.array(te_ids),
            emb=test["emb"].astype(np.float32),
            prob=test["probs"].astype(np.float32),
            label=test["labels"].astype(np.int64),
            pred=test["preds"].astype(np.int64),
        )
        print(f"[step3] wrote {test['emb'].shape} embeddings -> {emb_out}")


if __name__ == "__main__":
    main()
