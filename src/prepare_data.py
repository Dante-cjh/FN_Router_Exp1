"""Validate + preprocess the raw downloads into ARG-format JSON splits.

Inputs (under datasets/):
    datasets/weibo21/{train,val,test}.pkl     pandas DF: content,label(0/1),category
    datasets/fakenewsnet/gossipcop_{fake,real}.csv   id,news_url,title,tweet_ids

Outputs:
    data/zh/{train,val,test}.json     Weibo21  -> ARG base format
    data/en/{train,val,test}.json     GossipCop -> ARG base format
    outputs/prep/report.md            human-readable validation report
    outputs/prep/summary.json         machine-readable stats

ARG base item schema (rationale fields are added later by step1):
    {"source_id": <int, unique & numeric>, "content": <str>, "label": <0|1>,
     "category": <str, zh only>, "orig_id": <str, en only>}

Notes
-----
* `source_id` is a fresh global integer (ARG's dataloader casts ids to a
  tensor, so they must be numeric). The original GossipCop id is kept as
  `orig_id` for traceability.
* GossipCop here is FakeNewsNet **title-only** (the CSVs carry no article body).
  We deduplicate titles and drop titles that appear with *both* labels, then do
  a stratified 70/10/20 split. The official ARG GossipCop uses fuller text; if
  you later obtain it, drop it in as data/en/ and skip the gossipcop step.

Usage:
    python -m src.prepare_data --dataset all
    python -m src.prepare_data --dataset weibo21
    python -m src.prepare_data --dataset gossipcop --test_size 0.2 --val_size 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.io_utils import normalize_gold_label, save_json  # noqa: E402

DATASETS_DIR = "datasets"
OUT_EN = "data/en"
OUT_ZH = "data/zh"
PREP_DIR = "outputs/prep"

MIN_LEN = 2            # drop near-empty content
_GID = {"n": 0}       # global numeric source_id counter


def _next_id() -> int:
    i = _GID["n"]
    _GID["n"] += 1
    return i


def _len_stats(texts):
    import numpy as np
    L = np.array([len(t) for t in texts]) if texts else np.array([0])
    return {"mean": round(float(L.mean()), 1), "min": int(L.min()),
            "max": int(L.max()), "p95": int(np.percentile(L, 95))}


def _label_counts(items):
    c = {0: 0, 1: 0}
    for it in items:
        c[it["label"]] += 1
    return {"real(0)": c[0], "fake(1)": c[1]}


# --------------------------------------------------------------------------- #
# Weibo21
# --------------------------------------------------------------------------- #
def prepare_weibo21(report, keep_canonical_split=False):
    src = os.path.join(DATASETS_DIR, "weibo21")
    stats = {"source": src, "keep_canonical_split": keep_canonical_split,
             "splits": {}}

    # Load all splits first (so we can remove cross-split duplicate content).
    raw = {}
    for split in ("train", "val", "test"):
        df = pd.read_pickle(os.path.join(src, f"{split}.pkl"))
        rows, dropped_short = [], 0
        for _, row in df.iterrows():
            content = str(row["content"]).strip()
            if len(content) < MIN_LEN:
                dropped_short += 1
                continue
            rows.append({"content": content,
                         "label": normalize_gold_label(row["label"]),
                         "category": str(row.get("category", ""))})
        raw[split] = {"rows": rows, "raw_rows": len(df),
                      "dropped_short": dropped_short}

    # Cross-split dedup (default ON): a post in train wins; a post in val wins
    # over test. Removes leakage that would unfairly boost the *trained* models
    # (RoBERTa/ARG) relative to the zero-shot GPT-5.4 in the diagnostic.
    dropped_leak = {"val": 0, "test": 0}
    if not keep_canonical_split:
        seen = set(r["content"] for r in raw["train"]["rows"])
        for split in ("val", "test"):
            kept = []
            for r in raw[split]["rows"]:
                if r["content"] in seen:
                    dropped_leak[split] += 1
                else:
                    kept.append(r)
                    seen.add(r["content"])
            raw[split]["rows"] = kept

    for split in ("train", "val", "test"):
        items = []
        for r in raw[split]["rows"]:
            items.append({"source_id": _next_id(), "content": r["content"],
                          "label": r["label"], "category": r["category"]})
        save_json(items, os.path.join(OUT_ZH, f"{split}.json"))
        stats["splits"][split] = {
            "raw_rows": raw[split]["raw_rows"], "kept": len(items),
            "dropped_short": raw[split]["dropped_short"],
            "dropped_crosssplit_leak": dropped_leak.get(split, 0),
            "labels": _label_counts(items),
            "content_len": _len_stats([it["content"] for it in items]),
        }
        print(f"[weibo21] {split}: {raw[split]['raw_rows']} -> {len(items)} kept "
              f"(short {raw[split]['dropped_short']}, "
              f"leak {dropped_leak.get(split, 0)})")
    report["weibo21"] = stats
    return stats


# --------------------------------------------------------------------------- #
# GossipCop (FakeNewsNet CSVs)
# --------------------------------------------------------------------------- #
def prepare_gossipcop(report, test_size, val_size, seed):
    from sklearn.model_selection import train_test_split
    src = os.path.join(DATASETS_DIR, "fakenewsnet")
    fake = pd.read_csv(os.path.join(src, "gossipcop_fake.csv"))
    real = pd.read_csv(os.path.join(src, "gossipcop_real.csv"))
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    n_raw = len(df)

    df["title"] = df["title"].astype(str).str.strip()

    # 1) drop empty / too-short titles
    empty_mask = df["title"].str.len() < MIN_LEN
    n_empty = int(empty_mask.sum())
    df = df[~empty_mask]

    # 2) drop titles that appear with BOTH labels (conflicting)
    conflict = (df.groupby("title")["label"].nunique() > 1)
    conflict_titles = set(conflict[conflict].index)
    n_conflict_rows = int(df["title"].isin(conflict_titles).sum())
    df = df[~df["title"].isin(conflict_titles)]

    # 3) drop exact-duplicate titles (keep first) -> prevents train/test leakage
    before = len(df)
    df = df.drop_duplicates(subset="title", keep="first").reset_index(drop=True)
    n_dup = before - len(df)

    # 4) stratified split: train / (val+test), then val / test
    rel_val = val_size / (1.0 - test_size)
    train_df, tmp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df["label"],
        random_state=seed)
    val_df, test_df = train_test_split(
        tmp_df, test_size=(test_size / (val_size + test_size)),
        stratify=tmp_df["label"], random_state=seed)

    stats = {
        "source": src, "raw_rows": n_raw,
        "dropped_empty": n_empty,
        "dropped_conflict_titles": len(conflict_titles),
        "dropped_conflict_rows": n_conflict_rows,
        "dropped_duplicate_titles": n_dup,
        "kept_total": len(df),
        "split_ratio": {"train": round(1 - val_size - test_size, 3),
                        "val": val_size, "test": test_size},
        "seed": seed, "splits": {},
    }

    for split, sdf in (("train", train_df), ("val", val_df), ("test", test_df)):
        items = []
        for _, row in sdf.iterrows():
            items.append({
                "source_id": _next_id(),
                "content": row["title"],
                "label": int(row["label"]),
                "orig_id": str(row["id"]),
            })
        save_json(items, os.path.join(OUT_EN, f"{split}.json"))
        stats["splits"][split] = {
            "kept": len(items), "labels": _label_counts(items),
            "content_len": _len_stats([it["content"] for it in items]),
        }
        print(f"[gossipcop] {split}: {len(items)} kept "
              f"({_label_counts(items)})")
    report["gossipcop"] = stats
    return stats


# --------------------------------------------------------------------------- #
# Validation of written files
# --------------------------------------------------------------------------- #
def validate_outputs(out_dir, name, report):
    """Re-load written splits; check id uniqueness, label domain, no content
    leakage across splits."""
    issues = []
    seen_ids, content_by_split = set(), {}
    total = 0
    for split in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{split}.json")
        if not os.path.exists(path):
            issues.append(f"missing {path}")
            continue
        items = json.load(open(path, encoding="utf-8"))
        total += len(items)
        contents = set()
        for it in items:
            if it["source_id"] in seen_ids:
                issues.append(f"duplicate source_id {it['source_id']} in {split}")
            seen_ids.add(it["source_id"])
            if it["label"] not in (0, 1):
                issues.append(f"bad label {it['label']} in {split}")
            contents.add(it["content"])
        content_by_split[split] = contents
    # cross-split content leakage
    splits = list(content_by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            ov = content_by_split[splits[i]] & content_by_split[splits[j]]
            if ov:
                issues.append(f"content leakage {splits[i]}∩{splits[j]}: {len(ov)}")
    report.setdefault("validation", {})[name] = {
        "total_items": total, "unique_source_ids": len(seen_ids),
        "issues": issues or ["none"],
    }
    status = "OK" if not issues else f"{len(issues)} ISSUE(S)"
    print(f"[validate] {name}: {total} items, {len(seen_ids)} unique ids -> {status}")
    for x in issues:
        print(f"    ! {x}")


def write_report(report, path):
    lines = ["# Data preparation & validation report\n"]
    if "weibo21" in report:
        w = report["weibo21"]
        lines.append("## Weibo21 (zh) — data/zh/\n")
        mode = ("canonical split kept" if w.get("keep_canonical_split")
                else "cross-split duplicate posts removed from val/test")
        lines.append(f"- Mode: **{mode}**\n")
        lines.append("| Split | raw | kept | leak removed | real(0) | fake(1) | len mean/p95/max |")
        lines.append("|---|---|---|---|---|---|---|")
        for s, d in w["splits"].items():
            cl = d["content_len"]
            lines.append(f"| {s} | {d['raw_rows']} | {d['kept']} | "
                         f"{d.get('dropped_crosssplit_leak', 0)} | "
                         f"{d['labels']['real(0)']} | {d['labels']['fake(1)']} | "
                         f"{cl['mean']}/{cl['p95']}/{cl['max']} |")
        lines.append("")
    if "gossipcop" in report:
        g = report["gossipcop"]
        lines.append("## GossipCop (en) — data/en/\n")
        lines.append(f"- Raw rows (fake+real): **{g['raw_rows']}**")
        lines.append(f"- Dropped: empty {g['dropped_empty']}, "
                     f"conflicting titles {g['dropped_conflict_titles']} "
                     f"({g['dropped_conflict_rows']} rows), "
                     f"duplicate titles {g['dropped_duplicate_titles']}")
        lines.append(f"- Kept total: **{g['kept_total']}**, "
                     f"stratified split {g['split_ratio']} (seed {g['seed']})")
        lines.append("- ⚠️ Content is **title-only** (FakeNewsNet CSVs have no "
                     "article body).\n")
        lines.append("| Split | kept | real(0) | fake(1) | len mean/p95/max |")
        lines.append("|---|---|---|---|---|")
        for s, d in g["splits"].items():
            cl = d["content_len"]
            lines.append(f"| {s} | {d['kept']} | {d['labels']['real(0)']} | "
                         f"{d['labels']['fake(1)']} | "
                         f"{cl['mean']}/{cl['p95']}/{cl['max']} |")
        lines.append("")
    lines.append("## Validation\n")
    for name, v in report.get("validation", {}).items():
        lines.append(f"- **{name}**: {v['total_items']} items, "
                     f"{v['unique_source_ids']} unique source_ids — "
                     f"issues: {', '.join(v['issues'])}")
    lines.append("\n## Next step\n")
    lines.append("Data is ready. To make the ARG advisor GPT-5.4, run step1 on "
                 "each split (see scripts/run_*.sh); otherwise proceed to "
                 "step2/step3/step5.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["all", "weibo21", "gossipcop"],
                    default="all")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3759)
    ap.add_argument("--keep_canonical_split", action="store_true",
                    help="keep the official Weibo21 split as-is (do NOT remove "
                         "cross-split duplicate posts). Off by default.")
    args = ap.parse_args()

    report = {}
    if args.dataset in ("all", "weibo21"):
        prepare_weibo21(report, keep_canonical_split=args.keep_canonical_split)
        validate_outputs(OUT_ZH, "weibo21", report)
    if args.dataset in ("all", "gossipcop"):
        prepare_gossipcop(report, args.test_size, args.val_size, args.seed)
        validate_outputs(OUT_EN, "gossipcop", report)

    os.makedirs(PREP_DIR, exist_ok=True)
    save_json(report, os.path.join(PREP_DIR, "summary.json"))
    write_report(report, os.path.join(PREP_DIR, "report.md"))
    print(f"\n[prepare_data] report -> {PREP_DIR}/report.md")


if __name__ == "__main__":
    main()
