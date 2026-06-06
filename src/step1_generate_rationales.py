"""Step 1 -- Regenerate ARG rationales with GPT-5.4 as the *advisor*.

This makes ARG's LLM advisor == GPT-5.4, so all three diagnostic legs share the
same large model (ARG advisor, and the step2 direct judge).

For each news item the advisor LLM is queried twice:
    - td : textual-description perspective  -> td_rationale / td_pred
    - cs : common-sense perspective         -> cs_rationale / cs_pred
*_acc = 1 if that perspective's prediction matches the gold label else 0.
"other" (==2) is used when the LLM reply is unparseable.

Only these six fields are (over)written:
    td_rationale, cs_rationale, td_pred, cs_pred, td_acc, cs_acc
Everything else in each item (content, label, source_id, split, time, ...) is
preserved exactly. `td_pred/cs_pred` are written in the *original* convention:
ints (0/1/2) for en, strings (real/fake/other) for zh -- both accepted by ARG's
dataloader. Override with --pred_format.

In-place mode (--in_place) overwrites the original dataset file, after making a
one-time `<file>.bak` backup, and writes atomically (temp + rename) so an
interrupted run can't corrupt the data. The (id, perspective) jsonl cache makes
the expensive API pass fully resumable.

Usage (in-place over the official ARG data):
    python -m src.step1_generate_rationales \
        --input data/zh/test.json --in_place \
        --cache outputs/cache/zh_test_rat.jsonl --language zh
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.io_utils import (  # noqa: E402
    append_jsonl, get_item_content, get_item_id, load_json, load_jsonl,
    normalize_gold_label,
)
from src.common.llm_client import LLMClient, gather_bounded  # noqa: E402
from src.common.prompts import get_perspective_prompt  # noqa: E402

PERSPECTIVES = ("td", "cs")
_PRED_STR2ID = {"real": 0, "fake": 1, "other": 2}


def _fmt_pred(pred_str: str, pred_format: str):
    """en/int -> 0/1/2 ; zh/str -> 'real'/'fake'/'other'."""
    if pred_format == "int":
        return _PRED_STR2ID[pred_str]
    return pred_str


def _atomic_write_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


async def _query(client, language, perspective, item_id, content, cache_path):
    system, user = get_perspective_prompt(language, perspective, content)
    res = await client.predict(system, user)
    pred = res["prediction"] if res["prediction"] in ("real", "fake") else "other"
    row = {"id": item_id, "perspective": perspective, "pred": pred,
           "rationale": res["rationale"]}
    append_jsonl(row, cache_path)
    return row


async def run(args):
    in_path = args.input
    out_path = in_path if args.in_place else args.output
    if not out_path:
        raise SystemExit("provide --output or use --in_place")

    pred_format = args.pred_format
    if pred_format == "auto":
        pred_format = "int" if args.language == "en" else "str"

    data = load_json(in_path)

    cache = {}
    for row in load_jsonl(args.cache):
        cache[(str(row["id"]), row["perspective"])] = row

    todo = []
    for item in data:
        iid = get_item_id(item)
        content = get_item_content(item)[: args.max_chars]
        for p in PERSPECTIVES:
            if (str(iid), p) not in cache:
                todo.append((iid, p, content))

    print(f"[step1] {in_path}: {len(data)} items, {len(todo)} LLM calls "
          f"({len(cache)} cached). pred_format={pred_format}")

    if todo:
        client = LLMClient()  # constructed only when there's work to do
        coros = [_query(client, args.language, p, iid, content, args.cache)
                 for (iid, p, content) in todo]
        results = await gather_bounded(coros, desc="advisor rationales")
        for row in results:
            cache[(str(row["id"]), row["perspective"])] = row

    # One-time backup before overwriting the original dataset.
    if args.in_place and not args.no_backup:
        bak = in_path + ".bak"
        if not os.path.exists(bak):
            with open(in_path, "r", encoding="utf-8") as src, \
                 open(bak, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print(f"[step1] backup -> {bak}")

    # Overwrite ONLY the six rationale fields; preserve everything else.
    out = []
    n_missing = 0
    for item in data:
        iid = get_item_id(item)
        gold = normalize_gold_label(item["label"])
        rec = dict(item)
        for p in PERSPECTIVES:
            row = cache.get((str(iid), p))
            if row is None:
                n_missing += 1
                pred_str, rationale = "other", ""
            else:
                pred_str, rationale = row["pred"], row["rationale"]
            acc = 1 if _PRED_STR2ID[pred_str] == gold else 0
            rec[f"{p}_rationale"] = rationale
            rec[f"{p}_pred"] = _fmt_pred(pred_str, pred_format)
            rec[f"{p}_acc"] = acc
        out.append(rec)

    _atomic_write_json(out, out_path)
    # Quick advisor-quality readout (how often each perspective was correct).
    td_acc = sum(r["td_acc"] for r in out) / max(len(out), 1)
    cs_acc = sum(r["cs_acc"] for r in out) / max(len(out), 1)
    print(f"[step1] wrote {len(out)} items -> {out_path} | "
          f"td_acc={td_acc:.3f} cs_acc={cs_acc:.3f} | "
          f"{n_missing} perspective(s) failed/unparseable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ARG split json (list)")
    ap.add_argument("--output", help="where to write (omit if --in_place)")
    ap.add_argument("--in_place", action="store_true",
                    help="overwrite --input (with a one-time .bak backup)")
    ap.add_argument("--no_backup", action="store_true",
                    help="skip the .bak backup in --in_place mode")
    ap.add_argument("--cache", required=True, help="jsonl resume cache path")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--pred_format", choices=["auto", "int", "str"],
                    default="auto",
                    help="td_pred/cs_pred encoding; auto = int(en)/str(zh)")
    ap.add_argument("--max_chars", type=int, default=4000,
                    help="truncate very long articles before sending")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
