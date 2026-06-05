"""Step 1 -- Build the ARG rationale dataset with GPT-5.4 as the *advisor*.

Input : an ARG/raw split file (JSON list) where each item has at least
        `content`, `label`, and `source_id` (or `id`).
Output: the same split, augmented to full ARG `rationale` format with fields
        td_rationale, cs_rationale, td_pred, cs_pred, td_acc, cs_acc.

For each news item the advisor LLM is queried twice:
    - td : textual-description perspective  -> td_rationale / td_pred
    - cs : common-sense perspective         -> cs_rationale / cs_pred
*_acc = 1 if that perspective's prediction matches the gold label else 0.
`*_pred` is stored as a string in {"real","fake","other"} which the ARG
dataloader maps to {0,1,2}. "other" is used when the LLM reply is unparseable.

Resumable: every (id, perspective) result is appended to a .jsonl cache;
re-running skips work already cached.

Usage:
    python -m src.step1_generate_rationales \
        --input  data/en/raw/test.json \
        --output data/en/test.json \
        --cache  outputs/cache/en_test_rationales.jsonl \
        --language en
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.io_utils import (  # noqa: E402
    append_jsonl, get_item_content, get_item_id, load_json, load_jsonl,
    normalize_gold_label, save_json,
)
from src.common.llm_client import LLMClient, gather_bounded  # noqa: E402
from src.common.prompts import get_perspective_prompt  # noqa: E402

PERSPECTIVES = ("td", "cs")


async def _query(client: LLMClient, language: str, perspective: str,
                 item_id, content: str, cache_path: str) -> dict:
    system, user = get_perspective_prompt(language, perspective, content)
    res = await client.predict(system, user)
    pred = res["prediction"] if res["prediction"] in ("real", "fake") else "other"
    row = {
        "id": item_id,
        "perspective": perspective,
        "pred": pred,
        "rationale": res["rationale"],
    }
    append_jsonl(row, cache_path)
    return row


async def run(args):
    data = load_json(args.input)
    client = LLMClient()

    # Load cache: {(id, perspective): row}
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

    print(f"[step1] {len(data)} items, {len(todo)} LLM calls to make "
          f"({len(cache)} cached).")

    if todo:
        coros = [_query(client, args.language, p, iid, content, args.cache)
                 for (iid, p, content) in todo]
        results = await gather_bounded(coros, desc="advisor rationales")
        for row in results:
            cache[(str(row["id"]), row["perspective"])] = row

    # Assemble ARG-format output.
    out = []
    n_missing = 0
    for item in data:
        iid = get_item_id(item)
        gold = normalize_gold_label(item["label"])
        rec = dict(item)
        rec["content"] = get_item_content(item)
        rec["source_id"] = iid
        rec["label"] = gold
        for p, idx in (("td", "2"), ("cs", "3")):
            row = cache.get((str(iid), p))
            if row is None:
                n_missing += 1
                pred, rationale = "other", ""
            else:
                pred, rationale = row["pred"], row["rationale"]
            pred_id = {"real": 0, "fake": 1, "other": 2}[pred]
            acc = 1 if pred_id == gold else 0
            rec[f"{p}_rationale"] = rationale
            rec[f"{p}_pred"] = pred
            rec[f"{p}_acc"] = acc
        out.append(rec)

    save_json(out, args.output)
    print(f"[step1] wrote {len(out)} items -> {args.output} "
          f"({n_missing} perspective(s) missing/failed).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw/ARG split json (list)")
    ap.add_argument("--output", required=True, help="ARG-format json to write")
    ap.add_argument("--cache", required=True, help="jsonl resume cache path")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--max_chars", type=int, default=4000,
                    help="truncate very long articles before sending")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
