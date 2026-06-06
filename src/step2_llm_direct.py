"""Step 2 -- GPT-5.4 *direct* judge (the "large model" in the diagnostic).

Runs a single holistic real/fake judgment per test item and writes a unified
prediction file consumed by the diagnostic (step5):

    [{"id": <id>, "label": <gold 0/1>, "pred": <0/1, or -1 if unparseable>,
      "rationale": <str>}, ...]

pred == -1 means the LLM reply could not be parsed; the diagnostic treats it
as an incorrect prediction (label != -1 always), which is the fair default.

Resumable via a .jsonl cache keyed by id.

Usage:
    python -m src.step2_llm_direct \
        --input  data/en/test.json \
        --output outputs/preds/gossipcop_gpt54.json \
        --cache  outputs/cache/en_test_direct.jsonl \
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
from src.common.prompts import get_direct_prompt  # noqa: E402

_PRED2ID = {"real": 0, "fake": 1}


async def _query(client, language, item_id, gold, content, cache_path):
    system, user = get_direct_prompt(language, content)
    res = await client.predict(system, user)
    pred = _PRED2ID.get(res["prediction"], -1)
    row = {"id": item_id, "label": gold, "pred": pred,
           "rationale": res["rationale"]}
    append_jsonl(row, cache_path)
    return row


async def run(args):
    data = load_json(args.input)

    cache = {str(r["id"]): r for r in load_jsonl(args.cache)}

    todo = []
    for item in data:
        iid = get_item_id(item)
        if str(iid) not in cache:
            gold = normalize_gold_label(item["label"])
            content = get_item_content(item)[: args.max_chars]
            todo.append((iid, gold, content))

    print(f"[step2] {len(data)} items, {len(todo)} LLM calls "
          f"({len(cache)} cached).")

    if todo:
        client = LLMClient()  # constructed only when there's work to do
        coros = [_query(client, args.language, iid, gold, content, args.cache)
                 for (iid, gold, content) in todo]
        results = await gather_bounded(coros, desc="direct judge")
        for row in results:
            cache[str(row["id"])] = row

    out = []
    n_bad = 0
    for item in data:
        iid = get_item_id(item)
        gold = normalize_gold_label(item["label"])
        row = cache.get(str(iid), {"pred": -1, "rationale": ""})
        if row["pred"] == -1:
            n_bad += 1
        out.append({"id": iid, "label": gold, "pred": int(row["pred"]),
                    "rationale": row.get("rationale", "")})

    save_json(out, args.output)
    acc = sum(1 for r in out if r["pred"] == r["label"]) / max(len(out), 1)
    print(f"[step2] wrote {len(out)} preds -> {args.output} | "
          f"acc={acc:.4f} | unparseable={n_bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="test split json")
    ap.add_argument("--output", required=True, help="unified preds json")
    ap.add_argument("--cache", required=True, help="jsonl resume cache")
    ap.add_argument("--language", required=True, choices=["en", "zh"])
    ap.add_argument("--max_chars", type=int, default=4000)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
