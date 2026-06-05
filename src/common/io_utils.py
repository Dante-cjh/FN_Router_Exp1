"""Shared IO helpers and label conventions.

Label convention (kept identical to the ARG official repo):
    real -> 0
    fake -> 1
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# Gold-label mapping. Accepts strings or ints, returns 0/1.
LABEL2ID = {"real": 0, "fake": 1, 0: 0, 1: 1, "0": 0, "1": 1}
ID2LABEL = {0: "real", 1: "fake"}

# LLM per-perspective judgment can be real / fake / other (3-class, ARG aux head).
LLM_LABEL2ID = {"real": 0, "fake": 1, "other": 2, 0: 0, 1: 1, 2: 2}


def normalize_gold_label(value: Any) -> int:
    """Map a gold label (str/int) to {0,1}."""
    if value in LABEL2ID:
        return LABEL2ID[value]
    v = str(value).strip().lower()
    if v in LABEL2ID:
        return LABEL2ID[v]
    raise ValueError(f"Unrecognized gold label: {value!r}")


def normalize_llm_label(value: Any) -> int:
    """Map an LLM judgment (real/fake/other) to {0,1,2}; unknown -> other(2)."""
    if value in LLM_LABEL2ID:
        return LLM_LABEL2ID[value]
    v = str(value).strip().lower()
    return LLM_LABEL2ID.get(v, 2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_item_id(item: Dict) -> Any:
    """ARG items use `source_id`; fall back to `id`."""
    if "source_id" in item:
        return item["source_id"]
    if "id" in item:
        return item["id"]
    raise KeyError("item has neither 'source_id' nor 'id'")


def get_item_content(item: Dict) -> str:
    for k in ("content", "text", "title"):
        if k in item and item[k]:
            return str(item[k])
    raise KeyError("item has no 'content'/'text'/'title' field")
