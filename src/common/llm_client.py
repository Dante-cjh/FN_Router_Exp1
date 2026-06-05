"""Async OpenAI-compatible LLM client driven by the .env file.

Reads these variables (see .env.example):
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME
    OPENAI_MAX_RETRIES, OPENAI_RETRY_DELAY, OPENAI_MAX_CONCURRENT

Designed for the diagnostic experiment: bounded concurrency, retry with
backoff, and tolerant JSON parsing of the {"prediction","rationale"} replies.
The same client also works against a local vLLM endpoint -- just point
OPENAI_BASE_URL/OPENAI_MODEL_NAME at the VLLM_* values.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError as e:  # pragma: no cover
    raise ImportError("Please `pip install openai>=1.0` to use llm_client.") from e

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; env vars may already be exported.
    pass


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrent: int = 10
    temperature: Optional[float] = None  # None -> omit (safer for newer models)

    @classmethod
    def from_env(cls, prefix: str = "OPENAI") -> "LLMConfig":
        def g(name, default=None):
            return os.getenv(f"{prefix}_{name}", default)

        base_url = g("BASE_URL")
        api_key = g("API_KEY")
        model = g("MODEL_NAME")
        if not base_url or not api_key or not model:
            raise RuntimeError(
                f"Missing {prefix}_BASE_URL / {prefix}_API_KEY / {prefix}_MODEL_NAME "
                "in environment (.env)."
            )
        temp = os.getenv(f"{prefix}_TEMPERATURE")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_retries=int(g("MAX_RETRIES", "3")),
            retry_delay=float(g("RETRY_DELAY", "2")),
            max_concurrent=int(g("MAX_CONCURRENT", "10")),
            temperature=float(temp) if temp not in (None, "") else None,
        )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(text: str) -> Tuple[Optional[str], str]:
    """Extract (prediction, rationale) from a model reply.

    Returns prediction in {"real","fake"} or None if unparseable.
    """
    if not text:
        return None, ""
    raw = text.strip()
    # Strip markdown fences.
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    if isinstance(obj, dict):
        pred = str(obj.get("prediction", "")).strip().lower()
        rationale = str(obj.get("rationale", "")).strip()
        pred = _coerce_label(pred)
        return pred, rationale

    # Fallback: keyword scan.
    low = raw.lower()
    pred = _coerce_label(low)
    return pred, raw


def _coerce_label(s: str) -> Optional[str]:
    s = s.lower()
    has_fake = "fake" in s or "虚假" in s or "假" in s
    has_real = "real" in s or "true" in s or "真实" in s or "真" in s
    if has_fake and not has_real:
        return "fake"
    if has_real and not has_fake:
        return "real"
    # If both/neither, prefer an exact token match.
    if s.strip() in ("fake", "real"):
        return s.strip()
    return None


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    async def _chat_once(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def chat(self, system: str, user: str) -> str:
        """Single call with retry/backoff. Returns raw content ('' on failure)."""
        async with self._sem:
            delay = self.config.retry_delay
            last_err = None
            for attempt in range(self.config.max_retries):
                try:
                    return await self._chat_once(system, user)
                except Exception as e:  # broad: network, rate-limit, 5xx
                    last_err = e
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
            print(f"[LLMClient] giving up after {self.config.max_retries} tries: {last_err}")
            return ""

    async def predict(self, system: str, user: str) -> Dict:
        """Call the model and parse a {prediction, rationale} reply."""
        raw = await self.chat(system, user)
        pred, rationale = parse_prediction(raw)
        return {"prediction": pred, "rationale": rationale, "raw": raw}


async def gather_bounded(coros: List, desc: str = "") -> List:
    """Run coroutines (already concurrency-limited by the client's semaphore)
    with a tqdm progress bar if available."""
    try:
        from tqdm.asyncio import tqdm_asyncio
        return await tqdm_asyncio.gather(*coros, desc=desc)
    except Exception:
        return await asyncio.gather(*coros)
