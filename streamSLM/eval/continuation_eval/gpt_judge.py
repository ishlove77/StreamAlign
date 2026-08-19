"""GPT-4o semantic-coherence judge.

Mirrors the rubric used by
``/home/TASTE-SpokenLM/gpt-4o-analysis.py`` so scores
are directly comparable to that file's `gpt-4o-stats.ipynb` baselines.

The OpenAI API key is read from ``~/.env`` (supports the
``export KEY="..."`` shell-syntax line). The key is never logged.
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Optional


ENV_PATH = Path.home() / ".env"
DEFAULT_MODEL = "gpt-4o-2024-08-06"

EVAL_SYSTEM_PROMPT = (
    "You are an assistant that evaluates the relevance and likelihood "
    "of a text continuation given the text prompt.\n"
    "Use the following rubric:\n"
    "1: very unlikely and irrelevant\n"
    "2: unlikely and marginally relevant\n"
    "3: moderately likely and relevant\n"
    "4: likely and relevant\n"
    "5: very likely and highly relevant\n"
    "First, briefly analyze the sample. Then, output exactly in the form: "
    "I would rate the score as _;"
)
_SCORE_RE = re.compile(r"I would rate the score as\s*([1-5])")


def _load_openai_key(env_path: Path = ENV_PATH) -> str:
    """Read OPENAI_API_KEY from ~/.env without printing it.

    Supports lines of the form ``OPENAI_API_KEY=...`` or
    ``export OPENAI_API_KEY="..."``. Falls back to the process env.
    """
    if env_path.is_file():
        pat = re.compile(
            r'^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*["\']?(?P<v>[^"\'\n]+)["\']?\s*$'
        )
        for line in env_path.read_text().splitlines():
            m = pat.match(line)
            if m:
                return m.group("v")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found in {env_path} or in process env"
        )
    return key


def _score_one(
    client,
    model: str,
    prompt_text: str,
    continuation_text: str,
    max_retries: int = 5,
) -> tuple[Optional[int], str]:
    """Return (score, raw_response). score is None on parse failure."""
    user_msg = (
        f'Text prompt:\n"{prompt_text}"\n'
        f'Text continuation:\n"{continuation_text}"'
    )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                top_p=1.0,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            m = _SCORE_RE.search(raw)
            if not m:
                raise ValueError(f"no rating in response: {raw[:200]!r}")
            score = int(m.group(1))
            return score, raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(min(30.0, sleep))
    print(f"[gpt-judge] gave up after {max_retries} retries: {last_err}",
          flush=True)
    return None, ""


def score_pairs(
    pairs: list[tuple[str, str, str]],
    model: str = DEFAULT_MODEL,
) -> dict[str, dict]:
    """Score a list of (key, prompt_text, continuation_text) tuples.

    Returns ``{key: {"score": int|None, "rationale": str}}`` where
    ``rationale`` is the full free-form analysis preceding the rating.
    """
    from openai import OpenAI

    api_key = _load_openai_key()
    client = OpenAI(api_key=api_key)
    print(f"[gpt-judge] model={model} n={len(pairs)}", flush=True)

    results: dict[str, dict] = {}
    t0 = time.time()
    for i, (key, prompt_text, cont_text) in enumerate(pairs):
        score, raw = _score_one(client, model, prompt_text, cont_text)
        results[key] = {"score": score, "rationale": raw}
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[gpt-judge] {i + 1}/{len(pairs)} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return results
