"""OpenAI runner used by the reusable bias-mitigation pipeline."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Callable, Dict, List, Optional

import pandas as pd
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from reusable_pipeline.config import get_openai_api_key

# Regex to recover JSON blocks inside verbose responses.
JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*?\}")
# Transient errors we should retry internally.
RETRYABLE_EXC = (
    APIError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


def _call_model(
    *,
    client,
    model_name: str,
    prompt: str,
    temperature: Optional[float],
    seed: Optional[int],
    reasoning_level: Optional[str],
    max_retries: int = 2,
) -> Dict[str, object]:
    """Call OpenAI GPT models and recover JSON decisions even inside verbose responses."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, object] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if seed is not None:
                kwargs["seed"] = seed
            if reasoning_level is not None:
                kwargs["reasoning_effort"] = reasoning_level

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            text_raw = (getattr(message, "content", "") or "").strip()

            parsed_json: Dict[str, object] = {}
            decision: Optional[int] = None
            match = JSON_BLOCK_PATTERN.search(text_raw)
            if match:
                try:
                    candidate = json.loads(match.group(0))
                    if isinstance(candidate, dict):
                        parsed_json = candidate
                        val = candidate.get("decision")
                        if str(val) in ("0", "1", 0, 1):
                            decision = int(val)
                except Exception:
                    parsed_json = {}

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            error = None if decision in (0, 1) else "no_decision_or_invalid_json"
            return {
                "raw_text": text_raw,
                "raw_response": text_raw,
                "json": parsed_json,
                "decision": decision,
                "prompt_tokens": prompt_tokens or None,
                "output_tokens": output_tokens or None,
                "total_tokens": total_tokens or None,
                "error": error,
            }
        except RETRYABLE_EXC as exc:
            if attempt == max_retries:
                return {
                    "raw_text": "",
                    "raw_response": "",
                    "json": {},
                    "decision": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2
        except Exception as exc:
            return {
                "raw_text": "",
                "raw_response": "",
                "json": {},
                "decision": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "raw_text": "",
        "raw_response": "",
        "json": {},
        "decision": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "error": "no_response",
    }


def run_scenario(
    *,
    model_name: str,
    scenario_name: str,
    df: pd.DataFrame,
    prompt_fn: Callable[[pd.Series], str],
    max_rows: int,
    tries_per_row: int = 3,
    retry_delay: float = 0.3,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    reasoning_level: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    key = api_key or get_openai_api_key()
    client = OpenAI(api_key=key)
    records: List[Dict[str, object]] = []
    subset = df.head(max_rows) if max_rows else df
    for idx, row in subset.iterrows():
        prompt = prompt_fn(row)
        response = None
        for attempt in range(tries_per_row):
            response = _call_model(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                seed=seed,
                reasoning_level=reasoning_level,
                max_retries=2,
            )
            decision = response.get("decision")
            if decision in (0, 1):
                break
            if response.get("error") and attempt + 1 < tries_per_row:
                time.sleep(retry_delay * (attempt + 1))
        if response is None:
            response = {
                "raw_text": "",
                "raw_response": "",
                "json": {},
                "decision": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": "no_response",
            }

        row_dict = {col: row[col] for col in df.columns}
        row_dict["scenario"] = scenario_name
        row_dict["prompt"] = prompt
        for key_name, value in response.items():
            row_dict[key_name] = value
        json_payload = response.get("json")
        if isinstance(json_payload, dict):
            justification = json_payload.get("justification")
            if justification is not None:
                row_dict["justification"] = justification
            implications = json_payload.get("implications")
            if implications is not None:
                try:
                    row_dict["implications"] = json.dumps(implications)
                except Exception:
                    row_dict["implications"] = str(implications)
            sensitive = json_payload.get("sensitive_influence")
            if sensitive is not None:
                row_dict["sensitive_influence"] = sensitive
        records.append(row_dict)

    return pd.DataFrame.from_records(records)
