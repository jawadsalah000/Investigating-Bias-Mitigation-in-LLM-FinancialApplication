"""Prompt utilities and Gemini runner used by the reusable pipeline."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Callable, Dict, List, Optional

import google.generativeai as genai
import pandas as pd
from google.api_core import exceptions as gax_exc

from reusable_pipeline.config import get_gemini_api_key
from reusable_pipeline.prompts import (
    Instruction_prompting_with_location,
    Instruction_prompting_with_name,
    NAME_PROXY_PROMPT_WITH_JUSTIFICATION,
    POSTCODE_PROXY_PROMPT,
    prompt_engineered_with_location_reason_first,
    prompt_engineered_with_name_reason_first,
    prompt_engineered_with_race_reason_first,
    prompt_template_baseline,
    prompt_template_no_race_gener,
    prompt_template_with_location_reason_first,
    prompt_template_with_name_reason_first,
    prompt_template_with_race_reason_first,
    prompt_template_with_race_with_gender,
)

# Regex to recover JSON blocks inside verbose responses.
JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*?\}")
# Transient errors we should retry internally.
RETRYABLE_EXC = (
    gax_exc.InternalServerError,
    gax_exc.ServiceUnavailable,
    gax_exc.DeadlineExceeded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_money(value: float) -> str:
    return f"${value:,.0f}"


def _format_ratio(value: float) -> str:
    return f"{value:.1f}"


def _ensure_api_key() -> str:
    return get_gemini_api_key()


def _build_model(model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.0):
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 256,
    }
    return genai.GenerativeModel(model_name, generation_config=generation_config)


def _call_model(model, prompt: str, max_retries: int = 2) -> Dict[str, object]:
    """Call Gemini and recover JSON decisions even inside verbose responses."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0},
                request_options={"timeout": 60},
            )
            text_raw = (getattr(response, "text", "") or "").strip()

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

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or 0

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


# ---------------------------------------------------------------------------
# Prompt wrappers
# ---------------------------------------------------------------------------


def prompt_baseline_no_social(row: pd.Series) -> str:
    return prompt_template_baseline.format(
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_baseline_with_social(row: pd.Series) -> str:
    return prompt_template_with_race_with_gender.format(
        derived_sex=row["derived_sex"],
        derived_race=row["derived_race"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_postcode_proxy(row: pd.Series) -> str:
    return POSTCODE_PROXY_PROMPT.format(
        applicant_uk_postcode=row["applicant_uk_postcode"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_chain_no_social(row: pd.Series) -> str:
    return prompt_template_no_race_gener.format(
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_chain_with_race(row: pd.Series) -> str:
    return prompt_template_with_race_reason_first.format(
        derived_sex=row["derived_sex"],
        derived_race=row["derived_race"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_chain_engineered_with_race(row: pd.Series) -> str:
    return prompt_engineered_with_race_reason_first.format(
        derived_sex=row["derived_sex"],
        derived_race=row["derived_race"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


# ---- Name proxy variations ----


def prompt_name_baseline(row: pd.Series) -> str:
    return NAME_PROXY_PROMPT_WITH_JUSTIFICATION.format(
        applicant_name=row.get("applicant_name", ""),
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_name_default(row: pd.Series) -> str:
    return NAME_PROXY_PROMPT_WITH_JUSTIFICATION.format(
        applicant_name=row.get("applicant_name", ""),
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_name_instruction(row: pd.Series) -> str:
    return Instruction_prompting_with_name.format(
        applicant_name=row.get("applicant_name", ""),
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_name_chain(row: pd.Series) -> str:
    return prompt_template_with_name_reason_first.format(
        applicant_name=row.get("applicant_name", ""),
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_name_chain_engineered(row: pd.Series) -> str:
    return prompt_engineered_with_name_reason_first.format(
        applicant_name=row.get("applicant_name", ""),
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


# ---- Location proxy variations ----


def prompt_location_instruction(row: pd.Series) -> str:
    return Instruction_prompting_with_location.format(
        applicant_uk_postcode=row["applicant_uk_postcode"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_location_chain(row: pd.Series) -> str:
    return prompt_template_with_location_reason_first.format(
        applicant_uk_postcode=row["applicant_uk_postcode"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


def prompt_location_chain_engineered(row: pd.Series) -> str:
    return prompt_engineered_with_location_reason_first.format(
        applicant_uk_postcode=row["applicant_uk_postcode"],
        loan_amount=_format_money(row["loan_amount"]),
        loan_to_value_ratio=_format_ratio(row["loan_to_value_ratio"]),
        property_value=_format_money(row["property_value"]),
        income=_format_money(row["income"]),
        debt_to_income_ratio=_format_ratio(row["debt_to_income_ratio"]),
        applicant_age=_format_ratio(row["applicant_age"]),
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_scenario(
    name: str,
    df: pd.DataFrame,
    prompt_fn: Callable[[pd.Series], str],
    model,
    max_rows: int,
    tries_per_row: int = 3,
    retry_delay: float = 0.3,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    subset = df.head(max_rows) if max_rows else df
    for idx, row in subset.iterrows():
        prompt = prompt_fn(row)
        response = None
        for attempt in range(tries_per_row):
            try:
                response = _call_model(model, prompt)
                decision = response.get("decision")
                if decision in (0, 1):
                    break
                if response.get("error") and attempt + 1 < tries_per_row:
                    time.sleep(retry_delay * (attempt + 1))
            except Exception as exc:  # ensure we always capture an error entry
                response = {
                    "raw_text": "",
                    "decision": None,
                    "raw_response": "",
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if attempt + 1 < tries_per_row:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    break
        if response is None:
            response = {
                "raw_text": "",
                "decision": None,
                "raw_response": "",
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": "no_response",
            }
        row_dict = {col: row[col] for col in df.columns}
        row_dict["scenario"] = name
        row_dict["prompt"] = prompt
        for key, value in response.items():
            row_dict[key] = value
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
        if (idx + 1) % 50 == 0:
            print(f"[{name}] processed {idx + 1} applications…")
    return pd.DataFrame(records)
