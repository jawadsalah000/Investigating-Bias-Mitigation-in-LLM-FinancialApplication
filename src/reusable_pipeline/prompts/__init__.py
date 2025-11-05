"""Prompt template helpers for the reusable pipeline."""

from .baseline import (
    prompt_template_baseline,
    prompt_template_with_race_with_gender,
    prompt_template_no_race_gener,
    prompt_template_with_race_reason_first,
    prompt_engineered_with_race_reason_first,
)
from .locations import (
    POSTCODE_METADATA,
    POSTCODE_PROXY_PROMPT,
    Instruction_prompting_with_location,
    prompt_template_with_location_reason_first,
    prompt_engineered_with_location_reason_first,
)
from .names import (
    NAME_PROXY_PROMPT_WITH_JUSTIFICATION,
    Instruction_prompting_with_name,
    prompt_template_with_name_reason_first,
    prompt_engineered_with_name_reason_first,
)

__all__ = [
    "Instruction_prompting_with_location",
    "Instruction_prompting_with_name",
    "NAME_PROXY_PROMPT_WITH_JUSTIFICATION",
    "POSTCODE_METADATA",
    "POSTCODE_PROXY_PROMPT",
    "prompt_engineered_with_location_reason_first",
    "prompt_engineered_with_name_reason_first",
    "prompt_engineered_with_race_reason_first",
    "prompt_template_baseline",
    "prompt_template_no_race_gener",
    "prompt_template_with_location_reason_first",
    "prompt_template_with_name_reason_first",
    "prompt_template_with_race_reason_first",
    "prompt_template_with_race_with_gender",
]
