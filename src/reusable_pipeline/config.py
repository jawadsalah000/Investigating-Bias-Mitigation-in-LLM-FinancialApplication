"""Shared configuration helpers for the reusable pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hmda_state_ga.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

ENV_GEMINI_KEY = "GEMINI_API_KEY"
ENV_OPENAI_KEY = "OPENAI_API_KEY"


def _read_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise RuntimeError(f"{key} is not configured. Export it before running the pipeline.")
    return value


def get_gemini_api_key() -> str:
    """Return the Gemini API key from the environment."""
    return _read_env(ENV_GEMINI_KEY)


def get_openai_api_key() -> str:
    """Return the OpenAI API key from the environment."""
    return _read_env(ENV_OPENAI_KEY)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Provide a cached OpenAI client configured with the project key."""
    return OpenAI(api_key=get_openai_api_key())


# ---------------------------------------------------------------------------
# Pipeline configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration options for a single model endpoint."""

    name: str
    provider: str
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    reasoning_level: Optional[str] = None
    input_token_rate: Optional[float] = None
    output_token_rate: Optional[float] = None


@dataclass
class PipelineSettings:
    """Top-level knobs that control how the pipeline runs."""

    data_source: Path = field(default_factory=lambda: Path(os.environ.get("PIPELINE_DATA_SOURCE", DEFAULT_DATA_PATH)))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("PIPELINE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    max_rows: int = int(os.environ.get("PIPELINE_MAX_ROWS", "0"))
    tries_per_row: int = int(os.environ.get("PIPELINE_TRIES_PER_ROW", "3"))
    retry_delay: float = float(os.environ.get("PIPELINE_RETRY_DELAY", "0.3"))

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def default_models() -> List[ModelConfig]:
    """Return the default model configurations used by the dissertation experiments."""
    return [
        ModelConfig(name="gpt-5", provider="openai", reasoning_level="minimal"),
        ModelConfig(name="gpt-5-nano", provider="openai", reasoning_level="minimal"),
        ModelConfig(name="gemini-2.5-flash-lite", provider="gemini", temperature=0.0),
    ]


def model_config_lookup(models: List[ModelConfig]) -> Dict[str, ModelConfig]:
    return {model.name: model for model in models}
