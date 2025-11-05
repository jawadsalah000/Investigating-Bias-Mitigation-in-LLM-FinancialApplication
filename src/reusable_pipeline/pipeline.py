"""High-level orchestration for the reusable bias-mitigation pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from reusable_pipeline.config import (
    ModelConfig,
    PipelineSettings,
    default_models,
    model_config_lookup,
)
from reusable_pipeline.data_preprocessor import DataPreprocessor
from reusable_pipeline.datasets import build_datasets
from reusable_pipeline.runners import gemini as gem_runner
from reusable_pipeline.runners import openai as gpt_runner
from reusable_pipeline.sampling import StratifiedSampler

PromptFn = Callable[[pd.Series], str]


PROMPT_FUNCTIONS: Dict[str, PromptFn] = {
    "baseline_no_social": gem_runner.prompt_baseline_no_social,
    "baseline_with_race_gender": gem_runner.prompt_baseline_with_social,
    "postcode_proxy": gem_runner.prompt_postcode_proxy,
    "location_baseline_no_social": gem_runner.prompt_baseline_no_social,
    "location_cot_no_social": gem_runner.prompt_chain_no_social,
    "name_proxy_default": gem_runner.prompt_name_default,
    "cot_no_social": gem_runner.prompt_chain_no_social,
    "cot_with_race": gem_runner.prompt_chain_with_race,
    "cot_engineered_with_race": gem_runner.prompt_chain_engineered_with_race,
    "name_baseline": gem_runner.prompt_name_baseline,
    "name_instruction": gem_runner.prompt_name_instruction,
    "name_cot": gem_runner.prompt_name_chain,
    "name_cot_engineered": gem_runner.prompt_name_chain_engineered,
    "location_instruction": gem_runner.prompt_location_instruction,
    "location_cot": gem_runner.prompt_location_chain,
    "location_cot_engineered": gem_runner.prompt_location_chain_engineered,
}


PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "baseline_no_social": {"dataset_key": "race_gender_explicit", "prompt_fn_key": "baseline_no_social"},
    "baseline_with_race_gender": {"dataset_key": "race_gender_explicit", "prompt_fn_key": "baseline_with_race_gender"},
    "name_proxy": {"dataset_key": "race_proxy_names", "prompt_fn_key": "name_proxy_default"},
    "postcode_proxy": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "postcode_proxy"},
    "location_cot_no_social": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "location_cot_no_social"},
    "location_baseline_no_social": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "location_baseline_no_social"},
    "cot_no_social": {"dataset_key": "race_gender_explicit", "prompt_fn_key": "cot_no_social"},
    "cot_with_race": {"dataset_key": "race_gender_explicit", "prompt_fn_key": "cot_with_race"},
    "cot_engineered_with_race": {"dataset_key": "race_gender_explicit", "prompt_fn_key": "cot_engineered_with_race"},
    "name_instruction": {"dataset_key": "race_proxy_names", "prompt_fn_key": "name_instruction"},
    "name_cot": {"dataset_key": "race_proxy_names", "prompt_fn_key": "name_cot"},
    "name_cot_engineered": {"dataset_key": "race_proxy_names", "prompt_fn_key": "name_cot_engineered"},
    "location_instruction": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "location_instruction"},
    "location_cot": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "location_cot"},
    "location_cot_engineered": {"dataset_key": "location_proxy_postcode", "prompt_fn_key": "location_cot_engineered"},
}


DEFAULT_DATASET_PROMPT_SELECTION: Dict[str, List[str]] = {
    "race_gender_explicit": [],
    "race_proxy_names": [],
    "location_proxy_postcode": ["location_cot_no_social"],
}

DEFAULT_RUN_SEEDS: Sequence[Optional[int]] = (42, 72, 123)
DEFAULT_RUN_COUNT: int = 1


@dataclass
class PipelineRunner:
    """Coordinates dataset preparation, prompt execution, and result collation."""

    settings: PipelineSettings = field(default_factory=PipelineSettings)
    models: List[ModelConfig] = field(default_factory=default_models)
    dataset_prompt_selection: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_DATASET_PROMPT_SELECTION.items()}
    )
    prompt_registry: Dict[str, Dict[str, str]] = field(default_factory=lambda: dict(PROMPT_REGISTRY))

    def __post_init__(self) -> None:
        self._model_lookup = model_config_lookup(self.models)

    # ------------------------------------------------------------------
    # Prompt selection helpers
    # ------------------------------------------------------------------

    def collect_selected_prompts(self) -> List[Tuple[str, str]]:
        selections: List[Tuple[str, str]] = []
        for dataset_key, prompt_list in self.dataset_prompt_selection.items():
            for prompt_name in prompt_list:
                selections.append((dataset_key, prompt_name))
        return selections

    def _resolve_prompt(self, prompt_name: str) -> Optional[Dict[str, object]]:
        entry = self.prompt_registry.get(prompt_name)
        if not entry:
            return None
        return {
            "dataset_key": entry["dataset_key"],
            "prompt_fn": PROMPT_FUNCTIONS.get(entry["prompt_fn_key"]),
        }

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _prepare_datasets(self, random_seed: Optional[int]) -> Dict[str, pd.DataFrame]:
        data_path = self.settings.data_source
        if not data_path.exists():
            raise FileNotFoundError(f"Data source not found at {data_path}")

        preprocessor = DataPreprocessor(str(data_path))
        full_df = preprocessor.run()

        seed_offset = int(random_seed or 0)

        features = [
            "income",
            "loan_amount",
            "loan_to_value_ratio",
            "debt_to_income_ratio",
            "property_value",
            "applicant_age",
        ]

        approvals = full_df[full_df["action_taken"] == 1]
        denials = full_df[full_df["action_taken"] == 2]
        if len(approvals) < 50 or len(denials) < 50:
            raise ValueError("Not enough approved/denied rows after preprocessing.")

        sampler_approved = StratifiedSampler(features, sample_size=50, n_bins=4, random_state=42 + seed_offset)
        sampler_denied = StratifiedSampler(features, sample_size=50, n_bins=4, random_state=99 + seed_offset)

        approved_100 = sampler_approved.sample(approvals)
        denied_100 = sampler_denied.sample(denials)

        base_100 = pd.concat([approved_100, denied_100], ignore_index=True)
        base_100 = base_100.sample(frac=1, random_state=1234 + seed_offset).reset_index(drop=True)

        approved_25 = approved_100.sample(n=25, random_state=2024 + seed_offset)
        denied_25 = denied_100.sample(n=25, random_state=2025 + seed_offset)
        base_50 = pd.concat([approved_25, denied_25], ignore_index=True)
        base_50 = base_50.sample(frac=1, random_state=2026 + seed_offset).reset_index(drop=True)

        datasets = build_datasets(base_50, base_100)

        proxy_df = datasets.get("race_proxy_names")
        if proxy_df is not None and "applicant_name" not in proxy_df.columns and "name" in proxy_df.columns:
            datasets["race_proxy_names"] = proxy_df.rename(columns={"name": "applicant_name"})

        postcode_df = datasets.get("location_proxy_postcode")
        if postcode_df is not None and "applicant_uk_postcode" not in postcode_df.columns:
            if "applicant UK postcode" in postcode_df.columns:
                datasets["location_proxy_postcode"] = postcode_df.rename(
                    columns={"applicant UK postcode": "applicant_uk_postcode"}
                )

        return datasets

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    def _run_gemini(
        self,
        model: ModelConfig,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        prompt_fn: PromptFn,
        seed: Optional[int],
    ) -> pd.DataFrame:
        import google.generativeai as genai

        if model.api_key:
            genai.configure(api_key=model.api_key)
        else:
            genai.configure(api_key=gem_runner._ensure_api_key())

        temperature = 0.0 if model.temperature is None else model.temperature
        gen_model = gem_runner._build_model(model_name=model_name, temperature=temperature)

        if seed is not None:
            print(f"  [info] seed={seed} ignored for Gemini models.")
        if model.reasoning_level is not None:
            print(f"  [info] reasoning_level='{model.reasoning_level}' ignored for Gemini models.")

        return gem_runner.run_scenario(
            name=scenario_name,
            df=df,
            prompt_fn=prompt_fn,
            model=gen_model,
            max_rows=self.settings.max_rows,
            tries_per_row=self.settings.tries_per_row,
            retry_delay=self.settings.retry_delay,
        )

    def _run_model(
        self,
        model_config: ModelConfig,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        prompt_fn: PromptFn,
        run_seed: Optional[int],
    ) -> pd.DataFrame:
        provider = model_config.provider.lower()
        seed = run_seed if run_seed is not None else model_config.seed

        if provider == "gemini" or model_name.startswith("gemini"):
            return self._run_gemini(model_config, model_name, scenario_name, df, prompt_fn, seed)

        if provider == "openai" or model_name.startswith("gpt"):
            return gpt_runner.run_scenario(
                model_name=model_name,
                scenario_name=scenario_name,
                df=df,
                prompt_fn=prompt_fn,
                max_rows=self.settings.max_rows,
                tries_per_row=self.settings.tries_per_row,
                retry_delay=self.settings.retry_delay,
                temperature=model_config.temperature,
                seed=seed,
                reasoning_level=model_config.reasoning_level,
                api_key=model_config.api_key,
            )

        raise NotImplementedError(f"Model runner not implemented for {model_name}")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        model_name: str,
        frames: List[pd.DataFrame],
        input_rate: Optional[float],
        output_rate: Optional[float],
        *,
        run_label: Optional[str] = None,
    ) -> None:
        model_dir = self.settings.output_dir / "models" / model_name
        combined_dir = model_dir / "Combined Results"
        raw_dir = model_dir / "Raw Results"

        if run_label:
            match = re.match(r"run_(\d+)", run_label.lower())
            run_suffix = match.group(1) if match else run_label
            target_dir = model_dir / f"Run {run_suffix}"
        else:
            target_dir = combined_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        raw_dir_for_write = (target_dir / "Raw Results") if run_label else raw_dir
        raw_dir_for_write.mkdir(parents=True, exist_ok=True)

        combined = pd.concat(frames, ignore_index=True)
        combined["cost_usd"] = _compute_cost_column(combined, input_rate, output_rate)

        if "run" not in combined.columns:
            combined["run"] = "run_1"

        combined.to_csv(raw_dir_for_write / "combined_raw_results.csv", index=False)

        table_df = _build_table_view(combined)
        table_df.to_csv(target_dir / "combined_table.csv", index=False)

        metrics_runs, metrics_summary = _compute_metrics(combined)
        if not metrics_runs.empty:
            metrics_runs.to_csv(target_dir / "combined_metrics_runs.csv", index=False)
        if not metrics_summary.empty:
            metrics_summary.to_csv(target_dir / "combined_metrics.csv", index=False)

        for scenario, frame in combined.groupby("scenario"):
            frame.to_csv(target_dir / f"{scenario}.csv", index=False)

        print_path = target_dir if run_label else combined_dir
        print(f"Saved results for model {model_name} under {print_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        run_seeds: Optional[Sequence[Optional[int]]] = None,
        run_count: int = 1,
    ) -> None:
        selected_prompts = self.collect_selected_prompts()
        if not selected_prompts:
            print("No prompts selected. Update dataset_prompt_selection before running.")
            return

        if run_seeds is not None:
            seeds: List[Optional[int]] = list(run_seeds)
        else:
            count = max(run_count, 1)
            seeds = list(range(1, count + 1))

        if not seeds:
            seeds = [None]

        self.settings.ensure_directories()

        results_by_model: Dict[str, List[pd.DataFrame]] = {model.name: [] for model in self.models}
        any_success = False

        for run_index, seed_value in enumerate(seeds, start=1):
            run_label = f"run_{run_index:02d}"
            dataset_seed = seed_value if seed_value is not None else run_index
            print(f"\n=== Starting {run_label} (dataset_seed={dataset_seed}) ===")

            try:
                datasets = self._prepare_datasets(random_seed=dataset_seed)
            except Exception as exc:
                print(f"  [error] dataset preparation failed: {exc}")
                continue

            for model in self.models:
                config = self._model_lookup[model.name]
                input_rate = config.input_token_rate
                output_rate = config.output_token_rate

                print(f"\n=== Running model: {model.name} ({run_label}) ===")
                frames: List[pd.DataFrame] = []

                for dataset_key, prompt_name in selected_prompts:
                    prompt_cfg = self._resolve_prompt(prompt_name)
                    if not prompt_cfg:
                        print(f"  [skip] prompt '{prompt_name}' is not recognized or dataset missing.")
                        continue

                    prompt_fn = prompt_cfg["prompt_fn"]
                    if prompt_fn is None:
                        print(f"  [skip] prompt '{prompt_name}' has no function mapped.")
                        continue

                    dataset_from_registry = prompt_cfg["dataset_key"]
                    if dataset_key != dataset_from_registry:
                        print(
                            f"  [warn] prompt '{prompt_name}' configured for dataset '{dataset_key}' but registry expects '{dataset_from_registry}'."
                        )
                    dataset = datasets.get(dataset_from_registry)
                    if dataset is None:
                        print(f"  [skip] dataset '{dataset_from_registry}' not available for prompt '{prompt_name}'.")
                        continue

                    working_df = dataset.copy()
                    print(f"  -> {prompt_name} on {len(working_df)} rows")

                    model_seed = seed_value if seed_value is not None else run_index

                    try:
                        result_df = self._run_model(config, model.name, prompt_name, working_df, prompt_fn, model_seed)
                        result_df["run"] = run_label
                        frames.append(result_df)
                    except NotImplementedError as exc:
                        print(f"  [skip] {exc}")
                    except Exception as exc:
                        print(f"  [error] prompt {prompt_name}: {exc}")

                if frames:
                    self._save_outputs(
                        model.name,
                        frames,
                        input_rate,
                        output_rate,
                        run_label=run_label,
                    )
                    results_by_model[model.name].extend(frames)
                    any_success = True
                else:
                    print(f"  [warn] No results saved for model {model.name} in {run_label}.")

        if not any_success:
            print("No models ran successfully.")
            return

        for model in self.models:
            frames = results_by_model.get(model.name)
            if frames:
                self._save_outputs(model.name, frames, model.input_token_rate, model.output_token_rate)

        _save_master_summary(results_by_model, self.settings.output_dir)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _compute_cost_column(
    results_df: pd.DataFrame,
    input_rate: Optional[float],
    output_rate: Optional[float],
) -> pd.Series:
    if input_rate is None and output_rate is None:
        return pd.Series([math.nan] * len(results_df))

    costs = []
    for _, row in results_df.iterrows():
        total_cost = 0.0
        has_value = False
        p_tok = row.get("prompt_tokens")
        o_tok = row.get("output_tokens")
        if input_rate is not None and pd.notna(p_tok):
            total_cost += float(p_tok) * input_rate
            has_value = True
        if output_rate is not None and pd.notna(o_tok):
            total_cost += float(o_tok) * output_rate
            has_value = True
        costs.append(total_cost if has_value else math.nan)
    return pd.Series(costs)


def _compute_metrics(combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "decision" not in combined.columns:
        return pd.DataFrame(), pd.DataFrame()

    working = combined.copy()
    working["_decision_numeric"] = pd.to_numeric(working["decision"], errors="coerce")
    if "run" not in working.columns:
        working["run"] = "run_1"

    metrics: List[Dict[str, object]] = []
    for (run_label, scenario), frame in working.groupby(["run", "scenario"]):
        valid = frame.dropna(subset=["_decision_numeric"])
        record: Dict[str, object] = {"run": run_label, "scenario": scenario}
        if not valid.empty:
            record["overall_rate"] = float(valid["_decision_numeric"].mean())
            if "variant" in valid.columns:
                variant_rates = valid.groupby("variant")["_decision_numeric"].mean()
                if not variant_rates.empty:
                    record["dp_gap"] = float(variant_rates.max() - variant_rates.min())
                    for variant, rate in variant_rates.items():
                        safe_key = re.sub(r"[^0-9a-zA-Z]+", "_", str(variant).lower()).strip("_")
                        if not safe_key:
                            safe_key = "variant"
                        record[f"{safe_key}_rate"] = float(rate)
        metrics.append(record)

    metrics_df = pd.DataFrame(metrics)
    summary_df = pd.DataFrame()
    if not metrics_df.empty:
        numeric_cols = [
            col for col in metrics_df.select_dtypes(include="number").columns if col not in {"run"}
        ]
        if numeric_cols:
            summary = metrics_df.groupby("scenario")[numeric_cols].agg(["mean", "std"]).reset_index()
            summary.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns.to_flat_index()
            ]
            summary_df = summary

    return metrics_df, summary_df


def _build_table_view(combined: pd.DataFrame) -> pd.DataFrame:
    dataset_exclude = {
        "scenario",
        "prompt",
        "raw_text",
        "raw_response",
        "json",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "implications",
        "sensitive_influence",
        "cost_usd",
        "decision",
        "justification",
        "error",
        "run",
        "elapsed_seconds",
    }
    dataset_cols = [c for c in combined.columns if c not in dataset_exclude]

    extra_cols = [
        col
        for col in [
            "decision",
            "justification",
            "error",
            "run",
            "cost_usd",
            "elapsed_seconds",
        ]
        if col in combined.columns
    ]

    ordered_cols = dataset_cols + [col for col in extra_cols if col not in dataset_cols]
    ordered_cols = list(dict.fromkeys(ordered_cols))
    return combined[ordered_cols]


def _save_master_summary(results_by_model: Dict[str, List[pd.DataFrame]], output_dir: Path) -> None:
    """Persist cross-model rollups mirroring the final notebook consolidation."""
    combined_frames: List[pd.DataFrame] = []
    per_model_metrics: List[pd.DataFrame] = []
    per_model_summary: List[pd.DataFrame] = []

    for model_name, frames in results_by_model.items():
        if not frames:
            continue
        model_df = pd.concat(frames, ignore_index=True).copy()
        model_df["model"] = model_name
        combined_frames.append(model_df)

        metrics_df, summary_df = _compute_metrics(model_df)
        if not metrics_df.empty:
            metrics_copy = metrics_df.copy()
            metrics_copy["model"] = model_name
            per_model_metrics.append(metrics_copy)
        if not summary_df.empty:
            summary_copy = summary_df.copy()
            summary_copy["model"] = model_name
            per_model_summary.append(summary_copy)

    if not combined_frames:
        return

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    master_df = pd.concat(combined_frames, ignore_index=True)
    master_df.to_csv(combined_dir / "all_models_runs.csv", index=False)

    master_table = _build_table_view(master_df)
    master_table.to_csv(combined_dir / "all_models_table.csv", index=False)

    metrics_all, summary_all = _compute_metrics(master_df)
    if not metrics_all.empty:
        metrics_all.to_csv(combined_dir / "all_models_metrics_runs.csv", index=False)
    if not summary_all.empty:
        summary_all.to_csv(combined_dir / "all_models_metrics_summary.csv", index=False)

    if per_model_metrics:
        pd.concat(per_model_metrics, ignore_index=True).to_csv(
            combined_dir / "per_model_metrics_runs.csv", index=False
        )
    if per_model_summary:
        pd.concat(per_model_summary, ignore_index=True).to_csv(
            combined_dir / "per_model_metrics_summary.csv", index=False
        )

    print(f"Saved consolidated summaries under {combined_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    run_seeds: Optional[Sequence[Optional[int]]] = DEFAULT_RUN_SEEDS,
    run_count: int = DEFAULT_RUN_COUNT,
) -> None:
    runner = PipelineRunner()
    runner.run(run_seeds=run_seeds, run_count=run_count)


if __name__ == "__main__":
    main()
