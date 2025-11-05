"""Command-line interface for the reusable pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from reusable_pipeline.config import PipelineSettings, default_models, model_config_lookup
from reusable_pipeline.pipeline import (
    DEFAULT_DATASET_PROMPT_SELECTION,
    DEFAULT_RUN_COUNT,
    DEFAULT_RUN_SEEDS,
    PipelineRunner,
    PROMPT_REGISTRY,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reusable LLM bias-mitigation pipeline end-to-end.",
    )
    parser.add_argument("--data-source", type=Path, help="Path to the HMDA CSV slice to preprocess.")
    parser.add_argument("--output-dir", type=Path, help="Directory to store model outputs.")
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional cap on rows per scenario (0 means full dataset).",
    )
    parser.add_argument(
        "--tries-per-row",
        type=int,
        help="Number of retries per row when a model fails to produce a decision.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        help="Delay in seconds between retries when models fail to respond.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model name to run (can be provided multiple times). Defaults to all configured models.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt scenario to include (can be provided multiple times). Defaults to curated selection.",
    )
    parser.add_argument(
        "--run-seed",
        dest="run_seeds",
        action="append",
        type=int,
        help="Specific random seed(s) for dataset sampling. Overrides --run-count if provided.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        help="Number of sequential runs to execute when --run-seed is not provided.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt scenario keys and exit.",
    )
    return parser


def _list_prompts() -> None:
    print("Available prompt scenarios:\n")
    for name, entry in sorted(PROMPT_REGISTRY.items()):
        dataset = entry.get("dataset_key", "unknown")
        prompt_key = entry.get("prompt_fn_key", "")
        print(f"- {name:<30} dataset={dataset:<24} fn={prompt_key}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_prompts:
        _list_prompts()
        return

    settings = PipelineSettings()
    if args.data_source is not None:
        settings.data_source = args.data_source.resolve()
    if args.output_dir is not None:
        settings.output_dir = args.output_dir.resolve()
    if args.max_rows is not None:
        settings.max_rows = args.max_rows
    if args.tries_per_row is not None:
        settings.tries_per_row = args.tries_per_row
    if args.retry_delay is not None:
        settings.retry_delay = args.retry_delay

    available_models = model_config_lookup(default_models())
    if args.models:
        selected_models: List = []
        unknown_models: List[str] = []
        for model_name in args.models:
            config = available_models.get(model_name)
            if config:
                selected_models.append(config)
            else:
                unknown_models.append(model_name)
        if unknown_models:
            parser.error(f"Unknown model(s): {', '.join(sorted(set(unknown_models)))}")
        models = selected_models
    else:
        models = list(available_models.values())

    if args.prompts:
        selection = {key: [] for key in DEFAULT_DATASET_PROMPT_SELECTION}
        unknown_prompts: List[str] = []
        for prompt_name in args.prompts:
            entry = PROMPT_REGISTRY.get(prompt_name)
            if not entry:
                unknown_prompts.append(prompt_name)
                continue
            dataset_key = entry["dataset_key"]
            selection.setdefault(dataset_key, []).append(prompt_name)
        if unknown_prompts:
            parser.error(f"Unknown prompt(s): {', '.join(sorted(set(unknown_prompts)))}")
    else:
        selection = {k: list(v) for k, v in DEFAULT_DATASET_PROMPT_SELECTION.items()}

    if args.run_seeds:
        run_seeds: Optional[Sequence[Optional[int]]] = tuple(args.run_seeds)
    elif args.run_count is not None:
        run_seeds = None
    else:
        run_seeds = DEFAULT_RUN_SEEDS

    run_count = args.run_count if args.run_count is not None else DEFAULT_RUN_COUNT

    runner = PipelineRunner(
        settings=settings,
        models=models,
        dataset_prompt_selection=selection,
    )

    try:
        runner.run(run_seeds=run_seeds, run_count=run_count)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
