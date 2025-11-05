# Investigating Bias Mitigation in Large Language Models for Financial Decision-Making

This repository accompanies the MSc dissertation:

> **“Investigating Bias Mitigation in Large Language Models for Classification Tasks in Financial Decision-Making”**  
> Jawad Salah, MSc Venture Capital & Private Equity with FinTech, University College London (2025)  
> Supervised by Prof. Ramin Okhrati (UCL) & Andrea Bassani (NatWest Group PLC)

The project analyses whether state-of-the-art LLMs exhibit social bias when making mortgage approval decisions, and benchmarkes prompt-engineering strategies for mitigation. GPT‑5, GPT‑5 Nano, and Gemini 2.5 Flash Lite were evaluated using fairness metrics including **Demographic Approval Parity (DAP)** and the novel **Absolute Approval Bias (AAB)** introduced in the dissertation.

---

## Repository Layout

```
.
├── data/
│   └── raw/
│       └── hmda_state_ga.csv      # HMDA slice used across experiments
├── notebooks/
│   └── pipeline/                  # Jupyter notebooks for rapid inspection
├── outputs/                       # Populated when the pipeline is executed
│   ├── combined/
│   └── models/<model>/            # Per-model runs + combined summaries
├── src/
│   └── reusable_pipeline/
│       ├── cli.py                 # Command-line entry point
│       ├── config.py              # Paths, settings, and model config helpers
│       ├── data_preprocessor.py   # HMDA cleaning and feature engineering
│       ├── datasets.py            # Proxy dataset construction logic
│       ├── pipeline.py            # High-level orchestration
│       ├── prompts/               # Prompt templates used in the study
│       └── runners/               # Gemini/OpenAI execution adapters
├── requirements.txt
└── Old/                           # Legacy notebooks & datasets (read-only archive)
```

The legacy `Old/` directory is retained for historical completeness but is not required for the reusable pipeline.

### Analysis notebooks

- `notebooks/pipeline/Testing.ipynb` – streamlined location postcode proxy analysis. It loads the pipeline’s master CSV, computes approval-rate gaps between postcode variants, and offers quick inspection helpers.

---

## Environment Setup

1. **Create & activate a virtual environment (Python 3.10+)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure API credentials**
   Export the API keys used by the pipeline. You can add these to your shell profile or a `.env` file that you load manually.
   ```bash
   export GEMINI_API_KEY="your-google-generativeai-key"
   export OPENAI_API_KEY="your-openai-key"
   ```

4. **Optional settings**
   - `PIPELINE_DATA_SOURCE`: override the default HMDA CSV path.
   - `PIPELINE_OUTPUT_DIR`: change the directory used for results.
   - `PIPELINE_MAX_ROWS`, `PIPELINE_TRIES_PER_ROW`, `PIPELINE_RETRY_DELAY`: tweak runtime behaviour.

---

## Running the Reusable Pipeline

The CLI provides a reproducible interface around the orchestrator. From the repository root run:

```bash
python -m reusable_pipeline.cli --run-count 1
```

Key options:

- `--list-prompts` – view available prompt scenario keys.
- `--prompt <name>` – include a specific scenario (can be repeated).
- `--model <name>` – restrict execution to selected models (default runs all configured models).
- `--max-rows <n>` – limit rows per scenario for quick smoke-tests.
- `--run-seed <seed>` – specify dataset sampling seeds (can be repeated). Overrides `--run-count`.
- `--data-source`, `--output-dir` – override defaults without touching environment variables.

All results are written under `outputs/` and grouped by model. Combined summaries are refreshed after each full run.

---

## Results & Reporting

For each execution you will see:

- `outputs/models/<model>/Run XX/` – raw CSV outputs per run.
- `outputs/models/<model>/Combined Results/` – run-agnostic tables and metrics for that model.
- `outputs/combined/all_models_runs.csv` – master sheet containing every model/run/variant produced by the pipeline.
- `outputs/combined/all_models_table.csv` – trimmed, analyst-friendly view used in the notebooks.
- `outputs/combined/*metrics*.csv` – run-level and summary fairness metrics, both overall and per-model.

The notebooks in `notebooks/pipeline/` provide lightweight sanity checks and exploratory plots over the generated outputs.

---

## Citation

If you use this work, please cite:

```
Salah, J. (2025). Investigating Bias Mitigation in Large Language Models for Classification Tasks
in Financial Decision-Making. MSc Dissertation, University College London.
```

---

## Contact

For questions about the dissertation or collaboration opportunities, please reach out via the contact details published alongside the thesis.
