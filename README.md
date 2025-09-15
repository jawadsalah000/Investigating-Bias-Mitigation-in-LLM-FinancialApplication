# Investigating Bias Mitigation in Large Language Models for Financial Decision-Making

This repository contains all code, datasets, and results accompanying the MSc dissertation:

> **“Investigating Bias Mitigation in Large Language Models for Classification Tasks in Financial Decision-Making”**
> Jawad Salah, MSc Venture Capital & Private Equity with FinTech, University College London (2025)
> Supervised by Prof. Ramin Okhrati (UCL) & Andrea Bassani (NatWest Group PLC).

---

## 📑 Overview

The dissertation evaluates whether latest-generation Large Language Models (LLMs) exhibit **social biases** in financial decision making tasks, before investigation various methods to mitigate that bias. Using **mortgage underwriting** as a testbed, experiments were run on GPT-5, GPT-5 Nano, and Gemini 2.5 Flash Lite with multiple **prompt-engineering strategies**.

Two fairness metrics were used:

* **Demographic Approval Parity (DAP)** – group-level disparity.
* **Absolute Approval Bias (AAB)** – model sensitivity to explicit race labels (introduced in this study).

The project demonstrates that **Chain-of-Thought (CoT) prompting** was the most effective mitigation method, though effectiveness was model-dependent and sometimes came at a computational cost.

---

## 📂 Repository Structure

```
Repository Root
│
├── Dataset
│   ├── Input
│   │   ├── Boot-Strapped Synthetic Dataset.xlsx
│   │   ├── Counterfactual Synthetic Dataset 1.xlsx
│   │   ├── Counterfactual Synthetic Dataset 2.xlsx
│   │   └── Pre-processed Dataset.xls
│   │
│   └── Results
│       ├── Bias Identification and Mitigation Tests
│       │   ├── GPT-5 Nano
│       │   │   └── Reasoning vs Bias Experiment
│       │   ├── GPT-5 / Minimal Reasoning
│       │   └── Gemini 2.5 Flash Lite
│       │
│       ├── Categorical vs. Continuous Experiment
│       ├── Proxies Experiment
│       └── Temperature Study
│
├── Notebooks
│   ├── 1) Data Pre-Processing and Exploration.ipynb
│   ├── 2) Counterfactual Input Testing.ipynb
│   ├── 3) Bias Identification Experiment.ipynb
│   ├── 4) Zero Shot Prompt Engineering for Bias Mitigation.ipynb
│   ├── 5.1) GPT 5 Results and Analysis.ipynb
│   ├── 5.2) GPT 5 Nano Results and Analysis.ipynb
│   ├── 5.3) Gemini Results and Analysis.ipynb
│   └── 6) Reasoning Level vs Bias Test (Nano).ipynb
│
├── scripts
│   └── prompts.py   # All baseline and engineered prompts
│
└── README.md
```

---

## ⚙️ How to Use

1. Clone the repo:

   ```bash
   git clone https://github.com/jawadsalah000/Investigating-Bias-Mitigation-in-LLM-FinancialApplication.git
   cd Investigating-Bias-Mitigation-in-LLM-FinancialApplication
   ```
2. Install dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```
3. Explore the Jupyter Notebooks in the `Notebooks/` folder for step-by-step replication of experiments.
4. Use `scripts/prompts.py` to access the exact prompts used (baseline, reasoning, CoT, fairness phrasing).

---

## 📊 Key Contributions

* Introduced **Absolute Approval Bias (AAB)** metric to detect race-sensitivity bias.
* Ran **systematic audits on the GPT-5 series and Gemini 2.5**.
* Found **CoT prompting most effective** in reducing disparities across models using group-level bias metrics.
* Released full datasets, notebooks, and result files for **reproducibility and transparency**.

---

## 📖 Citation

If you use this work, please cite:

```
Salah, J. (2025). Investigating Bias Mitigation in Large Language Models for Classification Tasks 
in Financial Decision-Making. MSc Dissertation, University College London.
```

---

Do you want me to also add a **“Results at a Glance” section** in the README with your key tables/figures (like GPT-5 vs Nano vs Gemini baseline bias outcomes) so people see findings instantly?
