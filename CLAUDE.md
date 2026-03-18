# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-stage R&D credit scoring system (AUS: Automated Underwriting System) for SME default prediction. The project has three historical phases plus an ongoing Devil's Advocate optimization layer.

- **Target**: Binary default prediction (1,514 companies, 57 positives = 3.76% imbalance)
- **Final validated performance**: ROC-AUC 0.7890 ± 0.0292, KS 0.5856
- **Evaluation standard**: Stratified 5-Fold CV × 3 Repeats (15 measurements) — single-split results are unreliable given the small positive class

## Running Experiments

```bash
# Optuna hyperparameter optimization (RF / XGB / GBM, 150 total trials) + Stacking
python 04_devils_advocate/run_phase3.py

# 13-model comparison (RF, XGB, LGB, CatBoost, BalancedRF, EasyEnsemble, LogReg, SVM, Stacking)
python 04_devils_advocate/run_model_comparison.py
```

Data paths are hardcoded inside these scripts and point to:
- `03_flowscore_ml_ver2/outputs/AUS_v2_Golden_Training_Set.csv` — 1,514 rows, original labels
- `03_flowscore_ml_ver2/outputs/AUS_v2_Ready_To_Train.csv` — 1,514 rows, preprocessed

All Jupyter notebooks are run interactively; there is no automated test runner.

## Repository Structure

```
01_manual_classification/   — Phase 1: manual expert scoring + Streamlit dashboard (app.py)
02_flowscore_ml_ver1/       — Phase 2: financial-ratio ML v1, scorecard baseline
03_flowscore_ml_ver2/       — Phase 3: behavioral-feature ML v2, outputs/
04_devils_advocate/         — Phase 4: systematic challenge experiments (active development)
raw_data/                   — Source data (gitignored CSVs)
```

The `.gitignore` excludes `*.csv`, `*.xlsx`, and `output/` directories, so data files are never committed.

## Validated Pipeline (Do Not Change Without Devil's Advocate Re-validation)

The following pipeline is the result of Phase 0–3 experiments and should be treated as the production baseline:

**Preprocessing order:**
1. `pd.to_numeric(errors='coerce')` — handles Korean text like "자본잠식" (capital erosion flag in DEBT_RATIO; 116 rows, 2.98× default rate)
2. `fillna(df.median())` — median imputation
3. Winsorize 1%–99% per feature
4. Log transform (after shifting negatives)

**Feature set (13 features):**
- Core (3): `CASH_RATIO`, `EMPLOYEE_COUNT`, `NET_PROFIT_MARGIN`
- Neutral (6): `receivable_Total_Amt`, `BNPL_Success_Rate`, `INTEREST_COVERAGE_RATIO`, `REP_CHANGE_COUNT`, `GROSS_PROFIT_MARGIN`, `SALES_REVENUE`
- Engineered (4):
  - `FE_GROWTH_QUALITY` = `SALES_GROWTH_RATE × OPERATING_MARGIN`
  - `FE_DEBT_SERVICE` = `CASH_RATIO × (INTEREST_COVERAGE_RATIO + 1)`
  - `FE_PROFIT_EFFICIENCY` = `(SALES_REVENUE × OPERATING_MARGIN/100) / (EMPLOYEE_COUNT + 1)`
  - `FE_ZERO_COUNT` = count of zero-valued columns per row

> ⚠️ 12 out of the original 21 features were found HARMFUL by LOFO analysis. Adding features back without re-running LOFO will degrade performance.

**Model:**
```python
RandomForestClassifier(
    n_estimators=700, max_depth=None,
    min_samples_split=10, max_features="log2",
    class_weight="balanced_subsample", random_state=42
)
```

## Key Architectural Decisions

**Why not AutoML / per-dataset model selection?**
With only 57 positive samples, model selection noise exceeds any real accuracy gain. Optuna (150 trials) and Stacking both failed to beat the base RF. Changing models per evaluation batch breaks score comparability, which is a hard requirement for credit scoring.

**When to re-run Devil's Advocate:**
Re-validate the pipeline when ~20+ new positive cases have accumulated (approximately semi-annually). Do not tune the model for individual new companies.

**BNPL features:**
100% of Y=1 (default) companies had zero BNPL transaction data. These features are noise for the positive class and should be kept only if zero-count itself is informative (captured by `FE_ZERO_COUNT`).

## Phase 4 Experiment Results Summary

| Phase | AUC | Std | Change |
|-------|-----|-----|--------|
| 0: Baseline (5-Fold CV) | 0.7554 | 0.0541 | — |
| 1: Preprocessing | 0.7593 | 0.0547 | +0.0039 |
| 2: Feature reduction | 0.7890 | 0.0292 | +0.0297 |
| 3: Optuna + Stacking | 0.7890 | 0.0292 | 0.0000 |

Full methodology and gate criteria: `04_devils_advocate/STRATEGY.md`
Full numerical results: `04_devils_advocate/RESULTS.md`
