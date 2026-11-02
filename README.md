# Quantifying the Impact of Self-Report Bias on Machine Learning Stroke Classification from NHANES (2003-2023): A Monte Carlo Probabilistic Bias Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NHANES](https://img.shields.io/badge/Data-NHANES%202003--2023-green.svg)](https://www.cdc.gov/nchs/nhanes/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![R 4.4](https://img.shields.io/badge/R-4.4-blue.svg)](https://www.r-project.org/)

> **Status.** Second revision in response to peer review. This repository accompanies the revised paper and is provided for peer review and reproducibility. The code and outputs reflect the revised analysis. Files retained from the earlier analysis are listed under [Superseded files](#superseded-files).

---

## Overview

This repository contains the complete analytical pipeline for classifying prevalent self-reported stroke (NHANES item MCQ160F) from routinely measured clinical variables among U.S. adults in the National Health and Nutrition Examination Survey (NHANES, 2003-2023). A probabilistic bias analysis then estimates how outcome misclassification affects apparent model performance.

MCQ160F asks whether a doctor or other health professional has ever told the participant that they had a stroke. The outcome is therefore prevalent self-reported stroke history, not incident stroke. Validation studies report sensitivity of self-reported stroke ranging from 36% to 98% and specificity from 96% to 99.6%. Most machine learning studies that use self-reported outcomes evaluate performance against these labels without quantifying this uncertainty. This study addresses that gap.

### Key contributions

- **End-to-end ML pipeline** for classifying self-reported stroke from 20 sociodemographic, behavioral, clinical, and laboratory predictors (selected from 22 candidates) in MEC-examined U.S. adults
- **Fold-wise HyperImpute** in which imputation models are fit on training rows only and applied to held-out rows, which prevents leakage between training and evaluation data
- **AutoGluon ensemble** with repeated stratified cross-validation, holdout evaluation, fold-level Youden's J thresholds, and a logistic regression reference model
- **Threshold-free comparison of learner families** (AUROC and AUPRC) alongside threshold-dependent metrics
- **SHAP interpretation** of model attribution (association with predicted probability, not causal effects)
- **Probabilistic bias analysis** (1,000 iterations) that reconstructs the latent true stroke outcome from the observed self-report under nondifferential misclassification, using external sensitivity and specificity ranges, and reports performance against the latent outcome with implied true prevalence
- **Sensitivity analyses of the latent-outcome reconstruction**, one estimating latent risk by maximum likelihood under the misclassification model without truncation (Magder and Hughes, 1997), and one cross-fitting q(s) within the holdout so that no participant's own label informs their reconstruction and the interval includes the estimation uncertainty in q(s)
- **Pre-specified Tier C exclusion sensitivity analysis** (30-50% missingness), which is conditional on a predictor meeting the threshold. No predictor did in the revised cohort, so it was not run

---

## Repository structure

```
├── README.md
├── data/
│   ├── nhanes_stroke_analytic_2003_2023.csv       # Pre-imputation analytic dataset
│   ├── nhanes_stroke_analytic_2003_2023.parquet   # Pre-imputation (Parquet format)
│   └── nhanes_survey_design.rds                   # Survey design object
├── scripts/
│   ├── nhanes_data_extraction.R                   # Extraction, cohort restriction, cross-cycle harmonization
│   ├── table1_weighted.R                          # Weighted Table 1 (survey design)
│   ├── missingness_heatmap.R                      # Missingness visualization
│   ├── fold_imputation.py                         # Train-only HyperImpute wrapper and smoke test
│   ├── hyperimpute_pipeline.py                    # Final imputation on training rows and diagnostics
│   ├── autogluon_modeling.py                      # AutoGluon, SHAP, calibration, probabilistic bias analysis
│   ├── bias_sensitivity_mh.py                     # Maximum likelihood (Magder-Hughes) reconstruction; regenerates the iteration file
│   ├── bias_sensitivity_crossfit.py               # Cross-fitted q(s) within the holdout
│   ├── bias_sensitivity_oof.py                    # Diagnostic only (OOF training predictions; not used)
│   └── fig_pipeline_workflow.py                   # Workflow figure (manuscript Figure 6)
├── artifacts/                                     # Tables, diagnostics, and figures (see below)
└── models/
    └── final/                                     # Trained AutoGluon model (holdout evaluation)
```

Imputation is performed within each cross-validation fold on training rows only, so no full-sample imputed dataset is released. The pre-imputation analytic dataset is provided in `data/`. Fold-wise imputations (`data/imputed_folds/`) and run logs (`logs/`) are regenerated by the scripts and are not tracked.

### Key artifacts

| File | Description |
|------|-------------|
| `variable_codebook.csv` | Variable definitions, types, tiers, and cycle-specific coding |
| `ridstatr_by_cycle.csv` | Interview and MEC examination status by cycle |
| `cohort_flow_by_cycle.csv` | Cohort flow by cycle (outcome response, MEC examination, pregnancy, 2021-2023 phlebotomy weight) |
| `outcome_by_cycle.csv` | Stroke prevalence by cycle |
| `harmonization_checks_by_cycle.csv` | Cross-cycle checks for harmonized predictors |
| `bp_method_comparison_2017.csv` | Paired blood pressure method comparison for 2017-2018 |
| `missingness_report.csv`, `missingness_by_cycle.csv`, `missingness_mechanism_tests.csv` | Missingness rates, tier assignments, and MAR/MCAR diagnostics |
| `imputation_diagnostics.csv`, `correlation_diff_pre_post_imputation.csv` | Imputation quality metrics |
| `table1_weighted_final.csv` | Weighted Table 1 |
| `missingness_tier_classification.csv`, `exclusion_characterization.csv`, `component_participation_by_cycle.csv` | Tier assignment, characteristics of excluded participants, and survey component participation by cycle |
| `cv_results.csv` | Per-fold CV performance, including fold-level thresholds |
| `autogluon_leaderboard.csv`, `autogluon_config.json` | AutoGluon model rankings and configuration |
| `model_comparison_holdout.csv`, `model_comparison_by_family.csv` | Holdout AUROC, AUPRC, and threshold metrics per model and by learner family, with logistic regression reference |
| `table2_performance.csv`, `holdout_metrics_ci.csv` | Performance summary table for the manuscript and holdout bootstrap confidence intervals |
| `holdout_auroc_by_measurement_period.csv` | Holdout AUROC by instrument period |
| `calibration_deciles_holdout.csv` | Observed and predicted risk by decile with 95% Wilson intervals |
| `holdout_predictions.csv` | Holdout predicted probabilities and observed labels (input to all bias analyses) |
| `shap_importance.csv` | SHAP feature importance rankings |
| `monte_carlo_misclassification.csv` | Primary bias analysis, one row per iteration. Sampled self-report parameters are stored as `se_selfreport` and `sp_selfreport`, separate from the model's `recall` and `specificity` (regenerated by `bias_sensitivity_mh.py`) |
| `bias_analysis_summary.csv` | Primary bias analysis, naive value, median, and 95% simulation interval per metric |
| `pba_mh_iterations.csv`, `pba_mh_summary.csv`, `pba_mh_manuscript_numbers.json` | Maximum likelihood reconstruction paired with the primary analysis, including the share of participants truncated at zero (Supplementary Table S9) |
| `pba_crossfit_iterations.csv`, `pba_crossfit_summary.csv`, `pba_crossfit_manuscript_numbers.json` | Cross-fitted q(s), sensitivity version of the latent-outcome column of Table 2 (Supplementary Table S10) |
| `oof_predictions_train.csv`, `pba_oof_*` | Diagnostic only. OOF training predictions were not on the holdout score scale, so these results are not used |
| `fig_*.png` | Missingness heatmap, imputation correlation, ROC and precision-recall, calibration with 95% Wilson intervals, SHAP (bar, beeswarm, dependence), bias analysis (eight panels), and workflow (`fig_pipeline_workflow`, Figure 6) |

Every figure is also saved as `.pdf` and `.svg`. `ensemble_weights.csv` is written only when the ensemble weights can be extracted from the fitted predictor and is not part of the current release.

### Artifact provenance

Artifacts from `autogluon_modeling.py` were generated by the pipeline at commit `11f7e2f`. Later edits to that script rename the stored self-report parameters in the bias analysis and update comments and printed output, and do not change any result. The corrected iteration file and all sensitivity analyses are produced by the standalone scripts below, which read `holdout_predictions.csv` and need no retraining.

### Superseded files

| File | Status |
|------|--------|
| `artifacts/superseded/sensitivity_no_tierc_results.csv` | From the original submission (commit `dd34897`, earlier cohort). Superseded. No predictor met the Tier C threshold in the revised cohort, so this analysis was not repeated |

---

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `nhanes_data_extraction.R` | Extracts and harmonizes 22 candidate predictors across NHANES cycles (2003-2023). Restricts the cohort to MEC-examined, non-pregnant adults aged 20-80 (with 2021-2023 participants required to have a phlebotomy weight). Harmonizes age top-coding at 80, physical activity, alcohol skip patterns, and blood pressure across cycles |
| 2 | `table1_weighted.R` | Generates survey-weighted Table 1 stratified by stroke status |
| 3 | `missingness_heatmap.R` | Visualizes variable by cycle missingness with the tiered decision framework |
| 4 | `fold_imputation.py` | Smoke test of the train-only HyperImpute wrapper (optional, a few minutes) |
| 5 | `hyperimpute_pipeline.py` | Fits column-wise adaptive imputation on training rows and produces distributional diagnostics |
| 6 | `autogluon_modeling.py` | 80/20 holdout split, repeated stratified 5-fold by 5 CV with fold-wise imputation, final retrain, logistic regression reference, family comparison, SHAP, calibration, and probabilistic bias analysis. The pre-specified Tier C sensitivity analysis runs only if a predictor meets the Tier C threshold |
| 7 | `bias_sensitivity_mh.py` | Maximum likelihood reconstruction without truncation, paired with the primary analysis, and regeneration of `monte_carlo_misclassification.csv` (seconds) |
| 8 | `bias_sensitivity_crossfit.py` | Cross-fitted q(s) within the holdout with bootstrap resampling (under a minute) |
| 9 | `fig_pipeline_workflow.py` | Workflow figure (Figure 6) |

---

## Reproducing the analysis

### Prerequisites

**R (4.4 or later)**
```r
install.packages(c("nhanesdata", "survey", "tableone", "dplyr", "tidyr", "purrr",
                    "ggplot2", "viridis", "ragg", "svglite", "e1071"))
# nhanesdata from GitHub
# remotes::install_github("kyleGrealis/nhanesdata")
```

**Python (3.12 or later)**

The environment is defined in `pyproject.toml` and `uv.lock`.
```bash
uv sync
```
Alternatively, install with pip.
```bash
pip install autogluon hyperimpute shap pandas "numpy<2" "scipy<1.13" \
    "pyarrow<17" matplotlib seaborn scikit-learn==1.5.2
```

> **Note.** scikit-learn is pinned to 1.5.2 because HyperImpute uses the deprecated `multi_class` parameter in `LogisticRegression`. CatBoost replaces XGBoost in the HyperImpute learner library to avoid OpenMP conflicts on macOS Apple Silicon.

### Running the pipeline

Run the scripts sequentially from the project root. Delete `data/imputed_folds/` before a full rerun so that stale imputations are not reused.

```bash
# 1. Data extraction and cohort restriction (R)
Rscript scripts/nhanes_data_extraction.R

# 2. Weighted Table 1 (R)
Rscript scripts/table1_weighted.R

# 3. Missingness heatmap (R)
Rscript scripts/missingness_heatmap.R

# 4. Imputation wrapper smoke test (Python, optional)
python scripts/fold_imputation.py

# 5. Imputation (Python)
python scripts/hyperimpute_pipeline.py

# 6. Modeling, SHAP, and bias analysis (Python, long runtime)
python scripts/autogluon_modeling.py 2>&1 | tee artifacts/run_log.txt

# 7-8. Sensitivity analyses of the bias analysis (Python, no retraining)
python scripts/bias_sensitivity_mh.py
python scripts/bias_sensitivity_crossfit.py

# 9. Workflow figure
python scripts/fig_pipeline_workflow.py
```

Steps 7 and 8 read only `artifacts/holdout_predictions.csv` and write logs to `logs/`. `bias_sensitivity_mh.py` checks that every performance column matches the existing `monte_carlo_misclassification.csv` before rewriting it. The diagnostic `bias_sensitivity_oof.py` needs the saved predictor in `models/final/` (or the cached `oof_predictions_train.csv`) and is not part of the reported analysis.

Expected checks after step 1 include a non-empty `cohort_flow_by_cycle.csv`, a maximum age of 80 in every cycle, and nonzero physical activity prevalence in 2003-2005.

### Configuration

Key parameters in `autogluon_modeling.py` are defined at the top of the script.

```python
N_SPLITS = 5              # CV folds
N_REPEATS = 5             # CV repeats
TIME_LIMIT = 600          # Seconds per CV fold
TIME_LIMIT_FINAL = 3000   # Seconds for final model retrain
PRESETS = "best_quality"  # AutoGluon preset
N_SHAP = 2000             # SHAP samples
N_MC = 1000               # Bias analysis iterations
SENS_RANGE = (0.36, 0.98) # Self-report sensitivity range
SPEC_RANGE = (0.96, 0.996)# Self-report specificity range
FORCE_SENSITIVITY_VARS = []  # Variables forced into the Tier C sensitivity analysis (none)
```

For a quick test run, change to the following.
```python
N_SPLITS = 2; N_REPEATS = 1; TIME_LIMIT = 60; N_SHAP = 20; N_MC = 100
PRESETS = "medium_quality"
```

---

## Data

### Source

All data are from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/), a program of the National Center for Health Statistics (NCHS). NHANES data are publicly available and do not require IRB approval for secondary analysis.

Data were accessed through the [`nhanesdata`](https://github.com/kyleGrealis/nhanesdata) R package, which provides pre-harmonized parquet files across survey cycles.

### Cohort

- **Cycles.** 2003-2004 through 2021-2023, excluding 2019-2020 because of COVID-19
- **Population.** Non-pregnant adults aged 20-80 with a valid stroke response who completed the MEC examination (RIDSTATR = 2, WTMEC2YR > 0). For 2021-2023, a positive phlebotomy weight (WTPH2YR) is also required for blood analytes. Age is top-coded at 80 in all cycles
- **Sample size.** 47,696 participants (1,962 stroke cases, 4.1% prevalence)
- **Holdout.** A random 20% holdout set evaluated once after model selection. Validation is internal only, using a random subset of the same pooled survey

Exclusions by step and cycle are in `artifacts/cohort_flow_by_cycle.csv`.

### Predictors (20 variables)

| Domain | Variables |
|--------|-----------|
| Demographics | Age, sex, race/ethnicity, education, poverty-income ratio |
| Cardiovascular | Mean systolic BP, mean diastolic BP, antihypertensive medication |
| Metabolic | HbA1c, total cholesterol, HDL cholesterol, BMI, waist circumference |
| Behavioral | Smoking status, alcohol (drinks per drinking day), physical activity |
| Laboratory | Serum creatinine, white blood cell count |
| Comorbidities | Self-reported diabetes, self-reported coronary heart disease |

Several predictors are self-reported (diabetes, coronary heart disease, smoking, alcohol, physical activity, and medication use). The bias analysis addresses outcome misclassification only and does not quantify predictor measurement error.

---

## Interpretation and limitations

- The outcome is prevalent self-reported stroke from repeated cross-sectional surveys. Results do not describe incident stroke risk.
- SHAP values describe model attribution and association with predicted probability. They do not indicate causal effects, and some predictors may be consequences or correlates of prior stroke.
- Evaluation is internal. No later-cycle, separate-survey, or clinically confirmed outcome validation was performed.
- The Youden's J threshold is a statistical operating point and not a clinical action threshold.
- The latent-outcome results assume nondifferential misclassification and published sensitivity and specificity ranges. The size of the corrected gain in AUROC depends on how latent risk is reconstructed (median 0.874 with the primary isotonic reconstruction and 0.847 without truncation).

---

## Citation

If you use this code or methodology, please cite the following.

```
Togunwa, T.O., Kolls, B.J., Ukachukwu, A.-E.K. (2026). Quantifying the Impact of
Self-Report Bias on Machine Learning Stroke Classification from NHANES (2003-2023):
A Monte Carlo Probabilistic Bias Analysis.
[ref to be updated once paper is published].
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Taofeeq O. Togunwa**
togunwa@umich.edu
University of Michigan Medical School
