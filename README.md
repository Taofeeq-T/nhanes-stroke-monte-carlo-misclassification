# Quantifying the Impact of Self-Report Bias on Machine Learning Stroke Prediction from NHANES: A Monte Carlo Sensitivity Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NHANES](https://img.shields.io/badge/Data-NHANES%202003--2023-green.svg)](https://www.cdc.gov/nchs/nhanes/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![R 4.4](https://img.shields.io/badge/R-4.4-blue.svg)](https://www.r-project.org/)

> **Status:** Manuscript under review. This repository accompanies the submitted paper and is provided for peer review and reproducibility.

---

## Overview

This repository contains the complete analytical pipeline for predicting self-reported stroke from routinely measured clinical variables in the National Health and Nutrition Examination Survey (NHANES, 2003–2023), with a novel Monte Carlo sensitivity analysis that quantifies how outcome misclassification affects apparent model performance.

Self-reported stroke (NHANES item MCQ160F) has variable diagnostic accuracy — validation studies report sensitivity ranging from 36% to 98% and specificity from 96% to 99.6%. Most machine learning studies using self-reported outcomes evaluate performance against these potentially erroneous labels without acknowledging or quantifying this uncertainty. This study addresses that gap.

### Key contributions

- **End-to-end ML pipeline** for stroke prediction from 20 sociodemographic, behavioral, clinical, and laboratory predictors across 52,507 U.S. adults
- **HyperImpute** for adaptive column-wise imputation with full distributional diagnostics
- **AutoGluon ensemble** with repeated stratified cross-validation, holdout evaluation, and Youden's J threshold optimization
- **SHAP interpretation** with beeswarm, bar, and dependence plots
- **Monte Carlo misclassification sensitivity analysis** (1,000 iterations) propagating plausible outcome error through all evaluation metrics
- **Tier C sensitivity analysis** assessing robustness to high-missingness variables

---

## Repository structure

```
├── README.md
├── data/
│   ├── nhanes_stroke_analytic_2003_2023.csv       # Pre-imputation analytic dataset
│   ├── nhanes_stroke_imputed_2003_2023.csv        # Post-imputation dataset
│   └── nhanes_stroke_imputed_2003_2023.parquet     # Post-imputation (Parquet format)
├── scripts/
│   ├── nhanes_data_extraction.R                   # Data extraction & cross-cycle harmonization
│   ├── missingness_heatmap.R                      # Missingness visualization (Figure)
│   ├── hyperimpute_pipeline.py                    # HyperImpute imputation + diagnostics
│   ├── table1_weighted.R                          # Weighted Table 1 (survey design)
│   └── autogluon_modeling.py                      # AutoGluon + SHAP + Monte Carlo + sensitivity
├── artifacts/
│   ├── variable_codebook.csv                      # Variable definitions, types, tiers
│   ├── missingness_report.csv                     # Missingness rates and tier assignments
│   ├── missingness_mechanism_tests.csv            # Chi-squared MAR/MCAR diagnostics
│   ├── imputation_diagnostics.csv                 # Pre/post imputation quality metrics
│   ├── correlation_diff_pre_post_imputation.csv   # Correlation structure changes
│   ├── table1_weighted_final.csv                  # Weighted Table 1
│   ├── cv_results.csv                             # Per-fold CV performance metrics
│   ├── holdout_predictions.csv                    # Holdout predicted probabilities
│   ├── autogluon_leaderboard.csv                  # AutoGluon model rankings
│   ├── shap_importance.csv                        # SHAP feature importance rankings
│   ├── monte_carlo_misclassification.csv          # MC simulation results (1,000 iterations)
│   ├── sensitivity_no_tierc_results.csv           # Tier C sensitivity CV results
│   ├── fig_missingness_heatmap.png                # Missingness heatmap
│   ├── fig_correlation_pre_post_imputation.png    # Imputation quality: correlation heatmap
│   ├── fig_roc_prc_holdout.png                    # ROC + Precision-Recall curves
│   ├── fig_calibration_holdout.png                # Calibration curve with histogram
│   ├── fig_dca_holdout.png                        # Decision curve analysis
│   ├── fig_shap_bar.png                           # SHAP values for top variables
│   ├── fig_shap_beeswarm.png                     # SHAP beeswarm plot
│   ├── fig_shap_dependence.png                    # SHAP dependence plots (top 6)
│   └── fig_monte_carlo_misclassification.png      # MC sensitivity distributions
└── models/
    └── final/                                     # Trained AutoGluon model (holdout evaluation)
```

---

## Pipeline overview

![Analytical Pipeline](artifacts/fig_pipeline_workflow.svg)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `nhanes_data_extraction.R` | Extracts and harmonizes 22 predictors across 9 NHANES cycles (2003–2023), handles cross-cycle variable renaming, 2003 legacy lab files, BP zero removal, alcohol skip-pattern coding, and multi-cycle weight adjustment |
| 2 | `table1_weighted.R` | Generates survey-weighted Table 1 stratified by stroke status |
| 3 | `missingness_heatmap.R` | Visualizes variable × cycle missingness with tiered decision framework |
| 4 | `hyperimpute_pipeline.py` | Imputes 20 variables using column-wise adaptive model selection (logistic regression, random forest, CatBoost); produces distributional diagnostics and correlation preservation checks |
| 5 | `autogluon_modeling.py` | Full modeling pipeline: 80/20 holdout split → repeated stratified 5-fold × 5 CV → final retrain → SHAP → calibration + ROC + PRC + DCA → Monte Carlo misclassification → Tier C sensitivity |

---

## Reproducing the analysis

### Prerequisites

**R (≥ 4.4):**
```r
install.packages(c("nhanesdata", "survey", "tableone", "dplyr", "tidyr",
                    "ggplot2", "viridis", "ragg", "svglite", "e1071"))
# nhanesdata from GitHub:
# remotes::install_github("kyleGrealis/nhanesdata")
```

**Python (≥ 3.12):**
```bash
pip install autogluon.tabular hyperimpute knockpy shap \
    pandas numpy scipy matplotlib seaborn scikit-learn==1.5.2
```

> **Note:** scikit-learn is pinned to 1.5.2 due to a compatibility issue with HyperImpute's use of the deprecated `multi_class` parameter in `LogisticRegression`. CatBoost is used in place of XGBoost in the HyperImpute learner library to avoid OpenMP library conflicts on macOS Apple Silicon.

### Running the pipeline

Scripts are designed to be run sequentially from the project root:

```bash
# 1. Data extraction (R)
Rscript scripts/nhanes_data_extraction.R

# 2. Weighted Table 1 (R)
Rscript scripts/table1_weighted.R

# 3. Missingness heatmap (R)
Rscript scripts/missingness_heatmap.R

# 4. Imputation (Python)
python scripts/hyperimpute_pipeline.py

# 5. Modeling + SHAP + Monte Carlo + sensitivity (Python) — ~10 hours
python scripts/autogluon_modeling.py
```

### Configuration

Key parameters in `autogluon_modeling.py` are defined at the top of the script for easy modification:

```python
N_SPLITS = 5              # CV folds
N_REPEATS = 5             # CV repeats
TIME_LIMIT = 600          # Seconds per CV fold
TIME_LIMIT_FINAL = 3000   # Seconds for final model retrain
PRESETS = "best_quality"  # AutoGluon preset
N_SHAP = 2000             # SHAP samples
N_MC = 1000               # Monte Carlo iterations
SENS_RANGE = (0.36, 0.98) # Self-report sensitivity range
SPEC_RANGE = (0.96, 0.996)# Self-report specificity range
```

For a quick test run (~5 minutes), change to:
```python
N_SPLITS = 2; N_REPEATS = 1; TIME_LIMIT = 60; N_SHAP = 20; N_MC = 100
PRESETS = "medium_quality"
```

---

## Data

### Source

All data are from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/), a program of the National Center for Health Statistics (NCHS). NHANES data are publicly available and do not require IRB approval for secondary analysis.

Data were accessed via the [`nhanesdata`](https://github.com/kyleGrealis/nhanesdata) R package, which provides pre-harmonized parquet files across survey cycles.

### Cohort

- **Cycles:** 2003–2004 through 2021–2023 (excluding 2019–2020 due to COVID-19)
- **Population:** Non-pregnant adults aged 20–80 with valid stroke response and MEC weights
- **Sample size:** 52,507 participants (2,170 stroke cases, 4.1% prevalence)

### Predictors (20 variables)

| Domain | Variables |
|--------|-----------|
| Demographics | Age, sex, race/ethnicity, education, poverty-income ratio |
| Cardiovascular | Mean systolic BP, mean diastolic BP, antihypertensive medication |
| Metabolic | HbA1c, total cholesterol, HDL cholesterol, BMI, waist circumference |
| Behavioral | Smoking status, alcohol drinks/day, physical activity |
| Laboratory | Serum creatinine, white blood cell count |
| Comorbidities | Self-reported diabetes, self-reported coronary heart disease |

---

## Citation

If you use this code or methodology, please cite:

```
Togunwa, T.O. (2026). Quantifying the Impact of Self-Report Bias on Machine
Learning Stroke Prediction from NHANES: A Monte Carlo Sensitivity Analysis.
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
