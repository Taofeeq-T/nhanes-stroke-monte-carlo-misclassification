"""
================================================================================
NHANES Stroke Prediction — AutoGluon Modeling Pipeline
================================================================================
Ensemble prediction with:
  1. Repeated stratified k-fold cross-validation
  2. SHAP-based feature importance and interpretation
  3. Monte Carlo misclassification sensitivity analysis
  4. Tier C sensitivity analysis (with/without alcohol)

Requirements:
  pip install autogluon.tabular shap matplotlib seaborn

References:
  - Erickson et al. (2020). AutoGluon-Tabular: Robust and Accurate AutoML
    for Structured Data. arXiv:2003.06505
  - Lundberg & Lee (2017). A Unified Approach to Interpreting Model
    Predictions. NeurIPS. (SHAP)
  - Lash et al. (2014). Good practices for quantitative bias analysis.
    IJE 43(6):1969-1985. (Misclassification framework)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, average_precision_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score
)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# --- Log console output to logs/stage6_modeling.log (for review bundle) ---
import sys as _sys
from pathlib import Path as _Path


import re as _re

_WARN = _re.compile(r"FutureWarning|ConvergenceWarning|DeprecationWarning|UserWarning|"
                    r"RuntimeWarning|^\s*warnings\.warn\(|site-packages.*\.py:\d+")


class _Tee:
    """Console output is unchanged; the log file drops library warning lines."""

    def __init__(self, console, logfile):
        self.console, self.logfile, self._buf = console, logfile, ""

    def write(self, data):
        self.console.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not _WARN.search(line):
                self.logfile.write(line + "\n")
        self.console.flush()
        self.logfile.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):          # fileno(), isatty(), encoding, etc.
        return getattr(self.console, name)


_log_dir = _Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = open(_log_dir / "stage6_modeling.log", "w")
_sys.stdout = _Tee(_sys.__stdout__, _log_file)
_sys.stderr = _Tee(_sys.__stderr__, _log_file)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve


def youden_threshold(y, p):
    """Threshold maximizing Youden's J on (y, p)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1))); fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0))); fp = int(np.sum((yhat == 1) & (y == 0)))
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    return {"threshold": thr, "balanced_accuracy": (se + sp) / 2,
            "f1": 2 * ppv * se / max(ppv + se, 1e-12), "sensitivity": se,
            "specificity": sp, "ppv": ppv, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def logistic_reference():
    """Unpenalized-in-practice logistic regression reference (one-hot + scaling)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))


def one_hot(X, columns=None):
    Xd = pd.get_dummies(X, drop_first=True, dtype=float)
    return Xd if columns is None else Xd.reindex(columns=columns, fill_value=0.0)


def autogluon_oof_proba(pred, model=None):
    """Out-of-fold P(stroke) on the predictor's own training data, if available."""
    for fn in ("predict_proba_oof", "get_oof_pred_proba"):
        if hasattr(pred, fn):
            try:
                out = getattr(pred, fn)(model=model) if model else getattr(pred, fn)()
                return np.asarray(out[1] if hasattr(out, "columns") else out)
            except Exception:
                continue
    return None


def calibration_intercept_slope(y, p, eps=1e-6):
    """Calibration-in-the-large (intercept with slope fixed at 1) and calibration slope."""
    from scipy.optimize import brentq
    from scipy.special import expit
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    a = brentq(lambda a: np.sum(y - expit(a + lp)), -20, 20)
    slope = LogisticRegression(C=1e6, max_iter=5000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    return a, slope


def holdout_metrics(y, p, thr):
    ci, cs = calibration_intercept_slope(y, p)
    tm = threshold_metrics(y, p, thr)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p), "calibration_intercept": ci,
            "calibration_slope": cs, "balanced_accuracy": tm["balanced_accuracy"],
            "f1": tm["f1"], "sensitivity": tm["sensitivity"],
            "specificity": tm["specificity"], "ppv": tm["ppv"]}


def bootstrap_ci(y, p, thr, n_boot=1000, seed=42):
    """Point estimates and 95% CIs from a bootstrap stratified by outcome."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    point = holdout_metrics(y, p, thr)
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boots.append(holdout_metrics(y[idx], p[idx], thr))
    b = pd.DataFrame(boots)
    return pd.DataFrame({"metric": list(point), "estimate": list(point.values()),
                         "ci_lower": [b[m].quantile(0.025) for m in point],
                         "ci_upper": [b[m].quantile(0.975) for m in point]})


# Publication-quality figure defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================================================================
# CONFIGURATION — adjust these for quick test vs. full run
# ==============================================================================
# Quick test:  N_SPLITS=2, N_REPEATS=1, TIME_LIMIT=60, N_SHAP=20, PRESETS="medium_quality"
# Full run:    N_SPLITS=5, N_REPEATS=5, TIME_LIMIT=600, N_SHAP=2000, PRESETS="best_quality_v150"

N_SPLITS = 5           # Number of CV folds
N_REPEATS = 5          # Number of CV repeats
TIME_LIMIT = 600       # Seconds per AutoGluon CV fold
TIME_LIMIT_FINAL = 3000  # Seconds for final model retrain (5x CV limit)
PRESETS = "best_quality"  # AutoGluon preset ("medium_quality", "high_quality", "best_quality")
N_SHAP = 2000          # Number of samples for SHAP analysis (20=quick, 2000=full)
HOLDOUT_FRAC = 0.20    # Fraction of data held out for final evaluation
N_MC = 1000            # Monte Carlo misclassification iterations
SENS_RANGE = (0.36, 0.98)  # Self-reported stroke sensitivity (https://doi.org/10.1371/journal.pone.0137538)
SPEC_RANGE = (0.96, 0.996)  # Self-reported stroke specificity (https://doi.org/10.1371/journal.pone.0137538)

# ==============================================================================
# 1. LOAD DATA & SETUP
# ==============================================================================

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"
artifacts_dir = project_root / "artifacts"
models_dir = project_root / "models"
artifacts_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

print("=" * 70)
print("STEP 1: Load imputed data")
print("=" * 70)

# Load the PRE-imputation analytic dataset. Imputation is performed inside the
# modeling loop, fitted on training rows only (see fold_imputation.py).
df = pd.read_csv(data_dir / "nhanes_stroke_analytic_2003_2023.csv")
codebook = pd.read_csv(artifacts_dir / "variable_codebook.csv")
miss_report = pd.read_csv(artifacts_dir / "missingness_report.csv")

# Identify Tier D (excluded) and Tier C (sensitivity analysis) variables
tier_d_vars = miss_report[miss_report["tier"].str.startswith("D")]["variable"].tolist()
tier_c_vars = miss_report[miss_report["tier"].str.startswith("C")]["variable"].tolist()
# Alcohol was pre-specified for the exclusion sensitivity analysis. After the
# cohort restriction its missingness sits near the 30% Tier B/C boundary, so it
# is kept in the sensitivity analysis regardless of tier. Set to [] to make the
# sensitivity analysis purely tier-driven.
FORCE_SENSITIVITY_VARS = []
tier_c_vars = sorted(set(tier_c_vars) | set(FORCE_SENSITIVITY_VARS))

print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Stroke prevalence: {df['stroke'].mean()*100:.1f}% ({df['stroke'].sum():,} cases)")
print(f"Tier D (excluded): {tier_d_vars}")
print(f"Tier C (sensitivity): {tier_c_vars}")

# ==============================================================================
# 2. DEFINE FEATURE SET
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 2: Define feature set")
print("=" * 70)

# All predictors except IDs, weights, outcome, and Tier D
exclude_cols = ["seqn", "year", "sdmvpsu", "sdmvstra", "wtmec2yr",
                "wtint2yr", "wtph2yr", "wt_pooled", "stroke"] + tier_d_vars

feature_cols = [c for c in df.columns if c not in exclude_cols]

# Define Tier C-excluded feature set for sensitivity analysis
feature_cols_no_tierc = [c for c in feature_cols if c not in tier_c_vars]

print(f"Primary feature set: {len(feature_cols)} variables")
print(f"  {feature_cols}")
print(f"Sensitivity feature set (no Tier C): {len(feature_cols_no_tierc)} variables")
print(f"  Removed: {tier_c_vars}")

# ==============================================================================
# 3. HOLDOUT SPLIT + REPEATED STRATIFIED K-FOLD CV
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 3: Holdout split + Repeated Stratified K-Fold CV")
print("=" * 70)

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from fold_imputation import impute_split

IMPUTE_CACHE = data_dir / "imputed_folds"   # cached fold-wise imputations

TOTAL_FOLDS = N_SPLITS * N_REPEATS

X = df[feature_cols]
y = df["stroke"]

# --- 80/20 stratified holdout split ---
# The holdout set is NEVER seen during CV training or model selection.
# It provides the final unbiased performance estimate for the paper.
X_train_cv_raw, X_holdout_raw, y_train_cv, y_holdout = train_test_split(
    X, y,
    test_size=HOLDOUT_FRAC,
    stratify=y,
    random_state=RANDOM_SEED,
)

print(f"\nHoldout split:")
print(f"  Training (CV):  {len(X_train_cv_raw):,} rows ({y_train_cv.sum():,} stroke, "
      f"{y_train_cv.mean()*100:.1f}%)")
print(f"  Holdout (test): {len(X_holdout_raw):,} rows ({y_holdout.sum():,} stroke, "
      f"{y_holdout.mean()*100:.1f}%)")

print(f"\nCV strategy: {N_SPLITS}-fold × {N_REPEATS} repeats = {TOTAL_FOLDS} fits")
print(f"Eval metrics: AUROC, Balanced Accuracy, AUPRC, Brier Score")

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED,
)

# Storage for results
cv_results = []
all_predictions = []  # For aggregated evaluation on CV folds
best_model_info = None
best_auroc = -1

print(f"\nTraining AutoGluon across {TOTAL_FOLDS} folds (on training set only)...\n")

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_train_cv_raw, y_train_cv)):
    print(f"--- Fold {fold_idx + 1}/{TOTAL_FOLDS} ---")

    # Fold-wise imputation: HyperImpute fitted on the fold's training rows only,
    # then applied to the fold's validation rows.
    X_tr_imp, X_te_imp = impute_split(
        X_train_cv_raw.iloc[train_idx], X_train_cv_raw.iloc[test_idx],
        codebook, feature_cols, cache_dir=IMPUTE_CACHE, tag=f"primary_fold{fold_idx}",
    )

    # Prepare train/test DataFrames with label column
    train_data = pd.concat([X_tr_imp, y_train_cv.iloc[train_idx]], axis=1)
    test_data = pd.concat([X_te_imp, y_train_cv.iloc[test_idx]], axis=1)

    # Train AutoGluon
    predictor = TabularPredictor(
        label="stroke",
        eval_metric="balanced_accuracy",
        path=str(models_dir / f"fold_{fold_idx}"),
        verbosity=2,
    )

    predictor.fit(
        train_data=train_data,
        time_limit=TIME_LIMIT,
        presets=PRESETS,
        num_cpus="auto",
    )

    # Predict on held-out fold
    y_pred_proba = predictor.predict_proba(test_data.drop(columns=["stroke"]))
    y_pred_proba_pos = y_pred_proba[1].values  # Probability of stroke=1

    y_true = test_data["stroke"].values
    y_pred = (y_pred_proba_pos >= 0.5).astype(int)

    # Compute metrics (threshold-dependent metrics at 0.50 kept for transparency)
    auroc = roc_auc_score(y_true, y_pred_proba_pos)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred_proba_pos)
    brier = brier_score_loss(y_true, y_pred_proba_pos)
    f1 = f1_score(y_true, y_pred)

    # Threshold learned from the fold's TRAINING data only (Youden's J on
    # AutoGluon's out-of-fold predictions for the training rows), then applied
    # to the fold's validation rows. NaN if OOF predictions are unavailable.
    oof_tr = autogluon_oof_proba(predictor)
    if oof_tr is not None and len(oof_tr) == len(train_idx):
        fold_thr = youden_threshold(y_train_cv.iloc[train_idx].values, oof_tr)
        tm = threshold_metrics(y_true, y_pred_proba_pos, fold_thr)
    else:
        tm = {k: np.nan for k in ["threshold", "balanced_accuracy", "f1", "sensitivity",
                                  "specificity", "ppv"]}

    # Logistic regression reference on the same imputed fold data
    Xtr_lr = one_hot(X_tr_imp); Xte_lr = one_hot(X_te_imp, Xtr_lr.columns)
    lr = logistic_reference().fit(Xtr_lr, y_train_cv.iloc[train_idx].values)
    p_lr = lr.predict_proba(Xte_lr)[:, 1]

    cv_results.append({
        "fold": fold_idx + 1,
        "auroc": auroc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "auprc": auprc,
        "brier_score": brier,
        "train_threshold": tm["threshold"],
        "balanced_accuracy_train_thr": tm["balanced_accuracy"],
        "f1_train_thr": tm["f1"],
        "sensitivity_train_thr": tm["sensitivity"],
        "specificity_train_thr": tm["specificity"],
        "ppv_train_thr": tm["ppv"],
        "lr_auroc": roc_auc_score(y_true, p_lr),
        "lr_auprc": average_precision_score(y_true, p_lr),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_stroke_test": y_true.sum(),
    })

    # Store predictions for aggregated analysis
    all_predictions.append(pd.DataFrame({
        "idx": test_idx,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba_pos,
        "fold": fold_idx + 1,
    }))

    # Track best fold for SHAP analysis
    if auroc > best_auroc:
        best_auroc = auroc
        best_model_info = {
            "fold": fold_idx + 1,
            "predictor": predictor,
            "test_data": test_data,
            "test_idx": test_idx,
            "auroc": auroc,
        }

    print(f"  AUROC={auroc:.4f}  BalAcc={bal_acc:.4f}  F1={f1:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")

    # Clean up to save disk space (keep best model)
    if fold_idx > 0 and fold_idx != best_model_info["fold"] - 1:
        import shutil
        shutil.rmtree(str(models_dir / f"fold_{fold_idx}"), ignore_errors=True)

# ==============================================================================
# 4. CV RESULTS SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 4: Cross-validation results")
print("=" * 70)

cv_df = pd.DataFrame(cv_results)

print(f"\nPerformance across {TOTAL_FOLDS} folds:")
print(f"  {'Metric':<25s} {'Mean':>8s} {'SD':>8s} {'Min':>8s} {'Max':>8s}")
print("  " + "-" * 55)
for metric in ["auroc", "auprc", "brier_score", "balanced_accuracy", "f1",
               "train_threshold", "balanced_accuracy_train_thr", "f1_train_thr",
               "sensitivity_train_thr", "specificity_train_thr", "ppv_train_thr",
               "lr_auroc", "lr_auprc"]:
    vals = cv_df[metric]
    print(f"  {metric:<25s} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} {vals.max():8.4f}")

# 95% CI for AUROC
auroc_mean = cv_df["auroc"].mean()
auroc_se = cv_df["auroc"].std() / np.sqrt(TOTAL_FOLDS)
auroc_ci_lo = auroc_mean - 1.96 * auroc_se
auroc_ci_hi = auroc_mean + 1.96 * auroc_se
print(f"\n  AUROC 95% CI: [{auroc_ci_lo:.4f}, {auroc_ci_hi:.4f}]")

cv_df.to_csv(artifacts_dir / "cv_results.csv", index=False)
print("Exported: cv_results.csv")

# ==============================================================================
# 4b. HOLDOUT SET EVALUATION
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 4b: Holdout set evaluation (final unbiased estimate)")
print("=" * 70)

# Retrain the best model on the FULL training set (all CV data)
# then evaluate on the holdout set that was never seen during CV.
# Imputation for the final model: fitted on the full 80% training set only,
# then applied to the sequestered holdout set.
X_train_cv, X_holdout = impute_split(
    X_train_cv_raw, X_holdout_raw, codebook, feature_cols,
    cache_dir=IMPUTE_CACHE, tag="primary_final",
)

print(f"\nRetraining best model on full training set ({len(X_train_cv):,} rows)...")

train_data_full = pd.concat([X_train_cv, y_train_cv], axis=1)
holdout_data = pd.concat([X_holdout, y_holdout], axis=1)

predictor_final = TabularPredictor(
    label="stroke",
    eval_metric="balanced_accuracy",
    path=str(models_dir / "final"),
    verbosity=2,
)

predictor_final.fit(
    train_data=train_data_full,
    time_limit=TIME_LIMIT_FINAL,
    presets=PRESETS,
    num_cpus="auto",
)

# Predict on holdout
y_holdout_proba = predictor_final.predict_proba(X_holdout)[1].values

# Record AutoGluon configuration relevant to class imbalance and thresholds.
# All reported metrics use predict_proba() with explicit thresholds (0.50 or the
# Youden threshold from CV OOF predictions); AutoGluon's internal decision
# threshold (if calibrated) is recorded here but NOT used for any reported metric.
import json
ag_config = {
    "eval_metric": "balanced_accuracy",
    "presets": PRESETS,
    "sample_weight": str(getattr(predictor_final, "sample_weight", None)),
    "internal_decision_threshold": getattr(predictor_final, "decision_threshold", None),
    "class_weights_or_resampling": "none specified (AutoGluon defaults)",
    "autogluon_version": __import__("autogluon.tabular").__version__
        if hasattr(__import__("autogluon.tabular"), "__version__") else None,
}
with open(artifacts_dir / "autogluon_config.json", "w") as f:
    json.dump(ag_config, f, indent=2, default=str)
print(f"AutoGluon config: {ag_config}")
y_holdout_pred = (y_holdout_proba >= 0.5).astype(int)
y_holdout_true = y_holdout.values

# Holdout metrics
holdout_auroc = roc_auc_score(y_holdout_true, y_holdout_proba)
holdout_balacc = balanced_accuracy_score(y_holdout_true, y_holdout_pred)
holdout_f1 = f1_score(y_holdout_true, y_holdout_pred)
holdout_auprc = average_precision_score(y_holdout_true, y_holdout_proba)
holdout_brier = brier_score_loss(y_holdout_true, y_holdout_proba)

print(f"\nHoldout performance (N={len(X_holdout):,}, {y_holdout.sum():,} stroke):")
print(f"  AUROC:              {holdout_auroc:.4f}")
print(f"  Balanced Accuracy:  {holdout_balacc:.4f}")
print(f"  F1 Score:           {holdout_f1:.4f}")
print(f"  AUPRC:              {holdout_auprc:.4f}")
print(f"  Brier Score:        {holdout_brier:.4f}")

print(f"\nClassification report at default threshold (0.5):")
print(classification_report(y_holdout_true, y_holdout_pred,
                            target_names=["No Stroke", "Stroke"]))

# --- Optimal classification threshold ---
# The default 0.5 threshold is too conservative for rare events (4.1% stroke).
# We find the optimal threshold using the OUT-OF-FOLD CV predictions (no leakage),
# then apply it to the holdout set.
#
# Strategy: maximize Youden's J statistic (Sensitivity + Specificity - 1),
# which is standard for clinical screening tools.

print("--- Optimal threshold search (from CV out-of-fold predictions) ---")

# Aggregate OOF predictions from CV
all_preds_df = pd.concat(all_predictions)
oof_preds = all_preds_df.groupby("idx").agg(
    y_true=("y_true", "first"),
    y_pred_proba=("y_pred_proba", "mean"),
).reset_index()

oof_true = oof_preds["y_true"].values
oof_proba = oof_preds["y_pred_proba"].values

# Search for optimal threshold using Youden's J on OOF data
from sklearn.metrics import roc_curve, precision_score, recall_score

fpr, tpr, roc_thresholds = roc_curve(oof_true, oof_proba)
youden_j = tpr - fpr
best_youden_idx = np.argmax(youden_j)
optimal_threshold_youden = roc_thresholds[best_youden_idx]

# Also find threshold that maximizes F1 on OOF data
thresholds_search = np.linspace(0.01, 0.50, 200)
f1_scores_search = [f1_score(oof_true, (oof_proba >= t).astype(int)) for t in thresholds_search]
optimal_threshold_f1 = thresholds_search[np.argmax(f1_scores_search)]

print(f"  Youden's J optimal threshold: {optimal_threshold_youden:.4f}")
print(f"  F1 optimal threshold:         {optimal_threshold_f1:.4f}")

# Use Youden's J threshold (standard for clinical screening)
optimal_threshold = optimal_threshold_youden

# Apply to holdout set (this is NOT leakage — threshold was found on CV data)
y_holdout_pred_optimal = (y_holdout_proba >= optimal_threshold).astype(int)
holdout_f1_optimal = f1_score(y_holdout_true, y_holdout_pred_optimal)
holdout_balacc_optimal = balanced_accuracy_score(y_holdout_true, y_holdout_pred_optimal)
holdout_recall_optimal = recall_score(y_holdout_true, y_holdout_pred_optimal)
holdout_precision_optimal = precision_score(y_holdout_true, y_holdout_pred_optimal,
                                            zero_division=0)

print(f"\nHoldout performance at optimal threshold ({optimal_threshold:.4f}):")
print(f"  Recall (Sensitivity):  {holdout_recall_optimal:.4f}")
print(f"  Precision (PPV):       {holdout_precision_optimal:.4f}")
print(f"  F1 Score:              {holdout_f1_optimal:.4f}")
print(f"  Balanced Accuracy:     {holdout_balacc_optimal:.4f}")

print(f"\nClassification report at optimal threshold ({optimal_threshold:.4f}):")
print(classification_report(y_holdout_true, y_holdout_pred_optimal,
                            target_names=["No Stroke", "Stroke"]))

print(f"Confusion matrix at optimal threshold:")
cm = confusion_matrix(y_holdout_true, y_holdout_pred_optimal)
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# Save holdout predictions for Monte Carlo analysis
holdout_preds_df = pd.DataFrame({
    "y_true": y_holdout_true,
    "y_pred_proba": y_holdout_proba,
    "y_pred_default": y_holdout_pred,
    "y_pred_optimal": y_holdout_pred_optimal,
    "optimal_threshold": optimal_threshold,
})
holdout_preds_df.to_csv(artifacts_dir / "holdout_predictions.csv", index=False)

# 95% CIs for holdout metrics (stratified bootstrap, 1,000 resamples).
# Threshold-dependent metrics at 0.50 and at the Youden threshold.
ci_050 = bootstrap_ci(y_holdout_true, y_holdout_proba, 0.5).assign(threshold="0.50")
ci_yj = bootstrap_ci(y_holdout_true, y_holdout_proba, optimal_threshold).assign(threshold="youden")
holdout_ci = pd.concat([ci_050, ci_yj])
holdout_ci.to_csv(artifacts_dir / "holdout_metrics_ci.csv", index=False)
print("\nHoldout metrics with 95% CIs:")
print(holdout_ci.round(3).to_string(index=False))

# ------------------------------------------------------------------------------
# Holdout discrimination by measurement period (Reviewer comment 6)
# BP: auscultatory 2003-2016 vs oscillometric 2017-2023.
# Physical activity: PAQ leisure-time items 2003-2006, GPAQ recreational items
# 2007-2018, leisure-time frequency/duration items 2021-2023.
# 95% CIs from 1,000 stratified bootstrap resamples of holdout participants.
# ------------------------------------------------------------------------------
hold_year = df.loc[X_holdout_raw.index, "year"].values
periods = {
    "BP auscultatory (2003-2016)":   hold_year < 2017,
    "BP oscillometric (2017-2023)":  hold_year >= 2017,
    "PA instrument 2003-2006":       hold_year <= 2005,
    "PA instrument 2007-2018":       (hold_year >= 2007) & (hold_year <= 2017),
    "PA instrument 2021-2023":       hold_year >= 2021,
}
for cyc in sorted(np.unique(hold_year)):                  # per-cycle holdout AUROC
    periods[f"Cycle {cyc}-{cyc + 1 if cyc < 2021 else 2023}"] = hold_year == cyc
rng_p = np.random.default_rng(RANDOM_SEED)
period_rows = []
for name, m in periods.items():
    yt, yp = y_holdout_true[m], y_holdout_proba[m]
    if yt.sum() < 5:
        continue
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    boots = []
    for _ in range(1000):
        idx = np.r_[rng_p.choice(pos, len(pos)), rng_p.choice(neg, len(neg))]
        boots.append(roc_auc_score(yt[idx], yp[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    period_rows.append({"period": name, "n": int(m.sum()), "n_stroke": int(yt.sum()),
                        "auroc": roc_auc_score(yt, yp), "ci_lower": lo, "ci_upper": hi})
period_df = pd.DataFrame(period_rows)
period_df.to_csv(artifacts_dir / "holdout_auroc_by_measurement_period.csv", index=False)
print("\nHoldout AUROC by measurement period:")
print(period_df.round(3).to_string(index=False))

# --- Calibration curve (reliability diagram) ---
# Against the OBSERVED self-reported label (descriptive only). Participants are
# grouped into deciles of predicted probability; vertical bars are 95% Wilson
# confidence intervals for the observed proportion in each decile.
def _wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    ph = k / n; den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

cal = pd.DataFrame({"y": y_holdout_true, "p": y_holdout_proba})
cal["bin"] = pd.qcut(cal["p"], q=10, labels=False, duplicates="drop")
cal_tab = cal.groupby("bin").agg(mean_pred=("p", "mean"), obs=("y", "mean"),
                                 events=("y", "sum"), n=("y", "size")).reset_index()
lo, hi = _wilson(cal_tab["events"], cal_tab["n"])
cal_tab["obs_lo"], cal_tab["obs_hi"] = lo, hi
cal_tab.to_csv(artifacts_dir / "calibration_deciles_holdout.csv", index=False)

lim = float(max(cal_tab["mean_pred"].max(), cal_tab["obs_hi"].max()) * 1.1)
fig, ax = plt.subplots(figsize=(7, 7))
ax.errorbar(cal_tab["mean_pred"], cal_tab["obs"],
            yerr=[cal_tab["obs"] - cal_tab["obs_lo"], cal_tab["obs_hi"] - cal_tab["obs"]],
            marker="o", linewidth=2, capsize=3, color="steelblue", label="Model (deciles, 95% CI)")
ax.plot([0, lim], [0, lim], linestyle="--", color="grey", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability", fontsize=12)
ax.set_ylabel("Observed proportion (self-reported stroke)", fontsize=12)
ax.set_title("Calibration against observed label (holdout)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim([0, lim]); ax.set_ylim([0, lim]); ax.set_aspect("equal"); ax.grid(alpha=0.3)

ax_inset = fig.add_axes([0.22, 0.50, 0.30, 0.18])
ax_inset.hist(y_holdout_proba, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax_inset.set_xlabel("Predicted P(stroke)", fontsize=8)
ax_inset.set_ylabel("Count", fontsize=8)
ax_inset.tick_params(labelsize=7)

plt.savefig(artifacts_dir / "fig_calibration_holdout.png", dpi=300, bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_calibration_holdout.pdf", bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_calibration_holdout.svg", bbox_inches="tight")
plt.close()
print("Saved: fig_calibration_holdout.png / .pdf / .svg")

# --- ROC Curve + Precision-Recall Curve (combined figure) ---
from sklearn.metrics import precision_recall_curve, roc_curve as sklearn_roc_curve

# ROC data
fpr_plot, tpr_plot, _ = sklearn_roc_curve(y_holdout_true, y_holdout_proba)

# PRC data
precision_plot, recall_plot, _ = precision_recall_curve(y_holdout_true, y_holdout_proba)
baseline_prevalence = y_holdout_true.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax1.plot(fpr_plot, tpr_plot, linewidth=2, color="steelblue",
         label=f"AutoGluon (AUROC={holdout_auroc:.3f})")
ax1.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
ax1.axvline(1 - (1 - fpr_plot[np.argmax(tpr_plot - fpr_plot)]), color="red",
            linestyle="-.", alpha=0.6, linewidth=1,
            label=f"Youden's J ({optimal_threshold:.3f})")
ax1.set_xlabel("1 − Specificity (FPR)")
ax1.set_ylabel("Sensitivity (TPR)")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_aspect("equal")
ax1.grid(alpha=0.2)
ax1.text(-0.1, 1.05, "a", transform=ax1.transAxes, fontsize=16, fontweight="bold")

# Precision-Recall Curve
ax2.plot(recall_plot, precision_plot, linewidth=2, color="steelblue",
         label=f"AutoGluon (AUPRC={holdout_auprc:.3f})")
ax2.axhline(baseline_prevalence, color="grey", linestyle="--",
            label=f"Baseline prevalence ({baseline_prevalence:.3f})")
ax2.set_xlabel("Recall (Sensitivity)")
ax2.set_ylabel("Precision (PPV)")
ax2.set_title("Precision-Recall Curve")
ax2.legend(loc="upper right")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.grid(alpha=0.2)
ax1.text(-0.1, 1.05, "b", transform=ax1.transAxes, fontsize=16, fontweight="bold")

plt.suptitle("Discrimination Performance (Holdout Set, N={:,})".format(len(y_holdout_true)),
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(artifacts_dir / "fig_roc_prc_holdout.png", dpi=300, bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_roc_prc_holdout.pdf", bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_roc_prc_holdout.svg", bbox_inches="tight")
plt.close()
print("Saved: fig_roc_prc_holdout.png / .pdf / .svg")

# ==============================================================================
# 5. BEST MODEL DETAILS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 5: Best model details")
print("=" * 70)

best_predictor = predictor_final
print(f"\nFinal model trained on full training set ({len(X_train_cv):,} rows)")
print(f"Holdout AUROC: {holdout_auroc:.4f}")

# AutoGluon leaderboard
leaderboard = best_predictor.leaderboard(silent=True)
print(f"\nAutoGluon leaderboard (top 10):")
print(leaderboard.head(10).to_string(index=False))

leaderboard.to_csv(artifacts_dir / "autogluon_leaderboard.csv", index=False)

# ------------------------------------------------------------------------------
# Model-type comparison on the holdout set (Reviewer 2 comment 8)
# For every model in the final AutoGluon predictor plus a logistic regression
# reference: AUROC and AUPRC (threshold-free), and threshold metrics at 0.50 and
# at each model's own Youden threshold learned from its training OOF predictions.
# ------------------------------------------------------------------------------
def _family(name):
    for k in ["NeuralNetTorch", "NeuralNetFastAI", "LightGBM", "CatBoost", "XGBoost",
              "RandomForest", "ExtraTrees", "KNeighbors", "LinearModel", "WeightedEnsemble"]:
        if k in name:
            return k
    return "Other"

comp_rows = []
y_tr_full = y_train_cv.values
for mname in leaderboard["model"].tolist():
    try:
        p_h = predictor_final.predict_proba(X_holdout, model=mname)[1].values
    except Exception as e:
        print(f"  skip {mname}: {e}"); continue
    oof_m = autogluon_oof_proba(predictor_final, model=mname)
    thr_m = youden_threshold(y_tr_full, oof_m) if (oof_m is not None and len(oof_m) == len(y_tr_full)) else np.nan
    row = {"model": mname, "family": _family(mname),
           "auroc": roc_auc_score(y_holdout_true, p_h),
           "auprc": average_precision_score(y_holdout_true, p_h),
           "brier_score": brier_score_loss(y_holdout_true, p_h)}
    row.update({f"{k}_t050": v for k, v in threshold_metrics(y_holdout_true, p_h, 0.5).items()
                if k in ["balanced_accuracy", "sensitivity", "specificity"]})
    if not np.isnan(thr_m):
        row.update({f"{k}_youden": v for k, v in threshold_metrics(y_holdout_true, p_h, thr_m).items()
                    if k in ["threshold", "balanced_accuracy", "sensitivity", "specificity"]})
    comp_rows.append(row)

# Logistic regression reference: fit on training set; Youden threshold from 5-fold OOF
Xtr_lr = one_hot(X_train_cv); Xho_lr = one_hot(X_holdout, Xtr_lr.columns)
oof_lr = cross_val_predict(logistic_reference(), Xtr_lr, y_tr_full, method="predict_proba",
                           cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED))[:, 1]
p_lr_h = logistic_reference().fit(Xtr_lr, y_tr_full).predict_proba(Xho_lr)[:, 1]
thr_lr = youden_threshold(y_tr_full, oof_lr)
row = {"model": "LogisticRegression_reference", "family": "LogisticRegression",
       "auroc": roc_auc_score(y_holdout_true, p_lr_h),
       "auprc": average_precision_score(y_holdout_true, p_lr_h),
       "brier_score": brier_score_loss(y_holdout_true, p_lr_h)}
row.update({f"{k}_t050": v for k, v in threshold_metrics(y_holdout_true, p_lr_h, 0.5).items()
            if k in ["balanced_accuracy", "sensitivity", "specificity"]})
row.update({f"{k}_youden": v for k, v in threshold_metrics(y_holdout_true, p_lr_h, thr_lr).items()
            if k in ["threshold", "balanced_accuracy", "sensitivity", "specificity"]})
comp_rows.append(row)

model_comp = pd.DataFrame(comp_rows).sort_values("auroc", ascending=False)
model_comp.to_csv(artifacts_dir / "model_comparison_holdout.csv", index=False)
family_best = model_comp.sort_values("auroc", ascending=False).groupby("family").head(1)
family_best.to_csv(artifacts_dir / "model_comparison_by_family.csv", index=False)
print("\nBest holdout AUROC by model family:")
print(family_best[["family", "model", "auroc", "auprc", "balanced_accuracy_t050",
                   "balanced_accuracy_youden"]].round(3).to_string(index=False))
print("Exported: model_comparison_holdout.csv, model_comparison_by_family.csv")

# --- Extract weighted ensemble composition ---
# The best model is typically a WeightedEnsemble that stacks base models.
# Extract which models are included and their stacking weights.
# Compatible with both old (.get_model_best()) and new (.model_best) AutoGluon APIs
try:
    best_model_name = best_predictor.model_best
except AttributeError:
    best_model_name = best_predictor.get_model_best()
print(f"\nBest model: {best_model_name}")

try:
    # Get full model info dict (stable public API)
    model_info = best_predictor.info()
    all_model_info = model_info.get("model_info", {})

    if best_model_name in all_model_info:
        best_info = all_model_info[best_model_name]
        print(f"Model type: {best_info.get('model_type', 'unknown')}")
        print(f"Stack level: {best_info.get('stacker_info', {}).get('stacker_level', 'N/A')}")

        # Children = base models used by the ensemble
        children = best_info.get("children_info", {})
        if children:
            print(f"\nBase models in ensemble ({len(children)}):")
            print(f"  {'Model':<45s} {'Val Score':>10s}")
            print("  " + "-" * 57)
            for child_name, child_info in children.items():
                child_score = all_model_info.get(child_name, {}).get("val_score", "?")
                if isinstance(child_score, float):
                    print(f"  {child_name:<45s} {child_score:10.4f}")
                else:
                    print(f"  {child_name:<45s} {str(child_score):>10s}")

    # Extract weights from the model object directly
    # AutoGluon's WeightedEnsembleModel stores weights internally
    try:
        ensemble_model = best_predictor._trainer.load_model(best_model_name)

        # Navigate the object to find weights
        weights = None
        child_names = None

        # Try common attribute paths across AutoGluon versions
        for attr_chain in [
            ("model", "weights_"),
            ("weights_",),
            ("model", "coef_"),
        ]:
            obj = ensemble_model
            found = True
            for attr in attr_chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and obj is not None:
                weights = np.array(obj).flatten()
                break

        # Get child model names
        for attr in ["base_model_names_", "stack_column_names"]:
            if hasattr(ensemble_model, attr):
                child_names = getattr(ensemble_model, attr)
                break

        if child_names is None and children:
            child_names = list(children.keys())

        if weights is not None and len(weights) > 0:
            if child_names is None:
                child_names = [f"model_{i}" for i in range(len(weights))]

            n = min(len(child_names), len(weights))
            weight_df = pd.DataFrame({
                "model": list(child_names)[:n],
                "weight": weights[:n],
            }).sort_values("weight", ascending=False)
            weight_df = weight_df[weight_df["weight"] > 0]

            print(f"\nWeighted Ensemble weights ({len(weight_df)} non-zero):")
            print(f"  {'Model':<45s} {'Weight':>8s}")
            print("  " + "-" * 55)
            for _, row in weight_df.iterrows():
                print(f"  {row['model']:<45s} {row['weight']:8.4f}")
            print(f"  {'TOTAL':<45s} {weight_df['weight'].sum():8.4f}")

            weight_df.to_csv(artifacts_dir / "ensemble_weights.csv", index=False)
            print("\nExported: ensemble_weights.csv")
        else:
            print("\nCould not extract individual weights from ensemble object.")
            print("Refer to the leaderboard and children info above for model composition.")

    except Exception as e2:
        print(f"\nCould not load ensemble model object: {e2}")

except Exception as e:
    print(f"\n⚠ Could not extract ensemble details: {e}")
    print("  Check autogluon_leaderboard.csv for model rankings.")

# Model architecture summary for methods section
print(f"\nFor methods section:")
print(f"  AutoGluon preset: best_quality")
print(f"  Time limit: {TIME_LIMIT}s per CV fold, {TIME_LIMIT_FINAL}s for final model")
print(f"  Eval metric: balanced_accuracy")
print(f"  Total models trained per fold: {len(leaderboard)}")
print(f"  Best model type: {best_model_name}")

# ==============================================================================
# 6. SHAP ANALYSIS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 6: SHAP feature importance")
print("=" * 70)

import shap

# Use holdout set for SHAP — this is the data the model has never seen
test_X = X_holdout.copy()
test_y = y_holdout

# SHAP's permutation explainer requires a fully numeric matrix.
# AutoGluon handles categoricals internally, but SHAP's masker uses
# np.isfinite which fails on string columns. We one-hot encode
# categoricals (no implied ordering) for SHAP, and decode back to
# the original format before passing to AutoGluon's predict_proba.
cat_cols_in_data = test_X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols_in_data = [c for c in test_X.columns if c not in cat_cols_in_data]
print(f"Categorical columns to one-hot encode for SHAP: {cat_cols_in_data}")

# Create one-hot encoded version for SHAP masker
test_X_encoded = pd.get_dummies(test_X, columns=cat_cols_in_data, drop_first=False, dtype=float)
encoded_columns = test_X_encoded.columns.tolist()

# Ensure all numeric columns are float
for col in numeric_cols_in_data:
    test_X_encoded[col] = test_X_encoded[col].astype(float)

print(f"Encoded feature matrix: {len(encoded_columns)} columns")

# Sample for SHAP (full dataset is too slow)
n_shap = min(N_SHAP, len(test_X))
shap_sample_idx = np.random.choice(len(test_X), n_shap, replace=False)
X_shap_encoded = test_X_encoded.iloc[shap_sample_idx]
X_shap_original = test_X.iloc[shap_sample_idx]

# Model function: takes one-hot array → reconstructs original DataFrame → predicts
original_columns = test_X.columns.tolist()

def model_predict(X_onehot_array):
    """Convert one-hot array back to DataFrame with original categoricals for AutoGluon."""
    df_encoded = pd.DataFrame(X_onehot_array, columns=encoded_columns)

    # Reconstruct original categorical columns from one-hot dummies
    df_original = df_encoded[numeric_cols_in_data].copy()
    for cat_col in cat_cols_in_data:
        # Find dummy columns for this categorical
        dummy_cols = [c for c in encoded_columns if c.startswith(cat_col + "_")]
        if dummy_cols:
            # idxmax across dummies → category name (strip prefix)
            prefix_len = len(cat_col) + 1
            df_original[cat_col] = df_encoded[dummy_cols].idxmax(axis=1).str[prefix_len:]

    # Reorder to match original column order
    df_original = df_original[original_columns]
    proba = best_predictor.predict_proba(df_original)
    return proba[1].values

print(f"Computing SHAP values for {n_shap} samples...")

try:
    explainer = shap.Explainer(model_predict, X_shap_encoded.values,
                                algorithm="permutation",
                                feature_names=encoded_columns)
    shap_values = explainer(X_shap_encoded.values)

    # Global feature importance (mean |SHAP|)
    # Aggregate one-hot dummies back to original variable level:
    # sum the mean |SHAP| of all dummies belonging to the same categorical
    shap_per_column = np.abs(shap_values.values).mean(axis=0)

    var_importance = {}
    for col in numeric_cols_in_data:
        idx = encoded_columns.index(col)
        var_importance[col] = shap_per_column[idx]
    for cat_col in cat_cols_in_data:
        dummy_cols = [c for c in encoded_columns if c.startswith(cat_col + "_")]
        dummy_idxs = [encoded_columns.index(c) for c in dummy_cols]
        var_importance[cat_col] = sum(shap_per_column[i] for i in dummy_idxs)

    shap_importance = pd.DataFrame({
        "variable": list(var_importance.keys()),
        "mean_abs_shap": list(var_importance.values()),
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\nSHAP feature importance (top 20):")
    print(f"  {'Variable':<30s} {'Mean |SHAP|':>12s}")
    print("  " + "-" * 45)
    for _, row in shap_importance.iterrows():
        print(f"  {row['variable']:<30s} {row['mean_abs_shap']:12.6f}")

    shap_importance.to_csv(artifacts_dir / "shap_importance.csv", index=False)

    # SHAP beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "fig_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(artifacts_dir / "fig_shap_beeswarm.pdf", bbox_inches="tight")
    plt.savefig(artifacts_dir / "fig_shap_beeswarm.svg", bbox_inches="tight")
    plt.close()
    print("\nSaved: fig_shap_beeswarm.png / .pdf / .svg")

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "fig_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(artifacts_dir / "fig_shap_bar.pdf", bbox_inches="tight")
    plt.savefig(artifacts_dir / "fig_shap_bar.svg", bbox_inches="tight")
    plt.close()
    print("Saved: fig_shap_bar.png / .pdf / .svg")

    # SHAP dependence plots for top continuous features
    # These show the non-linear relationship between each feature and its
    # SHAP value — the modern equivalent of odds ratio plots.
    # Only plot continuous features (dependence plots on one-hot dummies
    # are not informative since x-axis is binary 0/1).
    top_continuous = [f for f in shap_importance["variable"].tolist()
                      if f in numeric_cols_in_data and f in encoded_columns]
    top_n_dependence = min(6, len(top_continuous))
    top_features = top_continuous[:top_n_dependence]

    if top_features:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            if i >= len(axes):
                break
            ax = axes[i]
            feat_idx = encoded_columns.index(feat)
            shap.plots.scatter(
                shap_values[:, feat_idx],
                color=shap_values,
                ax=ax,
                show=False,
            )
            ax.set_title(feat, fontsize=11, fontweight="bold")

        # Hide unused subplots
        for j in range(len(top_features), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("SHAP Dependence Plots (Top Continuous Features)",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(artifacts_dir / "fig_shap_dependence.png", dpi=300, bbox_inches="tight")
        plt.savefig(artifacts_dir / "fig_shap_dependence.pdf", bbox_inches="tight")
        plt.savefig(artifacts_dir / "fig_shap_dependence.svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: fig_shap_dependence.png / .pdf / .svg (top {top_n_dependence} features)")

except Exception as e:
    print(f"\n⚠ SHAP computation failed: {e}")
    print("  Falling back to AutoGluon feature importance...")
    try:
        fi = best_predictor.feature_importance(holdout_data)
        # feature_importance returns a DataFrame with importance as values and features as index
        shap_importance = fi.reset_index()
        shap_importance.columns = ["variable"] + list(fi.columns)
        # Sort by the first importance column
        sort_col = shap_importance.columns[1]
        shap_importance = shap_importance.sort_values(sort_col, ascending=False)
        print(shap_importance.to_string(index=False))
        shap_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    except Exception as e2:
        print(f"  Feature importance also failed: {e2}")
        print("  Skipping feature importance analysis.")

# ==============================================================================
# 7. PROBABILISTIC BIAS ANALYSIS FOR OUTCOME MISCLASSIFICATION
# ==============================================================================
# Goal: estimate model performance against the LATENT true stroke outcome T,
# given the observed self-report Y* (MCQ160F) and external ranges for the
# sensitivity (Se) and specificity (Sp) of self-report.
#
# Model for the underlying outcome (nondifferential misclassification):
#   Y* is conditionally independent of the model score s given T, so
#     P(Y*=1 | s) = Se * pi(s) + (1 - Sp) * (1 - pi(s)),   pi(s) = P(T=1 | s)
#   => pi(s) = (q(s) - (1 - Sp)) / (Se + Sp - 1),  q(s) = P(Y*=1 | s)
# q(s) is estimated by isotonic regression of the observed holdout labels on
# the model score. For each participant, the posterior probability of T=1 is
#     P(T=1 | Y*=1, s) = Se * pi(s) / q(s)
#     P(T=1 | Y*=0, s) = (1 - Se) * pi(s) / (1 - q(s))
# In each iteration, Se and Sp are drawn from their ranges, T is sampled for
# every holdout participant from these posteriors, and the (fixed) model
# scores are evaluated against T. Implied true prevalence = mean(pi(s)).
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 7: Probabilistic bias analysis (latent outcome reconstruction)")
print("=" * 70)

from sklearn.isotonic import IsotonicRegression

print(f"\n  Iterations: {N_MC}")
print(f"  Sensitivity ~ U{SENS_RANGE}   Specificity ~ U{SPEC_RANGE}")

y_obs = y_holdout_true
s_hold = y_holdout_proba
thr = optimal_threshold
yhat_thr = (s_hold >= thr).astype(int)

# q(s) = P(Y*=1 | s): monotone calibration of the model score to observed labels
q_hat = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip") \
    .fit(s_hold, y_obs).predict(s_hold)

def _threshold_metrics(y, yhat):
    tp = np.sum((yhat == 1) & (y == 1)); fn = np.sum((yhat == 0) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0)); fp = np.sum((yhat == 1) & (y == 0))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    return {"balanced_accuracy": (sens + spec) / 2, "f1": 2 * ppv * sens / max(ppv + sens, 1e-12),
            "recall": sens, "specificity": spec, "precision": ppv,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}

naive = {"auroc": holdout_auroc, "auprc": holdout_auprc, "brier_score": holdout_brier,
         **_threshold_metrics(y_obs, yhat_thr)}

rng_pba = np.random.default_rng(RANDOM_SEED)
pba_results = []
for i in range(N_MC):
    se = rng_pba.uniform(*SENS_RANGE)
    sp = rng_pba.uniform(*SPEC_RANGE)
    pi_s = np.clip((q_hat - (1 - sp)) / (se + sp - 1), 0, 1)
    p_t1_pos = np.clip(se * pi_s / q_hat, 0, 1)
    p_t1_neg = np.clip((1 - se) * pi_s / (1 - q_hat), 0, 1)
    u = rng_pba.random(len(y_obs))
    T = np.where(y_obs == 1, u < p_t1_pos, u < p_t1_neg).astype(int)
    if T.sum() < 5 or T.sum() == len(T):
        continue
    pba_results.append({
        "iteration": i + 1, "sensitivity": se, "specificity": sp,
        "implied_true_prevalence": pi_s.mean(),
        "frac_scores_pi_clipped_zero": np.mean(pi_s == 0),
        "auroc": roc_auc_score(T, s_hold),
        "auprc": average_precision_score(T, s_hold),
        "brier_score": brier_score_loss(T, s_hold),
        **_threshold_metrics(T, yhat_thr),
    })

mc_df = pd.DataFrame(pba_results)
print(f"\n  Valid iterations: {len(mc_df)} / {N_MC}")
print(f"  Observed prevalence (holdout): {y_obs.mean():.4f}")
print(f"  Implied true prevalence: median {mc_df['implied_true_prevalence'].median():.4f} "
      f"[{mc_df['implied_true_prevalence'].quantile(0.025):.4f}, "
      f"{mc_df['implied_true_prevalence'].quantile(0.975):.4f}]")

summary_rows = []
print(f"\n  {'Metric':<20s} {'Naive':>8s} {'Median':>8s} {'2.5%':>8s} {'97.5%':>8s}")
print("  " + "-" * 56)
for m in ["auroc", "auprc", "brier_score", "balanced_accuracy", "f1",
          "recall", "specificity", "precision"]:
    v = mc_df[m]
    summary_rows.append({"metric": m, "naive": naive[m], "median": v.median(),
                         "si_lower": v.quantile(0.025), "si_upper": v.quantile(0.975)})
    print(f"  {m:<20s} {naive[m]:8.4f} {v.median():8.4f} {v.quantile(0.025):8.4f} {v.quantile(0.975):8.4f}")

mc_df.to_csv(artifacts_dir / "monte_carlo_misclassification.csv", index=False)
pd.DataFrame(summary_rows).to_csv(artifacts_dir / "bias_analysis_summary.csv", index=False)
print("\nExported: monte_carlo_misclassification.csv, bias_analysis_summary.csv")

# --- Plot: latent-outcome performance distributions vs naive ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
plot_metrics = [
    ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier_score", "Brier score"),
    ("balanced_accuracy", "Balanced accuracy\n(Youden threshold)"),
    ("f1", "F1 score\n(Youden threshold)"),
    ("recall", "Sensitivity\n(Youden threshold)"),
    ("precision", "PPV\n(Youden threshold)"),
    ("implied_true_prevalence", "Implied true prevalence"),
]
for i, (m, title) in enumerate(plot_metrics):
    ax = axes[i]
    v = mc_df[m]
    ax.hist(v, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ref = y_obs.mean() if m == "implied_true_prevalence" else naive[m]
    ref_lab = "Observed" if m == "implied_true_prevalence" else "Naive"
    ax.axvline(ref, color="red", linestyle="--", linewidth=2, label=f"{ref_lab}: {ref:.3f}")
    ax.axvline(v.quantile(0.025), color="orange", linestyle=":", linewidth=1.5,
               label=f"2.5%: {v.quantile(0.025):.3f}")
    ax.axvline(v.quantile(0.975), color="orange", linestyle=":", linewidth=1.5,
               label=f"97.5%: {v.quantile(0.975):.3f}")
    ax.set_xlabel(title); ax.set_ylabel("Count"); ax.set_title(title)
    ax.legend(fontsize=7)
    ax.text(-0.1, 1.05, chr(97 + i), transform=ax.transAxes, fontsize=16, fontweight="bold")

plt.suptitle("Probabilistic bias analysis: performance against the reconstructed latent outcome\n"
             f"(N={N_MC}, Se ~ U({SENS_RANGE[0]}, {SENS_RANGE[1]}), "
             f"Sp ~ U({SPEC_RANGE[0]}, {SPEC_RANGE[1]}), nondifferential misclassification)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(artifacts_dir / "fig_monte_carlo_misclassification.png", dpi=300, bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_monte_carlo_misclassification.pdf", bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_monte_carlo_misclassification.svg", bbox_inches="tight")
plt.close()
print("Saved: fig_monte_carlo_misclassification.png / .pdf / .svg")

# ------------------------------------------------------------------------------
# Table 2 export: every manuscript value written directly from code
# ------------------------------------------------------------------------------
_pba = pd.DataFrame(summary_rows).set_index("metric")
_ci = {t: g.set_index("metric") for t, g in holdout_ci.groupby("threshold")}
_fmt_ci = lambda t, m: f"{_ci[t].loc[m,'estimate']:.3f} [{_ci[t].loc[m,'ci_lower']:.3f}, {_ci[t].loc[m,'ci_upper']:.3f}]"
_h050 = threshold_metrics(y_holdout_true, y_holdout_proba, 0.5)
_hyj = threshold_metrics(y_holdout_true, y_holdout_proba, optimal_threshold)
_fmt_cv = lambda c: f"{cv_df[c].mean():.3f} ± {cv_df[c].std():.3f}" if c in cv_df and cv_df[c].notna().any() else "---"
_fmt_pba = lambda m: f"{_pba.loc[m,'median']:.3f} [{_pba.loc[m,'si_lower']:.3f}, {_pba.loc[m,'si_upper']:.3f}]"
table2 = pd.DataFrame([
    ["AUROC", _fmt_cv("auroc"), _fmt_ci("0.50", "auroc"), "---", _fmt_pba("auroc")],
    ["AUPRC", _fmt_cv("auprc"), _fmt_ci("0.50", "auprc"), "---", _fmt_pba("auprc")],
    ["Brier score", _fmt_cv("brier_score"), _fmt_ci("0.50", "brier_score"), "---", _fmt_pba("brier_score")],
    ["Calibration intercept", "---", _fmt_ci("0.50", "calibration_intercept"), "---", "---"],
    ["Calibration slope", "---", _fmt_ci("0.50", "calibration_slope"), "---", "---"],
    ["Balanced accuracy", _fmt_cv("balanced_accuracy_train_thr"), _fmt_ci("0.50", "balanced_accuracy"),
     _fmt_ci("youden", "balanced_accuracy"), _fmt_pba("balanced_accuracy")],
    ["F1 score", _fmt_cv("f1_train_thr"), _fmt_ci("0.50", "f1"), _fmt_ci("youden", "f1"), _fmt_pba("f1")],
    ["Sensitivity (model recall)", _fmt_cv("sensitivity_train_thr"), _fmt_ci("0.50", "sensitivity"),
     _fmt_ci("youden", "sensitivity"), _fmt_pba("recall")],
    ["Specificity", _fmt_cv("specificity_train_thr"), _fmt_ci("0.50", "specificity"),
     _fmt_ci("youden", "specificity"), _fmt_pba("specificity")],
    ["PPV", _fmt_cv("ppv_train_thr"), _fmt_ci("0.50", "ppv"), _fmt_ci("youden", "ppv"), _fmt_pba("precision")],
    ["TP / FN / TN / FP", "---", f"{_h050['tp']} / {_h050['fn']} / {_h050['tn']} / {_h050['fp']}",
     f"{_hyj['tp']} / {_hyj['fn']} / {_hyj['tn']} / {_hyj['fp']}", "---"],
], columns=["Metric", "Cross-validation mean ± SD (train-derived threshold)",
            "Holdout (t = 0.50) [95% CI]", f"Holdout (Youden t = {optimal_threshold:.3f}) [95% CI]",
            "Latent-outcome median [95% SI]"])
table2.to_csv(artifacts_dir / "table2_performance.csv", index=False)
print("\nTable 2:"); print(table2.to_string(index=False))
print("Exported: table2_performance.csv")

# ==============================================================================
# 8. TIER C SENSITIVITY ANALYSIS (with/without alcohol)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 8: Tier C sensitivity analysis")
print("=" * 70)

if tier_c_vars:
    print(f"\nRe-running CV WITHOUT Tier C variables: {tier_c_vars}")

    # Use the same 80/20 split — only remove Tier C columns
    # Tier C variables are removed BEFORE imputation so they do not inform
    # imputation of the remaining variables.
    X_train_cv_no_tierc_raw = X_train_cv_raw.drop(columns=tier_c_vars, errors="ignore")
    X_holdout_no_tierc_raw = X_holdout_raw.drop(columns=tier_c_vars, errors="ignore")

    rskf_sens = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,  # Same splits as primary analysis
    )

    sens_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf_sens.split(X_train_cv_no_tierc_raw, y_train_cv)):
        X_tr_imp, X_te_imp = impute_split(
            X_train_cv_no_tierc_raw.iloc[train_idx], X_train_cv_no_tierc_raw.iloc[test_idx],
            codebook, feature_cols_no_tierc, cache_dir=IMPUTE_CACHE, tag=f"notierc_fold{fold_idx}",
        )
        train_data = pd.concat([X_tr_imp, y_train_cv.iloc[train_idx]], axis=1)
        test_data = pd.concat([X_te_imp, y_train_cv.iloc[test_idx]], axis=1)

        predictor_sens = TabularPredictor(
            label="stroke",
            eval_metric="balanced_accuracy",
            path=str(models_dir / f"sens_fold_{fold_idx}"),
            verbosity=2,
        )

        predictor_sens.fit(
            train_data=train_data,
            time_limit=TIME_LIMIT,
            presets=PRESETS,
            num_cpus="auto",
        )

        y_pred_proba_sens = predictor_sens.predict_proba(
            test_data.drop(columns=["stroke"])
        )[1].values
        y_true_sens = test_data["stroke"].values

        auroc_sens = roc_auc_score(y_true_sens, y_pred_proba_sens)
        balacc_sens = balanced_accuracy_score(y_true_sens,
                                              (y_pred_proba_sens >= optimal_threshold).astype(int))

        sens_results.append({
            "fold": fold_idx + 1,
            "auroc": auroc_sens,
            "balanced_accuracy": balacc_sens,
        })

        # Clean up
        import shutil
        shutil.rmtree(str(models_dir / f"sens_fold_{fold_idx}"), ignore_errors=True)

        if (fold_idx + 1) % 5 == 0:
            print(f"  Completed {fold_idx + 1}/{TOTAL_FOLDS} sensitivity folds")

    sens_df = pd.DataFrame(sens_results)

    # --- Also evaluate on holdout set (apples-to-apples with primary model) ---
    print(f"\n  Training final no-Tier-C model on full training set...")
    X_train_cv_no_tierc, X_holdout_no_tierc = impute_split(
        X_train_cv_no_tierc_raw, X_holdout_no_tierc_raw, codebook, feature_cols_no_tierc,
        cache_dir=IMPUTE_CACHE, tag="notierc_final",
    )
    train_data_no_tierc_full = pd.concat([X_train_cv_no_tierc, y_train_cv], axis=1)

    predictor_no_tierc_final = TabularPredictor(
        label="stroke",
        eval_metric="balanced_accuracy",
        path=str(models_dir / "final_no_tierc"),
        verbosity=2,
    )
    predictor_no_tierc_final.fit(
        train_data=train_data_no_tierc_full,
        time_limit=TIME_LIMIT_FINAL,
        presets=PRESETS,
        num_cpus="auto",
    )

    y_holdout_proba_no_tierc = predictor_no_tierc_final.predict_proba(X_holdout_no_tierc)[1].values
    holdout_auroc_no_tierc = roc_auc_score(y_holdout_true, y_holdout_proba_no_tierc)
    holdout_balacc_no_tierc = balanced_accuracy_score(
        y_holdout_true, (y_holdout_proba_no_tierc >= optimal_threshold).astype(int)
    )

    import shutil
    shutil.rmtree(str(models_dir / "final_no_tierc"), ignore_errors=True)

    print(f"\nComparison: Primary vs. Sensitivity (no Tier C)")
    print(f"\n  CV results ({TOTAL_FOLDS} folds):")
    print(f"  {'Metric':<25s} {'Primary':>10s} {'No Tier C':>10s} {'Diff':>10s}")
    print("  " + "-" * 60)
    for metric in ["auroc", "balanced_accuracy"]:
        primary_mean = cv_df[metric].mean()
        sens_mean = sens_df[metric].mean()
        diff = sens_mean - primary_mean
        print(f"  {metric:<25s} {primary_mean:10.4f} {sens_mean:10.4f} {diff:+10.4f}")

    print(f"\n  Holdout results:")
    print(f"  {'Metric':<25s} {'Primary':>10s} {'No Tier C':>10s} {'Diff':>10s}")
    print("  " + "-" * 60)
    print(f"  {'auroc':<25s} {holdout_auroc:10.4f} {holdout_auroc_no_tierc:10.4f} "
          f"{holdout_auroc_no_tierc - holdout_auroc:+10.4f}")
    print(f"  {'balanced_accuracy':<25s} {holdout_balacc:10.4f} {holdout_balacc_no_tierc:10.4f} "
          f"{holdout_balacc_no_tierc - holdout_balacc:+10.4f}")

    auroc_diff = abs(holdout_auroc - holdout_auroc_no_tierc)
    if auroc_diff < 0.01:
        print(f"\n  ✓ Holdout AUROC difference ({auroc_diff:.4f}) < 0.01 — Tier C variables")
        print(f"    do not materially affect performance. Primary model is robust.")
    else:
        print(f"\n  ⚠ Holdout AUROC difference ({auroc_diff:.4f}) ≥ 0.01 — Tier C variables")
        print(f"    have a meaningful impact. Report both results in the paper.")

    sens_df.to_csv(artifacts_dir / "sensitivity_no_tierc_results.csv", index=False)
    print("\nExported: sensitivity_no_tierc_results.csv")
else:
    print("\nNo Tier C variables — sensitivity analysis not needed.")

# ==============================================================================
# 9. FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("MODELING COMPLETE")
print("=" * 70)
print(f"""
Primary Results:
  CV ({N_SPLITS}-fold × {N_REPEATS} repeats, on 80% training set):
    AUROC:              {cv_df['auroc'].mean():.4f} ± {cv_df['auroc'].std():.4f}
    Balanced Accuracy:  {cv_df['balanced_accuracy'].mean():.4f} ± {cv_df['balanced_accuracy'].std():.4f}
    F1 Score:           {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}
    AUPRC:              {cv_df['auprc'].mean():.4f} ± {cv_df['auprc'].std():.4f}
    Brier Score:        {cv_df['brier_score'].mean():.4f} ± {cv_df['brier_score'].std():.4f}

  Holdout (20% never seen during training):
    AUROC:              {holdout_auroc:.4f}
    Balanced Accuracy:  {holdout_balacc:.4f}
    F1 Score:           {holdout_f1:.4f}
    AUPRC:              {holdout_auprc:.4f}
    Brier Score:        {holdout_brier:.4f}

Probabilistic bias analysis (N={N_MC}, holdout set, latent outcome):
  Naive AUROC:                {holdout_auroc:.4f}
  AUROC vs latent outcome:    {mc_df['auroc'].median():.4f} [{mc_df['auroc'].quantile(0.025):.4f}, {mc_df['auroc'].quantile(0.975):.4f}]

Outputs (all figures in .png, .pdf, .svg):
  Data:
    holdout_predictions.csv                    (holdout predicted probabilities)
    cv_results.csv                             (per-fold CV metrics)
    autogluon_leaderboard.csv                  (model ensemble details)
    ensemble_weights.csv                       (weighted ensemble composition)
    shap_importance.csv                        (SHAP feature rankings)
    monte_carlo_misclassification.csv          (bias analysis iterations)
    bias_analysis_summary.csv                  (bias analysis median and 95% SI)
    sensitivity_no_tierc_results.csv           (Tier C sensitivity)

  Figures:
    fig_calibration_holdout                    (calibration / reliability diagram)
    fig_roc_prc_holdout                        (ROC + precision-recall curves)
    fig_shap_beeswarm                          (SHAP global importance)
    fig_shap_bar                               (SHAP bar plot)
    fig_shap_dependence                        (SHAP dependence plots, top 6)
    fig_monte_carlo_misclassification          (MC distribution plot)
""")