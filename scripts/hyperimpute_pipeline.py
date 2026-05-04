"""
================================================================================
NHANES Stroke Prediction — HyperImpute Imputation Pipeline
================================================================================
Codebook-driven variable roles + full diagnostic chain.

Workflow:
  1. Load data & codebook → dynamically assign variable roles
  2. Characterize missingness (rates + mechanism chi-sq tests)
  3. Apply tiered exclusion (data-driven, all decisions logged)
  4. Run HyperImpute with column-wise model selection
  5. Post-process: decode categoricals, enforce clinical ranges
  6. Validate: distributional checks + correlation structure comparison
  7. Export imputed dataset + all diagnostics

Requirements:
  pip install hyperimpute pandas numpy pyarrow scikit-learn scipy matplotlib seaborn

References:
  - Jarrett et al. (2022). HyperImpute: Generalized Iterative Imputation
    with Automatic Model Selection. ICML 2022. arXiv:2206.07769
  - Madley-Dowd et al. (2019). The proportion of missing data should not be
    used to guide decisions on multiple imputation. IJE 48(4):1294-1302.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set pandas to use standard numpy-backed dtypes (avoids Arrow StringDtype issues)
pd.set_option("future.infer_string", False)

# ==============================================================================
# 1. LOAD DATA & CODEBOOK
# ==============================================================================

# Identify the project root from the script's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Define the two main destinations
data_dir = project_root / "data"
artifacts_dir = project_root / "artifacts"

# Ensure directories exist
artifacts_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

print("=" * 70)
print("STEP 1: Loading data and codebook")
print("=" * 70)

data_path = data_dir / "nhanes_stroke_analytic_2003_2023.csv"
df = pd.read_csv(data_path)

codebook_path = artifacts_dir / "variable_codebook.csv"
codebook = pd.read_csv(codebook_path)

print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Stroke cases: {df['stroke'].sum():,} ({df['stroke'].mean()*100:.1f}%)")
print(f"Cycles: {sorted(df['year'].unique())}")
print(f"Codebook: {len(codebook)} variables defined")

# ==============================================================================
# 2. DYNAMIC VARIABLE ROLE ASSIGNMENT (from codebook)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 2: Variable role assignment (from codebook)")
print("=" * 70)

# Administrative variables (never imputed)
id_vars = ["seqn", "year"]
design_vars = ["sdmvpsu", "sdmvstra", "wtmec2yr", "wtint2yr"]
outcome_var = "stroke"

# Filter codebook to predictors only
predictors_df = codebook[~codebook["variable"].isin([outcome_var] + id_vars + design_vars)]

# Split by type
continuous_predictors = predictors_df[predictors_df["type"] == "continuous"]["variable"].tolist()
categorical_predictors = predictors_df[predictors_df["type"] == "categorical"]["variable"].tolist()
binary_predictors = predictors_df[predictors_df["type"] == "binary"]["variable"].tolist()

all_predictors = continuous_predictors + categorical_predictors + binary_predictors

# Safety check: ensure columns exist in the actual dataset
missing_cols = [c for c in all_predictors if c not in df.columns]
if missing_cols:
    print(f"\n⚠ WARNING: Codebook lists columns not in data: {missing_cols}")
    all_predictors = [c for c in all_predictors if c in df.columns]
    continuous_predictors = [c for c in continuous_predictors if c in df.columns]
    categorical_predictors = [c for c in categorical_predictors if c in df.columns]
    binary_predictors = [c for c in binary_predictors if c in df.columns]

print(f"\nCandidate predictors: {len(all_predictors)}")
print(f"  Continuous:   {len(continuous_predictors)} → {continuous_predictors}")
print(f"  Categorical:  {len(categorical_predictors)} → {categorical_predictors}")
print(f"  Binary:       {len(binary_predictors)} → {binary_predictors}")

# ==============================================================================
# 3. MISSINGNESS CHARACTERIZATION
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 3: Missingness characterization")
print("=" * 70)

# --- Missingness rates and tier assignment ---
miss_report = pd.DataFrame({
    "variable": all_predictors,
    "n_total": [len(df) for _ in all_predictors],
    "n_miss": [df[c].isna().sum() for c in all_predictors],
    "n_obs": [df[c].notna().sum() for c in all_predictors],
    "pct_miss": [df[c].isna().mean() * 100 for c in all_predictors],
})

def assign_tier(pct):
    if pct < 5:   return "A: Impute freely (<5%)"
    if pct < 30:  return "B: Impute + verify diagnostics (SD ratio, mean shift, KS statistic, correlation preservation)"
    if pct < 50:  return "C: Impute + sensitivity analysis (30-50%)"
    return "D: EXCLUDE from primary model (>50%)"

miss_report["tier"] = miss_report["pct_miss"].apply(assign_tier)
miss_report = miss_report.sort_values("pct_miss", ascending=False)

print("\nMissingness summary:")
print(miss_report.to_string(index=False))

# --- Missingness mechanism diagnostics (chi-squared tests) ---
# Tests whether each variable's missingness is associated with stroke status.
# Significant association → data is NOT MCAR → imputation is justified
# (complete-case analysis would be biased).

print("\n--- Missingness mechanism diagnostics ---")
print("Testing association between missingness and stroke status (chi-sq):\n")
print(f"  {'Variable':<28s} {'chi2':>8s} {'p':>10s}   {'Interpretation'}")
print("  " + "-" * 70)

mechanism_results = []
for var in all_predictors:
    n_miss = df[var].isna().sum()
    if 0 < n_miss < len(df):
        miss_indicator = df[var].isna().astype(int)
        contingency = pd.crosstab(miss_indicator, df["stroke"])
        if contingency.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            mechanism = "MAR/MNAR likely" if p < 0.05 else "MCAR plausible"
            flag = " ***" if p < 0.05 else ""
            print(f"  {var:<28s} {chi2:8.2f} {p:10.4f}   {mechanism}{flag}")
            mechanism_results.append({
                "variable": var, "chi2": chi2, "p_value": p, "mechanism": mechanism
            })
    else:
        print(f"  {var:<28s} {'—':>8s} {'—':>10s}   No missingness")

print("\n  *** = p < 0.05 (missingness is differential by stroke status)")
print("  Variables with MAR/MNAR: imputation justified over complete-case analysis")
print("  Variables with MCAR: imputation still valid, complete-case also unbiased")

# ==============================================================================
# 4. TIERED EXCLUSION (DATA-DRIVEN)
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 4: Tiered exclusion decisions (data-driven)")
print("=" * 70)

# --- Tier D: Exclude from primary model (>50% missing) ---
tier_d_vars = miss_report[miss_report["tier"].str.startswith("D")]["variable"].tolist()

print("\n--- TIER D: EXCLUDED from primary model (>50% missing) ---")
if tier_d_vars:
    for var in tier_d_vars:
        pct = miss_report.loc[miss_report["variable"] == var, "pct_miss"].values[0]
        reason = f"{pct:.1f}% missing"
        if var == "triglycerides":
            reason += "; also requires fasting subsample weights (wtsaf2yr) " \
                      "which conflict with MEC weights used for all other variables"
        elif var == "sleep_hrs":
            reason += "; structurally absent in early NHANES cycles (2003-2004)"
        print(f"  EXCLUDE: {var:<28s} → {reason}")
else:
    print("  (none)")

# --- Tier C: Flag for sensitivity analysis (30-50% missing) ---
tier_c_vars = miss_report[miss_report["tier"].str.startswith("C")]["variable"].tolist()

print("\n--- TIER C: FLAGGED for sensitivity analysis (30-50% missing) ---")
if tier_c_vars:
    for var in tier_c_vars:
        pct = miss_report.loc[miss_report["variable"] == var, "pct_miss"].values[0]
        print(f"  FLAG:    {var:<28s} → {pct:.1f}% missing")
    print("  Action: Run primary model WITH and WITHOUT these variables;")
    print("          report both results. If AUROC differs by <0.01, robust.")
else:
    print("  (none)")

# --- Tier B: Impute + verify diagnostics (SD ratio, mean shift, KS statistic, correlation preservation) ---
tier_b_vars = miss_report[miss_report["tier"].str.startswith("B")]["variable"].tolist()

print("\n--- TIER B: IMPUTE + report diagnostics (5-30% missing) ---")
if tier_b_vars:
    for var in tier_b_vars:
        pct = miss_report.loc[miss_report["variable"] == var, "pct_miss"].values[0]
        print(f"  IMPUTE:  {var:<28s} → {pct:.1f}% missing")
else:
    print("  (none)")

# --- Tier A: Impute freely (<5% missing) ---
tier_a_vars = miss_report[miss_report["tier"].str.startswith("A")]["variable"].tolist()

print("\n--- TIER A: IMPUTE freely (<5% missing) ---")
if tier_a_vars:
    for var in tier_a_vars:
        pct = miss_report.loc[miss_report["variable"] == var, "pct_miss"].values[0]
        print(f"  IMPUTE:  {var:<28s} → {pct:.1f}% missing")
else:
    print("  (none)")

# --- Apply exclusions ---
vars_to_impute = [v for v in all_predictors if v not in tier_d_vars]

print(f"\n--- SUMMARY ---")
print(f"  Total candidate predictors:  {len(all_predictors)}")
print(f"  Excluded (Tier D, >50%):     {len(tier_d_vars)} → {tier_d_vars}")
print(f"  Flagged (Tier C, 30-50%):    {len(tier_c_vars)} → {tier_c_vars}")
print(f"  Entering imputation:         {len(vars_to_impute)}")

# Update predictor type lists to reflect exclusions
continuous_predictors = [c for c in continuous_predictors if c not in tier_d_vars]
categorical_predictors = [c for c in categorical_predictors if c not in tier_d_vars]
binary_predictors = [c for c in binary_predictors if c not in tier_d_vars]

# ==============================================================================
# 5. PREPARE DATA FOR HYPERIMPUTE
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 5: Prepare data for HyperImpute")
print("=" * 70)

df_impute = df[vars_to_impute].copy()

# Encode categoricals (HyperImpute requires a fully numeric matrix)
# The nhanesdata R package + CSV export can produce Arrow-backed StringDtype
# columns which cause type conflicts in HyperImpute's internal LabelEncoder.
# We explicitly convert to plain Python strings first, then to integer codes.
cat_mappings = {}
for col in categorical_predictors:
    if col in df_impute.columns:
        # Force to plain object dtype (plain Python strings), not Arrow StringDtype
        df_impute[col] = df_impute[col].astype(object)
        categories = df_impute[col].dropna().unique().tolist()
        cat_mappings[col] = categories
        df_impute[col] = pd.Categorical(df_impute[col], categories=categories).codes
        # pd.Categorical.codes returns -1 for NaN; convert to actual NaN
        df_impute[col] = df_impute[col].replace(-1, np.nan).astype(float)
        print(f"  Encoded {col}: {categories}")

# Ensure binary variables are numeric float
for col in binary_predictors:
    if col in df_impute.columns:
        df_impute[col] = pd.to_numeric(df_impute[col], errors="coerce").astype(float)

# Ensure ALL continuous variables are float64 (guard against int/object leaks)
for col in continuous_predictors:
    if col in df_impute.columns:
        df_impute[col] = pd.to_numeric(df_impute[col], errors="coerce").astype(float)

# Final dtype verification: everything must be float64
non_numeric = [c for c in df_impute.columns if not np.issubdtype(df_impute[c].dtype, np.floating)]
if non_numeric:
    print(f"\n⚠ WARNING: Non-numeric columns detected after encoding: {non_numeric}")
    print("  Forcing to float64...")
    for col in non_numeric:
        df_impute[col] = pd.to_numeric(df_impute[col], errors="coerce").astype(float)

print(f"\nAll columns dtype: {df_impute.dtypes.unique().tolist()}")
print(f"Prepared matrix: {df_impute.shape[0]:,} rows x {df_impute.shape[1]} columns")
print(f"Total missing values: {df_impute.isna().sum().sum():,}")
print(f"Overall missingness rate: {df_impute.isna().mean().mean()*100:.1f}%")

# ==============================================================================
# 6. RUN HYPERIMPUTE
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 6: Run HyperImpute")
print("=" * 70)

from hyperimpute.plugins.imputers import Imputers

# HyperImpute: column-wise model selection from a library of learners.
# For each column, it evaluates logistic/linear regression, random forest,
# and CatBoost via cross-validation, selecting the best imputer per feature.
# This adapts to the structure of each variable's missingness pattern.
#
# NOTE: CatBoost is used instead of XGBoost because macOS Apple Silicon
# loads multiple conflicting copies of libomp (from Homebrew, sklearn, and
# PyTorch), causing segfaults in XGBoost's OpenMP threading. CatBoost is
# self-contained and avoids this issue. Both are gradient boosting methods
# with comparable imputation quality on tabular data.

print("\nInitializing HyperImpute with column-wise model selection...")
print("This may take 10-30 minutes depending on dataset size.\n")

# Three diverse learner classes for column-wise model selection:
#   - Linear models (logistic_regression / linear_regression)
#   - Tree ensemble (random_forest / random_forest_regressor)
#   - Gradient boosting (catboost / catboost_regressor)
# CatBoost used instead of XGBoost to avoid OpenMP multi-library crash on macOS.
# sklearn pinned to ==1.5.2 to avoid deprecated `multi_class` parameter bug.
imputer = Imputers().get(
    "hyperimpute",
    optimizer="hyperband",
    classifier_seed=["logistic_regression", "random_forest", "catboost"],
    regression_seed=["linear_regression", "random_forest_regressor", "catboost_regressor"],
    random_state=RANDOM_SEED,
)

print("Fitting HyperImpute...")
df_imputed_values = imputer.fit_transform(df_impute.copy())
df_imputed = pd.DataFrame(df_imputed_values, columns=df_impute.columns, index=df_impute.index)

print(f"\nImputation complete.")
print(f"Remaining missing values: {df_imputed.isna().sum().sum()}")

# ==============================================================================
# 7. POST-IMPUTATION PROCESSING
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 7: Post-imputation processing")
print("=" * 70)

# --- Decode categorical variables back to labels ---
for col, categories in cat_mappings.items():
    if col in df_imputed.columns:
        df_imputed[col] = df_imputed[col].round().clip(0, len(categories) - 1).astype(int)
        df_imputed[col] = df_imputed[col].map(dict(enumerate(categories)))
        print(f"  Decoded {col}: {df_imputed[col].value_counts().to_dict()}")

# --- Enforce plausible clinical ranges ---
# Imputation can produce out-of-range values; clip to biologically plausible bounds.
# BP zeroes were already converted to NA in the R extraction script.
range_constraints = {
    "age": (20, 80), "mean_sbp": (70, 250), "mean_dbp": (40, 150),
    "bmxwaist": (50, 200), "bmxbmi": (12, 80),
    "alcohol_drinks_per_day": (0, 20),
    "hba1c": (3, 18), "total_chol": (80, 500), "hdl_c": (10, 150),
    "triglycerides": (20, 2000), "creatinine": (0.1, 15), "wbc": (1, 50),
    "pir": (0, 5), "sleep_hrs": (1, 18),
}

for col, (lo, hi) in range_constraints.items():
    if col in df_imputed.columns:
        n_clipped = ((df_imputed[col] < lo) | (df_imputed[col] > hi)).sum()
        if n_clipped > 0:
            print(f"  Clipped {n_clipped} out-of-range values in {col} to [{lo}, {hi}]")
        df_imputed[col] = df_imputed[col].clip(lo, hi)

# --- Round binary variables to 0/1 ---
for col in binary_predictors:
    if col in df_imputed.columns:
        df_imputed[col] = df_imputed[col].round().clip(0, 1).astype(int)

# ==============================================================================
# 8. IMPUTATION QUALITY DIAGNOSTICS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 8: Imputation quality diagnostics")
print("=" * 70)

# --- Distributional comparison: pre vs. post imputation ---

print("\n--- Continuous variables: pre vs. post imputation ---")
print(f"  {'Variable':<25s} {'Pre-Mean':>10s} {'Post-Mean':>10s} {'Pre-SD':>10s} "
      f"{'Post-SD':>10s} {'KS-p':>10s} {'Flag':>6s}")
print("  " + "-" * 85)

for col in continuous_predictors:
    if col not in df_imputed.columns:
        continue
    pre = df_impute[col].dropna()
    post = df_imputed[col]
    if len(pre) < 10:
        continue

    pre_mean, post_mean = pre.mean(), post.mean()
    pre_sd, post_sd = pre.std(), post.std()
    ks_stat, ks_p = stats.ks_2samp(pre.values, post.values)

    # Flag: >0.1 SD shift in mean or >20% change in variance
    mean_shift = abs(post_mean - pre_mean) / (pre_sd + 1e-8)
    sd_ratio = post_sd / (pre_sd + 1e-8)
    flag = ""
    if mean_shift > 0.1:
        flag = "SHIFT"
    if sd_ratio < 0.8 or sd_ratio > 1.2:
        flag = "VAR" if not flag else flag + "+VAR"

    print(f"  {col:<25s} {pre_mean:10.2f} {post_mean:10.2f} {pre_sd:10.2f} "
          f"{post_sd:10.2f} {ks_p:10.4f} {flag:>6s}")

print("\n--- Binary variables: pre vs. post imputation ---")
print(f"  {'Variable':<25s} {'Pre-Prev':>10s} {'Post-Prev':>10s} {'Diff':>10s}")
print("  " + "-" * 60)

for col in binary_predictors:
    if col not in df_imputed.columns:
        continue
    pre_prev = df_impute[col].dropna().mean() * 100
    post_prev = df_imputed[col].mean() * 100
    diff = post_prev - pre_prev
    flag = " ***" if abs(diff) > 2 else ""
    print(f"  {col:<25s} {pre_prev:9.1f}% {post_prev:9.1f}% {diff:+9.1f}%{flag}")

# --- Correlation structure comparison: pre vs. post imputation ---
# Checks whether imputation artificially inflated correlations.
# Important for knockoff filter validity downstream.



print("\n--- Correlation structure: pre vs. post imputation ---")

corr_vars = [c for c in continuous_predictors + binary_predictors
             if c in df_imputed.columns]

# Pre: pairwise complete Spearman correlations
pre_corr = df_impute[corr_vars].corr(method="spearman")
# Post: full Spearman correlations (no missing data)
post_corr = df_imputed[corr_vars].corr(method="spearman")
# Difference
corr_diff = post_corr - pre_corr

upper_tri = corr_diff.where(np.triu(np.ones(corr_diff.shape, dtype=bool), k=1))
diffs = upper_tri.stack().values

print(f"  Pairwise correlation differences (post - pre):")
print(f"    Mean absolute change:   {np.abs(diffs).mean():.4f}")
print(f"    Max absolute change:    {np.abs(diffs).max():.4f}")
print(f"    Median change:          {np.median(diffs):+.4f}")
print(f"    Pairs with |diff|>0.05: {(np.abs(diffs) > 0.05).sum()} / {len(diffs)}")
print(f"    Pairs with |diff|>0.10: {(np.abs(diffs) > 0.10).sum()} / {len(diffs)}")
print(f"    Pairs with |diff|>0.15: {(np.abs(diffs) > 0.15).sum()} / {len(diffs)}")

if np.abs(diffs).max() > 0.15:
    print("  ⚠ WARNING: Some correlations shifted >0.15 after imputation.")
    print("    Review the difference heatmap for affected variable pairs.")
else:
    print("  ✓ Correlation structure well-preserved after imputation.")

# --- Plot: three-panel correlation heatmap ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

short_labels = {
    "age": "Age", "mean_sbp": "SBP", "mean_dbp": "DBP",
    "bmxwaist": "Waist", "bmxbmi": "BMI",
    "alcohol_drinks_per_day": "Alcohol",
    "hba1c": "HbA1c", "total_chol": "TC", "hdl_c": "HDL",
    "triglycerides": "TG", "creatinine": "Cr", "wbc": "WBC",
    "pir": "PIR", "sleep_hrs": "Sleep",
    "htn_med": "HTN Med", "phys_active": "PA",
    "diabetes_sr": "DM", "chd_sr": "CHD",
}
labels = [short_labels.get(v, v) for v in corr_vars]

mask = np.triu(np.ones_like(pre_corr, dtype=bool), k=0)




sns.heatmap(pre_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels, ax=axes[0],
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
axes[0].set_title("Pre-Imputation\n(Pairwise Complete)", fontsize=12, fontweight="bold")
axes[0].text(-0.10, 1.04, "a", transform=axes[0].transAxes,
             fontsize=16, fontweight="bold", va="top", ha="left")

sns.heatmap(post_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels, ax=axes[1],
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
axes[1].set_title("Post-Imputation\n(Complete Data)", fontsize=12, fontweight="bold")
axes[1].text(-0.10, 1.04, "b", transform=axes[1].transAxes,
             fontsize=16, fontweight="bold", va="top", ha="left")

sns.heatmap(corr_diff, mask=mask, annot=True, fmt="+.2f", cmap="PiYG_r",
            center=0, vmin=-0.2, vmax=0.2, square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels, ax=axes[2],
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
axes[2].set_title("Difference\n(Post − Pre)", fontsize=12, fontweight="bold")
axes[2].text(-0.10, 1.04, "c", transform=axes[2].transAxes,
             fontsize=16, fontweight="bold", va="top", ha="left")

plt.suptitle("Correlation Structure: Pre vs. Post Imputation (Spearman)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(artifacts_dir / "fig_correlation_pre_post_imputation.png", dpi=300, bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_correlation_pre_post_imputation.pdf", dpi=300, bbox_inches="tight")
plt.savefig(artifacts_dir / "fig_correlation_pre_post_imputation.svg", bbox_inches="tight")
plt.close()

print("\nSaved: fig_correlation_pre_post_imputation.png / .pdf / .svg")

# ==============================================================================
# 9. REASSEMBLE FULL DATASET
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 9: Reassemble and export")
print("=" * 70)

df_final = df[id_vars + design_vars + [outcome_var]].copy()

# Add imputed predictors
for col in vars_to_impute:
    df_final[col] = df_imputed[col].values

# Add back Tier D excluded variables (unimputed, with NAs) for sensitivity analyses
for col in tier_d_vars:
    if col in df.columns and col not in df_final.columns:
        df_final[col] = df[col].values
        print(f"  Added back (unimputed): {col} ({df[col].isna().mean()*100:.1f}% missing)")

print(f"\nFinal dataset: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")
print(f"Remaining NAs in imputed predictors: {df_final[vars_to_impute].isna().sum().sum()}")

# ==============================================================================
# 10. EXPORT
# ==============================================================================

# Imputed dataset
df_final.to_parquet(data_dir / "nhanes_stroke_imputed_2003_2023.parquet", index=False)
df_final.to_csv(data_dir / "nhanes_stroke_imputed_2003_2023.csv", index=False)
print(f"\nExported: nhanes_stroke_imputed_2003_2023.parquet / .csv")

# Missingness report with tier decisions
miss_report.to_csv(artifacts_dir / "missingness_report.csv", index=False)
print("Exported: missingness_report.csv")

# Mechanism diagnostics
if mechanism_results:
    pd.DataFrame(mechanism_results).to_csv(artifacts_dir / "missingness_mechanism_tests.csv", index=False)
    print("Exported: missingness_mechanism_tests.csv")

# Correlation difference matrix
corr_diff.to_csv(artifacts_dir / "correlation_diff_pre_post_imputation.csv")
print("Exported: correlation_diff_pre_post_imputation.csv")

# Imputation quality diagnostics
diagnostics = []
for col in continuous_predictors:
    if col not in df_imputed.columns:
        continue
    pre = df_impute[col].dropna()
    post = df_imputed[col]
    if len(pre) < 10:
        continue
    ks_stat, ks_p = stats.ks_2samp(pre.values, post.values)
    diagnostics.append({
        "variable": col, "type": "continuous",
        "pre_mean": pre.mean(), "post_mean": post.mean(),
        "pre_sd": pre.std(), "post_sd": post.std(),
        "pct_missing": df_impute[col].isna().mean() * 100,
        "ks_statistic": ks_stat, "ks_pvalue": ks_p,
        "mean_shift_sd": abs(post.mean() - pre.mean()) / (pre.std() + 1e-8),
        "sd_ratio": post.std() / (pre.std() + 1e-8),
    })

for col in binary_predictors:
    if col not in df_imputed.columns:
        continue
    pre = df_impute[col].dropna()
    post = df_imputed[col]
    diagnostics.append({
        "variable": col, "type": "binary",
        "pre_mean": pre.mean(), "post_mean": post.mean(),
        "pre_sd": pre.std(), "post_sd": post.std(),
        "pct_missing": df_impute[col].isna().mean() * 100,
        "ks_statistic": np.nan, "ks_pvalue": np.nan,
        "mean_shift_sd": abs(post.mean() - pre.mean()) / (pre.std() + 1e-8),
        "sd_ratio": post.std() / (pre.std() + 1e-8),
    })

pd.DataFrame(diagnostics).to_csv(artifacts_dir / "imputation_diagnostics.csv", index=False)
print("Exported: imputation_diagnostics.csv")

print("\n" + "=" * 70)
print("IMPUTATION COMPLETE")
print("=" * 70)
print(f"""
Pipeline summary:
  Candidate predictors:     {len(all_predictors)}
  Excluded (Tier D, >50%):  {len(tier_d_vars)} → {tier_d_vars}
  Flagged (Tier C, 30-50%): {len(tier_c_vars)} → {tier_c_vars}
  Imputed:                  {len(vars_to_impute)}

Outputs:
  nhanes_stroke_imputed_2003_2023.csv / .parquet  (imputed dataset)
  missingness_report.csv                          (tier assignments)
  missingness_mechanism_tests.csv                 (chi-sq diagnostics)
  imputation_diagnostics.csv                      (distributional checks)
  correlation_diff_pre_post_imputation.csv        (correlation changes)
  fig_correlation_pre_post_imputation.png / .pdf  (correlation heatmap)

Next steps:
  1. Review imputation_diagnostics.csv for quality flags
  2. Run knockoff filter feature selection on imputed data
  3. Train AutoGluon with repeated stratified CV
  4. SHAP interpretation + misclassification sensitivity analysis
  5. For Tier C vars: re-run model WITHOUT them as sensitivity check
""")