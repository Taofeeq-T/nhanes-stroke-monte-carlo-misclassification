"""
================================================================================
NHANES Stroke Prediction: Sensitivity analysis for the latent-outcome
reconstruction
================================================================================
The primary probabilistic bias analysis (autogluon_modeling.py, Step 7)
estimates q(s) = P(Y*=1 | s) by isotonic regression and inverts the
misclassification model to obtain pi(s) = P(T=1 | s). Where q(s) < 1 - Sp the
inverted value is negative and is set to zero, so those participants receive
a latent outcome T = 0 in that iteration.

This script repeats the analysis with a reconstruction in which pi(s) is
estimated directly by maximum likelihood under the stated misclassification
model (Magder & Hughes, 1997, Am J Epidemiol 146(2):195-203):

    pi(s) = expit(a + b * logit(s))
    P(Y*=1 | s) = Se * pi(s) + (1 - Sp) * (1 - pi(s))

(a, b) are refitted in every iteration for the sampled (Se, Sp). Because
q(s) = Se*pi + (1-Sp)*(1-pi) lies in (1 - Sp, Se) by construction, no value
is clipped and the posterior P(T=1 | Y*, s) is always well defined.

Both reconstructions are run inside the same loop with the same random
stream as Step 7 (seed 42, identical Se/Sp draws and identical uniforms for
sampling T), so the primary results are reproduced exactly and every
iteration yields a paired comparison.

Inputs : artifacts/holdout_predictions.csv  (written by autogluon_modeling.py)
Outputs: artifacts/pba_mh_iterations.csv         per-iteration results, both methods
         artifacts/pba_mh_summary.csv            median and 95% simulation interval
         artifacts/pba_mh_manuscript_numbers.json key numbers for the manuscript
         artifacts/monte_carlo_misclassification.csv  primary iterations, rewritten
             with sampled parameters as se_selfreport / sp_selfreport (M5); the
             script first checks that all performance columns match the released file
         logs/bias_sensitivity_mh.log             full console log


Usage  : python scripts/bias_sensitivity_mh.py
================================================================================
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ------------------------------------------------------------------------------
# Paths and logging (console output is mirrored to logs/bias_sensitivity_mh.log)
# ------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
artifacts_dir = project_root / "artifacts"
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, data):
        self.console.write(data)
        self.logfile.write(data)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


_log_file = open(log_dir / "bias_sensitivity_mh.log", "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ------------------------------------------------------------------------------
# Configuration: identical to autogluon_modeling.py
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
N_MC = 1000
SENS_RANGE = (0.36, 0.98)   # https://doi.org/10.1371/journal.pone.0137538
SPEC_RANGE = (0.96, 0.996)  # https://doi.org/10.1371/journal.pone.0137538
EPS = 1e-6

print("=" * 72)
print("Sensitivity analysis: Magder-Hughes reconstruction of the latent outcome")
print("=" * 72)
print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Iterations: {N_MC}   seed: {RANDOM_SEED}")
print(f"Se ~ U{SENS_RANGE}   Sp ~ U{SPEC_RANGE}")

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
h = pd.read_csv(artifacts_dir / "holdout_predictions.csv")
y_obs = h["y_true"].to_numpy().astype(int)
s = h["y_pred_proba"].to_numpy()
thr = float(h["optimal_threshold"].iloc[0])
yhat = (s >= thr).astype(int)
x = logit(np.clip(s, EPS, 1 - EPS))  # covariate for the MH model

print(f"\nHoldout n = {len(y_obs)}   observed positives = {y_obs.sum()} "
      f"(prevalence {y_obs.mean():.4f})   threshold = {thr:.4f}")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def threshold_metrics(y, yh):
    tp = np.sum((yh == 1) & (y == 1)); fn = np.sum((yh == 0) & (y == 1))
    tn = np.sum((yh == 0) & (y == 0)); fp = np.sum((yh == 1) & (y == 0))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    return {"balanced_accuracy": (sens + spec) / 2,
            "f1": 2 * ppv * sens / max(ppv + sens, 1e-12),
            "recall": sens, "model_specificity": spec, "precision": ppv,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def all_metrics(T):
    return {"auroc": roc_auc_score(T, s), "auprc": average_precision_score(T, s),
            "brier_score": brier_score_loss(T, s), **threshold_metrics(T, yhat)}


def sample_T(pi, q, se, u):
    """Draw latent T from P(T=1 | Y*, s) using pre-drawn uniforms u."""
    p1 = np.clip(se * pi / q, 0, 1)
    p0 = np.clip((1 - se) * pi / (1 - q), 0, 1)
    return np.where(y_obs == 1, u < p1, u < p0).astype(int)


def mh_negloglik(theta, se, sp):
    """Negative log-likelihood and gradient for the Magder-Hughes model."""
    a, b = theta
    pi = expit(a + b * x)
    q = (1 - sp) + (se + sp - 1) * pi
    nll = -np.sum(y_obs * np.log(q) + (1 - y_obs) * np.log1p(-q))
    dq = (se + sp - 1) * pi * (1 - pi)
    w = -(y_obs / q - (1 - y_obs) / (1 - q)) * dq
    return nll, np.array([w.sum(), (w * x).sum()])


def fit_mh(se, sp, init):
    res = minimize(mh_negloglik, init, args=(se, sp), jac=True, method="L-BFGS-B")
    return res.x, res.success


# ------------------------------------------------------------------------------
# Primary reconstruction input: isotonic q(s), fitted once (as in Step 7)
# ------------------------------------------------------------------------------
q_iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip") \
    .fit(s, y_obs).predict(s)

# Starting values for the MH fit: ordinary logistic recalibration of the score
theta0, _ = fit_mh(1.0 - 1e-6, 1.0 - 1e-6, np.array([0.0, 1.0]))
print(f"Logistic recalibration of score (no misclassification): "
      f"a = {theta0[0]:.3f}, b = {theta0[1]:.3f}")

naive = {"auroc": roc_auc_score(y_obs, s), "auprc": average_precision_score(y_obs, s),
         "brier_score": brier_score_loss(y_obs, s), **threshold_metrics(y_obs, yhat)}

# ------------------------------------------------------------------------------
# Monte Carlo loop (same random stream as autogluon_modeling.py Step 7)
# ------------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
rows, primary_rows, n_fail = [], [], 0
t0 = time.time()
theta_prev = theta0.copy()

for i in range(N_MC):
    se = rng.uniform(*SENS_RANGE)
    sp = rng.uniform(*SPEC_RANGE)
    u = rng.random(len(y_obs))

    # (1) Primary: isotonic q(s), inverted and clipped
    pi_iso = np.clip((q_iso - (1 - sp)) / (se + sp - 1), 0, 1)
    T_iso = sample_T(pi_iso, q_iso, se, u)

    # (2) Sensitivity: Magder-Hughes maximum likelihood, no clipping
    theta, ok = fit_mh(se, sp, theta_prev)
    if ok:
        theta_prev = theta
    else:
        n_fail += 1
    pi_mh = expit(theta[0] + theta[1] * x)
    q_mh = (1 - sp) + (se + sp - 1) * pi_mh
    T_mh = sample_T(pi_mh, q_mh, se, u)

    # Same validity rule as Step 7 (applied to the primary reconstruction)
    if T_iso.sum() < 5 or T_iso.sum() == len(T_iso):
        continue

    rec = {"iteration": i + 1, "se_selfreport": se, "sp_selfreport": sp,
           "iso_frac_pi_clipped_zero": np.mean(pi_iso == 0),
           "iso_implied_true_prevalence": pi_iso.mean(),
           "mh_a": theta[0], "mh_b": theta[1], "mh_converged": ok,
           "mh_min_pi": pi_mh.min(),
           "mh_implied_true_prevalence": pi_mh.mean()}
    m_iso = all_metrics(T_iso)
    rec.update({f"iso_{k}": v for k, v in m_iso.items()})
    rec.update({f"mh_{k}": v for k, v in all_metrics(T_mh).items()})
    rows.append(rec)

    # Primary iteration record in the format of Step 7, with the sampled
    # self-report parameters stored under distinct names (Reviewer 2, M5)
    primary_rows.append({
        "iteration": i + 1, "se_selfreport": se, "sp_selfreport": sp,
        "implied_true_prevalence": pi_iso.mean(),
        "frac_scores_pi_clipped_zero": np.mean(pi_iso == 0),
        "auroc": m_iso["auroc"], "auprc": m_iso["auprc"], "brier_score": m_iso["brier_score"],
        "balanced_accuracy": m_iso["balanced_accuracy"], "f1": m_iso["f1"],
        "recall": m_iso["recall"], "specificity": m_iso["model_specificity"],
        "precision": m_iso["precision"],
        "tp": m_iso["tp"], "fn": m_iso["fn"], "tn": m_iso["tn"], "fp": m_iso["fp"]})

mc = pd.DataFrame(rows)
print(f"\nValid iterations: {len(mc)} / {N_MC}   MH non-converged fits: {n_fail}   "
      f"elapsed {time.time() - t0:.1f}s")

# ------------------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------------------
def q3(v):
    return v.median(), v.quantile(0.025), v.quantile(0.975)


metrics = ["auroc", "auprc", "brier_score", "balanced_accuracy", "f1",
           "recall", "model_specificity", "precision"]
summary = []
print(f"\n{'Metric':<20s}{'Naive':>8s}   {'Primary (isotonic, clipped)':>30s}   "
      f"{'Magder-Hughes':>26s}")
print("-" * 92)
for m in metrics:
    a = q3(mc[f"iso_{m}"]); b = q3(mc[f"mh_{m}"])
    summary.append({"metric": m, "naive": naive[m],
                    "primary_median": a[0], "primary_si_lower": a[1], "primary_si_upper": a[2],
                    "mh_median": b[0], "mh_si_lower": b[1], "mh_si_upper": b[2]})
    print(f"{m:<20s}{naive[m]:8.3f}   {a[0]:8.3f} [{a[1]:.3f}, {a[2]:.3f}]       "
          f"{b[0]:8.3f} [{b[1]:.3f}, {b[2]:.3f}]")

for name, col_iso, col_mh in [("implied_true_prevalence", "iso_implied_true_prevalence",
                               "mh_implied_true_prevalence")]:
    a = q3(mc[col_iso]); b = q3(mc[col_mh])
    summary.append({"metric": name, "naive": y_obs.mean(),
                    "primary_median": a[0], "primary_si_lower": a[1], "primary_si_upper": a[2],
                    "mh_median": b[0], "mh_si_lower": b[1], "mh_si_upper": b[2]})
    print(f"{name:<20s}{y_obs.mean():8.3f}   {a[0]:8.3f} [{a[1]:.3f}, {a[2]:.3f}]       "
          f"{b[0]:8.3f} [{b[1]:.3f}, {b[2]:.3f}]")

clip = q3(mc["iso_frac_pi_clipped_zero"])
d_auc = mc["mh_auroc"] - mc["iso_auroc"]
dq = q3(d_auc)
print(f"\nPrimary reconstruction, share of holdout participants with pi(s) clipped to 0:")
print(f"  median {clip[0]:.3f}, 95% SI [{clip[1]:.3f}, {clip[2]:.3f}], "
      f"range {mc['iso_frac_pi_clipped_zero'].min():.3f} to "
      f"{mc['iso_frac_pi_clipped_zero'].max():.3f}")
print(f"  iterations with any clipping: {(mc['iso_frac_pi_clipped_zero'] > 0).mean():.3f}")
print(f"MH reconstruction, smallest pi(s) across iterations: {mc['mh_min_pi'].min():.2e} "
      f"(no clipping by construction)")
print(f"Paired AUROC difference (MH minus primary): median {dq[0]:+.3f}, "
      f"95% SI [{dq[1]:+.3f}, {dq[2]:+.3f}]")

# ------------------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------------------
mc.to_csv(artifacts_dir / "pba_mh_iterations.csv", index=False)

# Regenerate the primary iteration file (M5). Before overwriting, confirm that
# every performance column matches the released file from autogluon_modeling.py.
prim = pd.DataFrame(primary_rows)
mc_path = artifacts_dir / "monte_carlo_misclassification.csv"
if mc_path.exists():
    old = pd.read_csv(mc_path)
    check_cols = ["iteration", "implied_true_prevalence", "frac_scores_pi_clipped_zero",
                  "auroc", "auprc", "brier_score", "balanced_accuracy", "f1", "recall",
                  "specificity", "precision", "tp", "fn", "tn", "fp"]
    same_shape = len(old) == len(prim)
    max_diff = (np.abs(old[check_cols].to_numpy(float) - prim[check_cols].to_numpy(float)).max()
                if same_shape else np.inf)
    se_diff = (np.abs(old["sensitivity"].to_numpy() - prim["se_selfreport"].to_numpy()).max()
               if same_shape and "sensitivity" in old else np.nan)
    print(f"\nCheck against released {mc_path.name}: rows {len(old)} vs {len(prim)}, "
          f"max |diff| in performance columns {max_diff:.2e}, "
          f"max |diff| sampled Se {se_diff:.2e}")
    if max_diff > 1e-9:
        raise SystemExit("Mismatch with released iteration file; not overwriting.")
    print(f"Released 'specificity' column (model specificity, overwrote sampled Sp): "
          f"range {old['specificity'].min():.3f} to {old['specificity'].max():.3f}")
prim.to_csv(mc_path, index=False)
print(f"Sampled Sp now stored as 'sp_selfreport': range "
      f"{prim['sp_selfreport'].min():.3f} to {prim['sp_selfreport'].max():.3f}")
print(f"Rewrote {mc_path.name} with se_selfreport / sp_selfreport (M5)")
pd.DataFrame(summary).to_csv(artifacts_dir / "pba_mh_summary.csv", index=False)

numbers = {
    "n_holdout": int(len(y_obs)), "n_valid_iterations": int(len(mc)),
    "mh_nonconverged": int(n_fail), "threshold": thr,
    "clip_share_median": clip[0], "clip_share_si": [clip[1], clip[2]],
    "clip_share_range": [float(mc["iso_frac_pi_clipped_zero"].min()),
                         float(mc["iso_frac_pi_clipped_zero"].max())],
    "auroc_naive": naive["auroc"],
    "auroc_primary": list(q3(mc["iso_auroc"])),
    "auroc_mh": list(q3(mc["mh_auroc"])),
    "auroc_diff_mh_minus_primary": list(dq),
    "auprc_primary": list(q3(mc["iso_auprc"])), "auprc_mh": list(q3(mc["mh_auprc"])),
    "prevalence_primary": list(q3(mc["iso_implied_true_prevalence"])),
    "prevalence_mh": list(q3(mc["mh_implied_true_prevalence"])),
}
with open(artifacts_dir / "pba_mh_manuscript_numbers.json", "w") as f:
    json.dump(numbers, f, indent=2, default=float)

print("\nExported: pba_mh_iterations.csv, pba_mh_summary.csv, pba_mh_manuscript_numbers.json")
print(f"Log: {log_dir / 'bias_sensitivity_mh.log'}")
print(f"Run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")