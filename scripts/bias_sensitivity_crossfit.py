"""
================================================================================
NHANES Stroke Prediction: cross-fitted q(s) for the latent-outcome
reconstruction
================================================================================
In the primary probabilistic bias analysis (autogluon_modeling.py, Step 7)
q(s) = P(Y*=1 | s) is fitted on the same holdout labels that are then used to
reconstruct the latent outcome, and the simulation interval does not reflect
uncertainty in estimating q(s).

Here q(s) is cross-fitted within the holdout. In each Monte Carlo iteration
the holdout is split into K = 5 stratified folds; for the participants in each
fold, q(s) (or pi(s)) is estimated only from a bootstrap resample of the other
four folds. No participant's own label is used to estimate the risk function
that reconstructs their latent outcome, and the fold split and bootstrap are
redrawn in every iteration, so the simulation interval also carries the
estimation uncertainty in q(s).

Out-of-fold training predictions from the saved AutoGluon predictor were
considered first (bias_sensitivity_oof.py) but were not on the same scale as
the holdout scores (AUROC 0.737 vs 0.790; 99th percentile 0.845 vs 0.495), so
a q(s) learned on them does not transfer to the holdout scores.

Reconstructions, all paired on the same Se, Sp and uniforms (seed 42):
  primary  : isotonic q(s) fitted on all holdout labels (Step 7, reference)
  cfiso    : cross-fitted isotonic q(s), negative pi(s) set to 0
  cfmh     : cross-fitted Magder-Hughes, pi(s) = expit(a + b*logit s)
Fold splits and bootstrap draws use a separate stream (seed 43).

Inputs : artifacts/holdout_predictions.csv
Outputs: artifacts/pba_crossfit_iterations.csv
         artifacts/pba_crossfit_summary.csv   (sensitivity version of Table 2, bias column)
         artifacts/pba_crossfit_manuscript_numbers.json
         logs/bias_sensitivity_crossfit.log
Usage  : python scripts/bias_sensitivity_crossfit.py
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
from sklearn.model_selection import StratifiedKFold

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


_log_file = open(log_dir / "bias_sensitivity_crossfit.log", "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)

RANDOM_SEED = 42
CF_SEED = 43
N_MC = 1000
K_FOLDS = 5
SENS_RANGE = (0.36, 0.98)   # https://doi.org/10.1371/journal.pone.0137538
SPEC_RANGE = (0.96, 0.996)  # https://doi.org/10.1371/journal.pone.0137538
EPS = 1e-6
Q_BOUNDS = (1e-4, 1 - 1e-4)

print("=" * 72)
print("R2 sensitivity analysis: cross-fitted q(s) within the holdout")
print("=" * 72)
print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Iterations: {N_MC}   folds: {K_FOLDS}   seed: {RANDOM_SEED} (Se, Sp, T)   "
      f"fold/bootstrap seed: {CF_SEED}")
print(f"Se ~ U{SENS_RANGE}   Sp ~ U{SPEC_RANGE}")

h = pd.read_csv(artifacts_dir / "holdout_predictions.csv")
y_obs = h["y_true"].to_numpy().astype(int)
s = h["y_pred_proba"].to_numpy()
thr = float(h["optimal_threshold"].iloc[0])
yhat = (s >= thr).astype(int)
x = logit(np.clip(s, EPS, 1 - EPS))
n = len(y_obs)
print(f"\nHoldout n = {n}   observed positives = {y_obs.sum()} (prevalence {y_obs.mean():.4f})   "
      f"threshold = {thr:.4f}")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def threshold_metrics(y, yh):
    tp = np.sum((yh == 1) & (y == 1)); fn = np.sum((yh == 0) & (y == 1))
    tn = np.sum((yh == 0) & (y == 0)); fp = np.sum((yh == 1) & (y == 0))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    return {"balanced_accuracy": (sens + spec) / 2,
            "f1": 2 * ppv * sens / max(ppv + sens, 1e-12),
            "recall": sens, "model_specificity": spec, "precision": ppv}


def all_metrics(T):
    return {"auroc": roc_auc_score(T, s), "auprc": average_precision_score(T, s),
            "brier_score": brier_score_loss(T, s), **threshold_metrics(T, yhat)}


def sample_T(pi, q, se, u):
    p1 = np.clip(se * pi / q, 0, 1)
    p0 = np.clip((1 - se) * pi / (1 - q), 0, 1)
    return np.where(y_obs == 1, u < p1, u < p0).astype(int)


def mh_negloglik(theta, xx, yy, se, sp):
    a, b = theta
    pi = expit(a + b * xx)
    q = (1 - sp) + (se + sp - 1) * pi
    nll = -np.sum(yy * np.log(q) + (1 - yy) * np.log1p(-q))
    w = -(yy / q - (1 - yy) / (1 - q)) * (se + sp - 1) * pi * (1 - pi)
    return nll, np.array([w.sum(), (w * xx).sum()])


def fit_mh(xx, yy, se, sp, init):
    res = minimize(mh_negloglik, init, args=(xx, yy, se, sp), jac=True, method="L-BFGS-B")
    return res.x, res.success


def iso_fit_predict(s_fit, y_fit, s_new):
    return IsotonicRegression(y_min=Q_BOUNDS[0], y_max=Q_BOUNDS[1], out_of_bounds="clip") \
        .fit(s_fit, y_fit).predict(s_new)


naive = {"auroc": roc_auc_score(y_obs, s), "auprc": average_precision_score(y_obs, s),
         "brier_score": brier_score_loss(y_obs, s), **threshold_metrics(y_obs, yhat)}
q_hold = iso_fit_predict(s, y_obs, s)
theta0, _ = fit_mh(x, y_obs, 1 - 1e-6, 1 - 1e-6, np.array([0.0, 1.0]))

# ------------------------------------------------------------------------------
# Monte Carlo loop
# ------------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
rng_cf = np.random.default_rng(CF_SEED)
rows, n_fail, n_fits = [], 0, 0
t0 = time.time()

for i in range(N_MC):
    se = rng.uniform(*SENS_RANGE)
    sp = rng.uniform(*SPEC_RANGE)
    u = rng.random(n)

    # (a) primary reference: isotonic q(s) fitted on all holdout labels
    pi_p = np.clip((q_hold - (1 - sp)) / (se + sp - 1), 0, 1)
    T_p = sample_T(pi_p, q_hold, se, u)
    if T_p.sum() < 5 or T_p.sum() == len(T_p):           # same validity rule as Step 7
        continue

    # (b, c) cross-fitted estimates: new fold split and bootstrap every iteration
    q_iso = np.empty(n); pi_mh = np.empty(n)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=int(rng_cf.integers(0, 2**31 - 1)))
    for fit_idx, pred_idx in skf.split(s, y_obs):
        b_idx = rng_cf.choice(fit_idx, size=len(fit_idx), replace=True)
        q_iso[pred_idx] = iso_fit_predict(s[b_idx], y_obs[b_idx], s[pred_idx])
        theta, ok = fit_mh(x[b_idx], y_obs[b_idx], se, sp, theta0)
        n_fits += 1
        n_fail += (not ok)
        pi_mh[pred_idx] = expit(theta[0] + theta[1] * x[pred_idx])

    pi_iso = np.clip((q_iso - (1 - sp)) / (se + sp - 1), 0, 1)
    T_iso = sample_T(pi_iso, q_iso, se, u)
    q_mh = (1 - sp) + (se + sp - 1) * pi_mh
    T_mh = sample_T(pi_mh, q_mh, se, u)

    rec = {"iteration": i + 1, "se_selfreport": se, "sp_selfreport": sp,
           "primary_frac_pi_clipped_zero": np.mean(pi_p == 0),
           "primary_implied_true_prevalence": pi_p.mean(),
           "cfiso_frac_pi_clipped_zero": np.mean(pi_iso == 0),
           "cfiso_implied_true_prevalence": pi_iso.mean(),
           "cfmh_implied_true_prevalence": pi_mh.mean()}
    for tag, T in (("primary", T_p), ("cfiso", T_iso), ("cfmh", T_mh)):
        rec.update({f"{tag}_{k}": v for k, v in all_metrics(T).items()})
    rows.append(rec)
    if (i + 1) % 200 == 0:
        print(f"  iteration {i + 1}/{N_MC}  ({time.time() - t0:.0f}s)")

mc = pd.DataFrame(rows)
print(f"\nValid iterations: {len(mc)} / {N_MC}   MH fits: {n_fits} (non-converged {n_fail})   "
      f"elapsed {time.time() - t0:.1f}s")


# ------------------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------------------
def q3(v):
    return v.median(), v.quantile(0.025), v.quantile(0.975)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


metrics = ["auroc", "auprc", "brier_score", "balanced_accuracy", "f1",
           "recall", "model_specificity", "precision"]
tags = {"primary": "Primary (holdout-fitted)", "cfiso": "Cross-fitted isotonic",
        "cfmh": "Cross-fitted Magder-Hughes"}
summary = []
print(f"\n{'Metric':<24s}{'Naive':>7s}   {tags['primary']:<26s} {tags['cfiso']:<26s} {tags['cfmh']:<26s}")
print("-" * 114)
for m in metrics + ["implied_true_prevalence"]:
    row = {"metric": m, "naive": y_obs.mean() if m == "implied_true_prevalence" else naive[m]}
    cells = []
    for tag in tags:
        t = q3(mc[f"{tag}_{m}"])
        row.update({f"{tag}_median": t[0], f"{tag}_si_lower": t[1], f"{tag}_si_upper": t[2]})
        cells.append(fmt(t))
    summary.append(row)
    print(f"{m:<24s}{row['naive']:7.3f}   {cells[0]:<26s} {cells[1]:<26s} {cells[2]:<26s}")

clip_p = q3(mc["primary_frac_pi_clipped_zero"])
clip_c = q3(mc["cfiso_frac_pi_clipped_zero"])
d_iso = q3(mc["cfiso_auroc"] - mc["primary_auroc"])
d_mh = q3(mc["cfmh_auroc"] - mc["primary_auroc"])
print("\nShare of holdout participants with pi(s) set to 0")
print(f"  primary (holdout-fitted isotonic): median {clip_p[0]:.3f} [{clip_p[1]:.3f}, {clip_p[2]:.3f}]")
print(f"  cross-fitted isotonic           : median {clip_c[0]:.3f} [{clip_c[1]:.3f}, {clip_c[2]:.3f}]")
print("Paired AUROC difference vs primary")
print(f"  cross-fitted isotonic      : median {d_iso[0]:+.3f} [{d_iso[1]:+.3f}, {d_iso[2]:+.3f}]")
print(f"  cross-fitted Magder-Hughes : median {d_mh[0]:+.3f} [{d_mh[1]:+.3f}, {d_mh[2]:+.3f}]")

mc.to_csv(artifacts_dir / "pba_crossfit_iterations.csv", index=False)
pd.DataFrame(summary).to_csv(artifacts_dir / "pba_crossfit_summary.csv", index=False)
numbers = {
    "k_folds": K_FOLDS, "n_valid_iterations": int(len(mc)),
    "mh_fits": int(n_fits), "mh_nonconverged": int(n_fail),
    "clip_share_primary": list(clip_p), "clip_share_cfiso": list(clip_c),
    "auroc_diff_cfiso_minus_primary": list(d_iso),
    "auroc_diff_cfmh_minus_primary": list(d_mh),
    **{f"{tag}_{m}": list(q3(mc[f"{tag}_{m}"]))
       for tag in tags for m in metrics + ["implied_true_prevalence"]},
}
with open(artifacts_dir / "pba_crossfit_manuscript_numbers.json", "w") as f:
    json.dump(numbers, f, indent=2, default=float)

print("\nExported: pba_crossfit_iterations.csv, pba_crossfit_summary.csv, "
      "pba_crossfit_manuscript_numbers.json")
print(f"Log: {log_dir / 'bias_sensitivity_crossfit.log'}")
print(f"Run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")