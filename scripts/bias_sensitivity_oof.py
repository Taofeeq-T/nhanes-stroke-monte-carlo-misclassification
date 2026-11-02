"""
================================================================================
NHANES Stroke Prediction: q(s) estimated independently of the holdout labels
(Reviewer 2, comment R2)  [DIAGNOSTIC ONLY; RESULTS NOT USED]
================================================================================
Status: this route was abandoned. The OOF training predictions of the saved
predictor were not on the same scale as the holdout predictions (OOF AUROC
0.737 vs holdout 0.790; 99th percentile of the score 0.845 vs 0.495), so a q(s)
fitted on them does not transfer to the holdout scores. The script is kept to
document that check. The R2 analysis reported in the manuscript is
bias_sensitivity_crossfit.py.

In the primary probabilistic bias analysis (autogluon_modeling.py, Step 7)
q(s) = P(Y*=1 | s) is fitted on the same holdout labels that are then used to
reconstruct the latent outcome, and the simulation interval does not reflect
uncertainty in estimating q(s).

This script re-estimates q(s) from the out-of-fold (OOF) predictions of the
saved final AutoGluon predictor on the training set (n = 38,156), which never
used the holdout labels, and applies it to the fixed holdout scores. In each
Monte Carlo iteration the training rows are resampled with replacement before
fitting, so the interval also carries the estimation uncertainty in q(s).

Two reconstructions are run, both fitted on the bootstrapped OOF data:
  - isotonic : as in Step 7 (isotonic q(s), inverted, negative pi(s) set to 0)
  - mh       : Magder-Hughes maximum likelihood, pi(s) = expit(a + b*logit s)
The primary analysis (isotonic q(s) fitted on the holdout) is re-run in the
same loop for reference. Se, Sp, and the uniforms used to sample the latent
outcome come from the same random stream as Step 7 (seed 42), so all three
are paired; bootstrap indices use a separate stream (seed 43).

No model is trained. AutoGluon is needed only to load models/final.

Inputs : models/final (saved predictor), artifacts/holdout_predictions.csv
Outputs: artifacts/oof_predictions_train.csv     OOF scores + labels (training set)
         artifacts/pba_oof_iterations.csv        per-iteration results
         artifacts/pba_oof_summary.csv           sensitivity version of Table 2 (bias column)
         artifacts/pba_oof_manuscript_numbers.json
         logs/bias_sensitivity_oof.log
Usage  : python scripts/bias_sensitivity_oof.py
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

project_root = Path(__file__).resolve().parent.parent
artifacts_dir = project_root / "artifacts"
models_dir = project_root / "models"
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


_log_file = open(log_dir / "bias_sensitivity_oof.log", "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)

RANDOM_SEED = 42
BOOT_SEED = 43
N_MC = 1000
SENS_RANGE = (0.36, 0.98)   # https://doi.org/10.1371/journal.pone.0137538
SPEC_RANGE = (0.96, 0.996)  # https://doi.org/10.1371/journal.pone.0137538
EPS = 1e-6
Q_BOUNDS = (1e-4, 1 - 1e-4)

print("=" * 72)
print("R2 sensitivity analysis: q(s) estimated from out-of-fold training predictions")
print("=" * 72)
print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Iterations: {N_MC}   seed: {RANDOM_SEED} (Se, Sp, T)   bootstrap seed: {BOOT_SEED}")
print(f"Se ~ U{SENS_RANGE}   Sp ~ U{SPEC_RANGE}")

# ------------------------------------------------------------------------------
# 1. Out-of-fold predictions of the saved final predictor
# ------------------------------------------------------------------------------
oof_path = artifacts_dir / "oof_predictions_train.csv"
if oof_path.exists():
    oof_df = pd.read_csv(oof_path)
    print(f"\nLoaded cached OOF predictions: {oof_path.name}")
else:
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(models_dir / "final"))
    print(f"\nLoaded predictor: {models_dir / 'final'}   best model: {predictor.model_best}")
    oof = None
    for fn in ("predict_proba_oof", "get_oof_pred_proba"):
        if hasattr(predictor, fn):
            try:
                oof = getattr(predictor, fn)()
                print(f"OOF retrieved with predictor.{fn}()")
                break
            except Exception as e:
                print(f"  predictor.{fn}() failed: {e}")
    if oof is None:
        raise SystemExit("Could not retrieve OOF predictions from the saved predictor.")
    pos = predictor.positive_class
    s_col = oof[pos] if pos in oof.columns else oof[oof.columns[-1]]

    y_tr = None
    try:
        res = predictor.load_data_internal(data="train", return_X=False, return_y=True)
        y_tr = res[1] if isinstance(res, tuple) else res
    except Exception as e:
        print(f"  load_data_internal failed ({e}); using models/final/utils/data/y.pkl")
        y_tr = pd.read_pickle(models_dir / "final" / "utils" / "data" / "y.pkl")
    y_tr = pd.Series(y_tr)
    y_tr = y_tr.loc[s_col.index] if set(s_col.index) <= set(y_tr.index) else y_tr
    if len(y_tr) != len(s_col):
        raise SystemExit(f"Length mismatch: OOF {len(s_col)} vs labels {len(y_tr)}")
    y_vals = (y_tr.to_numpy() == pos).astype(int) if y_tr.dtype == object else y_tr.to_numpy().astype(int)
    oof_df = pd.DataFrame({"row_index": s_col.index, "y_true": y_vals,
                           "y_pred_proba_oof": s_col.to_numpy()})
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved {oof_path.name} (reused on later runs, so AutoGluon is not needed again)")

s_tr = oof_df["y_pred_proba_oof"].to_numpy()
y_tr = oof_df["y_true"].to_numpy().astype(int)
x_tr = logit(np.clip(s_tr, EPS, 1 - EPS))

h = pd.read_csv(artifacts_dir / "holdout_predictions.csv")
y_obs = h["y_true"].to_numpy().astype(int)
s = h["y_pred_proba"].to_numpy()
thr = float(h["optimal_threshold"].iloc[0])
yhat = (s >= thr).astype(int)
x_h = logit(np.clip(s, EPS, 1 - EPS))

# ------------------------------------------------------------------------------
# 2. Check that OOF and holdout scores are on a comparable scale
# ------------------------------------------------------------------------------
print(f"\nTraining OOF: n = {len(y_tr)}, positives = {y_tr.sum()} (prevalence {y_tr.mean():.4f}), "
      f"AUROC = {roc_auc_score(y_tr, s_tr):.4f}")
print(f"Holdout     : n = {len(y_obs)}, positives = {y_obs.sum()} (prevalence {y_obs.mean():.4f}), "
      f"AUROC = {roc_auc_score(y_obs, s):.4f}")
qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
print("Score quantiles   " + "  ".join(f"q{int(q*100):02d}" for q in qs))
print("  OOF            " + "  ".join(f"{np.quantile(s_tr, q):.3f}" for q in qs))
print("  Holdout        " + "  ".join(f"{np.quantile(s, q):.3f}" for q in qs))
print(f"Mean score: OOF {s_tr.mean():.4f}, holdout {s.mean():.4f}")
print(f"Holdout scores outside OOF range: {np.mean((s < s_tr.min()) | (s > s_tr.max())):.4f}")


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


def mh_negloglik(theta, x, y, se, sp):
    a, b = theta
    pi = expit(a + b * x)
    q = (1 - sp) + (se + sp - 1) * pi
    nll = -np.sum(y * np.log(q) + (1 - y) * np.log1p(-q))
    w = -(y / q - (1 - y) / (1 - q)) * (se + sp - 1) * pi * (1 - pi)
    return nll, np.array([w.sum(), (w * x).sum()])


def fit_mh(x, y, se, sp, init):
    res = minimize(mh_negloglik, init, args=(x, y, se, sp), jac=True, method="L-BFGS-B")
    return res.x, res.success


def iso_fit_predict(s_fit, y_fit, s_new):
    return IsotonicRegression(y_min=Q_BOUNDS[0], y_max=Q_BOUNDS[1], out_of_bounds="clip") \
        .fit(s_fit, y_fit).predict(s_new)


naive = {"auroc": roc_auc_score(y_obs, s), "auprc": average_precision_score(y_obs, s),
         "brier_score": brier_score_loss(y_obs, s), **threshold_metrics(y_obs, yhat)}

q_hold = iso_fit_predict(s, y_obs, s)                     # primary (holdout-fitted)
theta_prev, _ = fit_mh(x_tr, y_tr, 1 - 1e-6, 1 - 1e-6, np.array([0.0, 1.0]))

# ------------------------------------------------------------------------------
# 3. Monte Carlo loop
# ------------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
rng_boot = np.random.default_rng(BOOT_SEED)
rows, n_fail = [], 0
t0 = time.time()
n_tr = len(y_tr)

for i in range(N_MC):
    se = rng.uniform(*SENS_RANGE)
    sp = rng.uniform(*SPEC_RANGE)
    u = rng.random(len(y_obs))
    idx = rng_boot.integers(0, n_tr, n_tr)                # bootstrap of training rows

    # (a) primary reference: isotonic q(s) fitted on holdout labels
    pi_p = np.clip((q_hold - (1 - sp)) / (se + sp - 1), 0, 1)
    T_p = sample_T(pi_p, q_hold, se, u)
    if T_p.sum() < 5 or T_p.sum() == len(T_p):           # same validity rule as Step 7
        continue

    # (b) isotonic q(s) fitted on bootstrapped OOF training predictions
    q_iso = iso_fit_predict(s_tr[idx], y_tr[idx], s)
    pi_iso = np.clip((q_iso - (1 - sp)) / (se + sp - 1), 0, 1)
    T_iso = sample_T(pi_iso, q_iso, se, u)

    # (c) Magder-Hughes fitted on bootstrapped OOF training predictions
    theta, ok = fit_mh(x_tr[idx], y_tr[idx], se, sp, theta_prev)
    if ok:
        theta_prev = theta
    else:
        n_fail += 1
    pi_mh = expit(theta[0] + theta[1] * x_h)
    q_mh = (1 - sp) + (se + sp - 1) * pi_mh
    T_mh = sample_T(pi_mh, q_mh, se, u)

    rec = {"iteration": i + 1, "se_selfreport": se, "sp_selfreport": sp,
           "primary_frac_pi_clipped_zero": np.mean(pi_p == 0),
           "primary_implied_true_prevalence": pi_p.mean(),
           "oofiso_frac_pi_clipped_zero": np.mean(pi_iso == 0),
           "oofiso_implied_true_prevalence": pi_iso.mean(),
           "oofmh_a": theta[0], "oofmh_b": theta[1], "oofmh_converged": ok,
           "oofmh_implied_true_prevalence": pi_mh.mean()}
    for tag, T in (("primary", T_p), ("oofiso", T_iso), ("oofmh", T_mh)):
        rec.update({f"{tag}_{k}": v for k, v in all_metrics(T).items()})
    rows.append(rec)
    if (i + 1) % 200 == 0:
        print(f"  iteration {i + 1}/{N_MC}  ({time.time() - t0:.0f}s)")

mc = pd.DataFrame(rows)
print(f"\nValid iterations: {len(mc)} / {N_MC}   MH non-converged fits: {n_fail}   "
      f"elapsed {time.time() - t0:.1f}s")


# ------------------------------------------------------------------------------
# 4. Summaries
# ------------------------------------------------------------------------------
def q3(v):
    return v.median(), v.quantile(0.025), v.quantile(0.975)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


metrics = ["auroc", "auprc", "brier_score", "balanced_accuracy", "f1",
           "recall", "model_specificity", "precision"]
labels = {"primary": "Primary (holdout-fitted isotonic)",
          "oofiso": "OOF-fitted isotonic", "oofmh": "OOF-fitted Magder-Hughes"}
summary = []
print(f"\n{'Metric':<20s}{'Naive':>7s}   {labels['primary']:<26s} {labels['oofiso']:<26s} "
      f"{labels['oofmh']:<26s}")
print("-" * 110)
for m in metrics + ["implied_true_prevalence"]:
    row = {"metric": m, "naive": y_obs.mean() if m == "implied_true_prevalence" else naive[m]}
    cells = []
    for tag in ("primary", "oofiso", "oofmh"):
        t = q3(mc[f"{tag}_{m}"])
        row.update({f"{tag}_median": t[0], f"{tag}_si_lower": t[1], f"{tag}_si_upper": t[2]})
        cells.append(fmt(t))
    summary.append(row)
    print(f"{m:<20s}{row['naive']:7.3f}   {cells[0]:<26s} {cells[1]:<26s} {cells[2]:<26s}")

clip_p = q3(mc["primary_frac_pi_clipped_zero"])
clip_o = q3(mc["oofiso_frac_pi_clipped_zero"])
d_iso = q3(mc["oofiso_auroc"] - mc["primary_auroc"])
d_mh = q3(mc["oofmh_auroc"] - mc["primary_auroc"])
print(f"\nShare of holdout participants with pi(s) set to 0")
print(f"  primary (holdout-fitted isotonic): median {clip_p[0]:.3f} [{clip_p[1]:.3f}, {clip_p[2]:.3f}]")
print(f"  OOF-fitted isotonic             : median {clip_o[0]:.3f} [{clip_o[1]:.3f}, {clip_o[2]:.3f}]")
print(f"Paired AUROC difference vs primary")
print(f"  OOF-fitted isotonic      : median {d_iso[0]:+.3f} [{d_iso[1]:+.3f}, {d_iso[2]:+.3f}]")
print(f"  OOF-fitted Magder-Hughes : median {d_mh[0]:+.3f} [{d_mh[1]:+.3f}, {d_mh[2]:+.3f}]")

mc.to_csv(artifacts_dir / "pba_oof_iterations.csv", index=False)
pd.DataFrame(summary).to_csv(artifacts_dir / "pba_oof_summary.csv", index=False)
numbers = {
    "n_train_oof": int(len(y_tr)), "oof_auroc": float(roc_auc_score(y_tr, s_tr)),
    "oof_prevalence": float(y_tr.mean()), "holdout_auroc": float(naive["auroc"]),
    "n_valid_iterations": int(len(mc)), "mh_nonconverged": int(n_fail),
    "clip_share_primary": list(clip_p), "clip_share_oofiso": list(clip_o),
    "auroc_diff_oofiso_minus_primary": list(d_iso),
    "auroc_diff_oofmh_minus_primary": list(d_mh),
    **{f"{tag}_{m}": list(q3(mc[f"{tag}_{m}"]))
       for tag in ("primary", "oofiso", "oofmh") for m in metrics + ["implied_true_prevalence"]},
}
with open(artifacts_dir / "pba_oof_manuscript_numbers.json", "w") as f:
    json.dump(numbers, f, indent=2, default=float)

print("\nExported: oof_predictions_train.csv, pba_oof_iterations.csv, pba_oof_summary.csv, "
      "pba_oof_manuscript_numbers.json")
print(f"Log: {log_dir / 'bias_sensitivity_oof.log'}")
print(f"Run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")