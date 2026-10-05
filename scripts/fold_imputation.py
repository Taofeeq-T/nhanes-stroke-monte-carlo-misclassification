"""
================================================================================
Fold-wise HyperImpute imputation (fit on training data only)
================================================================================
HyperImpute (v0.1.x) is transductive: its plugin `transform()` re-runs
`fit_transform()` on whatever data it receives, so it has no native way to fit
on a training set and apply the fitted imputer to unseen rows.

`TrainOnlyHyperImputer` works around this. It
  1. runs HyperImpute's full column-wise model selection and iterative
     imputation on the TRAINING rows only (`fit`),
  2. keeps the per-column models HyperImpute selected and fitted in its final
     iteration, plus training-set means, category encodings and value limits,
  3. imputes NEW rows (`transform`) by initializing missing entries with
     training means and iterating the stored per-column models, without
     refitting anything on the new rows.

No information from validation or holdout rows (features or outcome) enters
model selection or fitting. The outcome is never part of the imputation matrix.

Post-processing (categorical decoding, clinical range clipping, binary
rounding) uses fixed rules or training-set categories only.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Fixed, pre-specified clinical plausibility ranges (not learned from data)
RANGE_CONSTRAINTS = {
    "age": (20, 80), "mean_sbp": (70, 250), "mean_dbp": (40, 150),
    "bmxwaist": (50, 200), "bmxbmi": (12, 80),
    "alcohol_drinks_per_day": (0, 20),
    "hba1c": (3, 18), "total_chol": (80, 500), "hdl_c": (10, 150),
    "triglycerides": (20, 2000), "creatinine": (0.1, 15), "wbc": (1, 50),
    "pir": (0, 5), "sleep_hrs": (1, 18),
}


def make_holdout_split(df: pd.DataFrame, test_size: float = 0.20,
                       random_state: int = RANDOM_SEED):
    """Positional train/holdout indices. Identical to the split in
    autogluon_modeling.py (same n, stratification on stroke, and seed)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(df)), test_size=test_size,
                            stratify=df["stroke"], random_state=random_state)


def variable_types(codebook: pd.DataFrame, cols: list[str]):
    """Return (continuous, categorical, binary) lists restricted to `cols`."""
    cb = codebook.set_index("variable")["type"]
    cont = [c for c in cols if cb.get(c) == "continuous"]
    cat = [c for c in cols if cb.get(c) == "categorical"]
    binv = [c for c in cols if cb.get(c) == "binary"]
    return cont, cat, binv


class TrainOnlyHyperImputer:
    def __init__(self, codebook: pd.DataFrame, cols: list[str],
                 random_state: int = RANDOM_SEED, max_iter: int = 20,
                 tol: float = 1e-3):
        self.cols = list(cols)
        self.cont, self.cat, self.bin = variable_types(codebook, self.cols)
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------ encode
    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].copy()
        for col in self.cat:
            X[col] = X[col].astype(object)
            codes = pd.Categorical(X[col], categories=self.cat_mappings[col]).codes
            X[col] = pd.Series(codes, index=X.index).replace(-1, np.nan).astype(float)
        for col in self.cont + self.bin:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        return X

    def _decode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cat:
            cats = self.cat_mappings[col]
            X[col] = X[col].round().clip(0, len(cats) - 1).astype(int)
            X[col] = X[col].map(dict(enumerate(cats)))
        for col, (lo, hi) in RANGE_CONSTRAINTS.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        for col in self.bin:
            X[col] = X[col].round().clip(0, 1).astype(int)
        return X

    # --------------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit HyperImpute on training rows. Returns imputed training rows."""
        from hyperimpute.plugins.imputers import Imputers

        # Category levels come from training rows only
        self.cat_mappings = {
            c: train_df[c].astype(object).dropna().unique().tolist() for c in self.cat
        }
        X_tr = self._encode(train_df)
        self.train_mask_any = X_tr.isna().any()

        self.plugin = Imputers().get(
            "hyperimpute",
            optimizer="hyperband",
            classifier_seed=["logistic_regression", "random_forest", "catboost"],
            regression_seed=["linear_regression", "random_forest_regressor", "catboost_regressor"],
            random_state=self.random_state,
        )
        imputed = self.plugin.fit_transform(X_tr.copy())
        imputed = pd.DataFrame(np.asarray(imputed, dtype=float),
                               columns=self.cols, index=train_df.index)

        # State needed to impute unseen rows
        m = self.plugin.model
        self._m = m
        X_enc = self._label_encode(X_tr.reset_index(drop=True))
        self.train_means = X_enc.mean()          # baseline (mean) initialization
        return self._decode(imputed)

    # ---------------------------------------------------------------- helpers
    def _label_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply HyperImpute's internal LabelEncoders (fitted on training)."""
        X = X.copy()
        X.columns = X.columns.map(str)
        for col, le in self._m.encoders.items():
            nn = X[col].notna()
            known = set(le.classes_.tolist())
            vals = X.loc[nn, col]
            unseen = ~vals.isin(known)
            if unseen.any():                      # unseen level -> treat as missing
                X.loc[vals[unseen].index, col] = np.nan
                nn = X[col].notna()
                vals = X.loc[nn, col]
            X.loc[nn, col] = le.transform(vals).astype(int)
        return X.astype(float)

    def _label_decode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self._m.encoders.items():
            codes = X[col].round().clip(0, len(le.classes_) - 1).astype(int)
            X[col] = le.inverse_transform(codes)
        return X

    # --------------------------------------------------------------- transform
    def transform(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Impute unseen rows with the models fitted on training rows only."""
        m = self._m
        X = self._encode(new_df)
        idx = X.index
        X = self._label_encode(X.reset_index(drop=True))
        mask = X.isna()
        X = X.fillna(self.train_means)

        all_cols = list(m.all_cols)
        for _ in range(self.max_iter):
            X_prev = X.copy()
            for col in m.imputation_order:
                if not mask[col].any():
                    continue
                if not self.train_mask_any[col] or col not in m.column_to_model:
                    continue                       # no fitted model: keep train mean
                est = m.column_to_model[col]
                covs = X[[c for c in all_cols if c != col]]
                pred = est.predict(covs[mask[col]])
                pred = np.asarray(getattr(pred, "values", pred)).squeeze()
                lo, hi = m.limits[col]
                X.loc[mask[col], col] = np.clip(pred, lo, hi)
            if np.nanmax(np.abs((X - X_prev).to_numpy())) < self.tol:
                break

        X = self._label_decode(X)
        X.index = idx
        return self._decode(X.astype(float))


def impute_split(train_df, test_df, codebook, cols, cache_dir: Path | None = None,
                 tag: str = "", random_state: int = RANDOM_SEED):
    """Fit on train_df, apply to test_df. Caches results as parquet if cache_dir."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        f_tr, f_te = cache_dir / f"{tag}_train.parquet", cache_dir / f"{tag}_test.parquet"
        if f_tr.exists() and f_te.exists():
            tr, te = pd.read_parquet(f_tr), pd.read_parquet(f_te)
            if set(tr.columns) != set(cols) or len(tr) != len(train_df) or len(te) != len(test_df):
                raise ValueError(f"Cached imputation '{tag}' does not match the requested "
                                 f"columns/rows. Delete {cache_dir} and rerun.")
            tr.index, te.index = train_df.index, test_df.index
            return tr[cols], te[cols]

    imp = TrainOnlyHyperImputer(codebook, cols, random_state=random_state)
    tr = imp.fit(train_df)
    te = imp.transform(test_df)

    if cache_dir is not None:
        tr.reset_index(drop=True).to_parquet(f_tr, index=False)
        te.reset_index(drop=True).to_parquet(f_te, index=False)
    return tr[cols], te[cols]


def smoke_test(analytic_csv: Path, codebook_csv: Path, cols: list[str],
               n_train: int = 3000, n_test: int = 1000, mask_frac: float = 0.10):
    """Quick check: impute artificially masked observed values in unseen rows
    and compare RMSE against training-mean imputation."""
    df = pd.read_csv(analytic_csv).sample(n_train + n_test, random_state=RANDOM_SEED)
    cb = pd.read_csv(codebook_csv)
    tr, te = df.iloc[:n_train], df.iloc[n_train:].copy()
    cont, _, _ = variable_types(cb, cols)
    rng = np.random.default_rng(RANDOM_SEED)
    truth = {}
    for c in cont:
        obs = te.index[te[c].notna()]
        hide = rng.choice(obs, size=int(mask_frac * len(obs)), replace=False)
        truth[c] = te.loc[hide, c].copy()
        te.loc[hide, c] = np.nan
    imp = TrainOnlyHyperImputer(cb, cols)
    imp.fit(tr)
    te_imp = imp.transform(te)
    print(f"{'variable':<25s} {'RMSE model':>11s} {'RMSE mean':>10s}")
    for c, s in truth.items():
        r_m = np.sqrt(((te_imp.loc[s.index, c] - s) ** 2).mean())
        r_b = np.sqrt(((tr[c].mean() - s) ** 2).mean())
        print(f"{c:<25s} {r_m:11.3f} {r_b:10.3f}")
    assert te_imp[cols].isna().sum().sum() == 0, "NaNs remain after transform"
    print("Smoke test passed: no missing values remain in transformed rows.")


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    (root / "logs").mkdir(exist_ok=True)
    _log = open(root / "logs" / "stage4_smoke_test.log", "w")

    class _Tee:
        def write(self, d):
            sys.__stdout__.write(d); _log.write(d); _log.flush()

        def flush(self):
            sys.__stdout__.flush(); _log.flush()

        def __getattr__(self, name):
            return getattr(sys.__stdout__, name)
    sys.stdout = sys.stderr = _Tee()
    cb = pd.read_csv(root / "artifacts" / "variable_codebook.csv")
    mr = pd.read_csv(root / "artifacts" / "missingness_report.csv")
    tier_d = mr[mr["tier"].str.startswith("D")]["variable"].tolist()
    cols = [v for v in cb["variable"] if v != "stroke" and v not in tier_d]
    smoke_test(root / "data" / "nhanes_stroke_analytic_2003_2023.csv",
               root / "artifacts" / "variable_codebook.csv", cols)