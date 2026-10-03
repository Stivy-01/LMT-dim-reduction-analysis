# -*- coding: utf-8 -*-
"""Proiezione PCA/LDA + validazione temporale (ex BLOCCO 5 di thesis_analysis)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import t as student_t


# ==============================================================================
# BLOCCO 5 — Proiezione PCA/LDA + validazione temporale (righe ~874-1043)
# LDA regolarizzata su baseline, validazione temporale, fit con backend
# numpy/statsmodels e mappatura dei termini statistici.
# -> futuro modulo: src/analysis/projection.py
# ==============================================================================
class RegularizedLDAProjection:
    def __init__(self, regularization: float = 1e-2) -> None:
        self.regularization = float(regularization)
        self.components_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "RegularizedLDAProjection":
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("Need at least two classes for LDA.")
        n_features = X.shape[1]
        overall_mean = X.mean(axis=0)
        Sw = np.zeros((n_features, n_features), dtype=float)
        Sb = np.zeros((n_features, n_features), dtype=float)
        for cls in classes:
            rows = X[labels == cls]
            centroid = rows.mean(axis=0)
            if len(rows) > 1:
                centered = rows - centroid
                Sw += centered.T @ centered
            diff = (centroid - overall_mean).reshape(-1, 1)
            Sb += len(rows) * (diff @ diff.T)
        Sw = (Sw + Sw.T) / 2.0 + self.regularization * np.eye(n_features)
        Sb = (Sb + Sb.T) / 2.0
        try:
            eigenvalues, eigenvectors = eigh(Sb, Sw, check_finite=False)
        except Exception:
            fallback = np.linalg.pinv(Sw) @ Sb
            eigenvalues, eigenvectors = np.linalg.eig(fallback)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        n_components = min(n_features, len(classes) - 1)
        self.components_ = eigenvectors[:, :n_components].T
        self.eigenvalues_ = eigenvalues[:n_components]
        self.classes_ = classes
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("Projection not fitted.")
        return X @ self.components_.T


def _temporal_validation(
    baseline_frame: pd.DataFrame, baseline_scores: np.ndarray, labels: pd.Series
) -> dict[str, Any]:
    if baseline_frame.empty or baseline_frame["mouse_id"].nunique() < 2:
        return {
            "status": "not_feasible",
            "n_validation_rows": 0,
            "accuracy": np.nan,
            "note": "insufficient baseline data",
        }
    rows = baseline_frame.copy()
    rows["_idx"] = np.arange(len(rows))
    train_idx: list[int] = []
    valid_idx: list[int] = []
    for _, group in rows.groupby("mouse_id", sort=False):
        ordered = group.sort_values(["interval_start", "_idx"])
        if len(ordered) < 2:
            train_idx.extend(ordered["_idx"].tolist())
            continue
        train_idx.extend(ordered.iloc[:-1]["_idx"].tolist())
        valid_idx.append(int(ordered.iloc[-1]["_idx"]))
    if not valid_idx:
        return {
            "status": "not_feasible",
            "n_validation_rows": 0,
            "accuracy": np.nan,
            "note": "no temporal holdout available",
        }
    train_mask = rows["_idx"].isin(train_idx).to_numpy()
    valid_mask = rows["_idx"].isin(valid_idx).to_numpy()
    if train_mask.sum() < 2 or valid_mask.sum() == 0:
        return {
            "status": "not_feasible",
            "n_validation_rows": int(valid_mask.sum()),
            "accuracy": np.nan,
            "note": "split collapsed",
        }
    model = RegularizedLDAProjection().fit(baseline_scores[train_mask], labels.loc[train_mask].to_numpy())
    train_proj = model.transform(baseline_scores[train_mask])
    valid_proj = model.transform(baseline_scores[valid_mask])
    centroids = {}
    train_labels = labels.loc[train_mask].to_numpy()
    for cls in np.unique(train_labels):
        centroids[cls] = train_proj[train_labels == cls].mean(axis=0)
    predictions = []
    for score in valid_proj:
        prediction = min(centroids, key=lambda cls: float(np.linalg.norm(score - centroids[cls])))
        predictions.append(prediction)
    truth = labels.loc[valid_mask].to_numpy()
    accuracy = float(np.mean(np.asarray(predictions) == truth))
    return {
        "status": "ok",
        "n_validation_rows": int(len(truth)),
        "accuracy": accuracy,
        "note": "leave-last-baseline-night-per-mouse",
    }


def _fit_numpy_terms(frame: pd.DataFrame, response: np.ndarray) -> pd.DataFrame:
    phase = frame["phase"].eq("post_stress").astype(float).to_numpy()
    treatment = frame["treatment"].eq("stressed").astype(float).to_numpy()
    relative_night = pd.to_numeric(frame["relative_night"], errors="coerce").to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(frame)), phase, treatment, phase * treatment, relative_night]
    )
    beta, _, rank, _ = np.linalg.lstsq(X, response, rcond=None)
    resid = response - X @ beta
    dof = max(len(response) - rank, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, a_min=0.0, a_max=None))
    t_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * student_t.sf(np.abs(t_stat), df=dof)
    critical = float(student_t.ppf(0.975, df=dof))
    return pd.DataFrame(
        {
            "term": [
                "Intercept",
                "phase_post_stress",
                "treatment_stressed",
                "phase_post_stress:treatment_stressed",
                "relative_night",
            ],
            "estimate": beta,
            "std_error": se,
            "ci_low": beta - critical * se,
            "ci_high": beta + critical * se,
            "p_value": pvals,
            "backend": "numpy_ols",
        }
    )


def _map_statsmodels_terms(result: Any, backend: str) -> pd.DataFrame:
    rows = []
    for term, estimate in result.params.items():
        if term == "Intercept":
            mapped = "Intercept"
        elif "C(phase)[T.post_stress]:C(treatment)[T.stressed]" in term or "C(treatment)[T.stressed]:C(phase)[T.post_stress]" in term:
            mapped = "phase_post_stress:treatment_stressed"
        elif "C(phase)[T.post_stress]" in term:
            mapped = "phase_post_stress"
        elif "C(treatment)[T.stressed]" in term:
            mapped = "treatment_stressed"
        elif term == "relative_night":
            mapped = "relative_night"
        else:
            continue
        standard_error = float(result.bse.get(term, np.nan))
        rows.append(
            {
                "term": mapped,
                "estimate": float(estimate),
                "std_error": standard_error,
                "ci_low": float(estimate) - 1.96 * standard_error,
                "ci_high": float(estimate) + 1.96 * standard_error,
                "p_value": float(result.pvalues.get(term, np.nan)),
                "backend": backend,
            }
        )
    return pd.DataFrame(rows)

