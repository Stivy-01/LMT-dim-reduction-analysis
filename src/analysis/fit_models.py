# -*- coding: utf-8 -*-
"""Fit modelli di proiezione + merge scores (ex BLOCCO 8 di thesis_analysis)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.config import DEFAULT_SEED
from src.analysis.projection import RegularizedLDAProjection, _temporal_validation
from src.analysis.quality_control import _log1p_matrix


# ==============================================================================
# BLOCCO 8 — Fit modelli di proiezione + merge scores (righe ~1300-1398)
# PCA + LDA su feature log1p, loadings, merge degli score nel frame QC.
# -> futuro modulo: src/analysis/fit_models.py
# ==============================================================================
def _fit_projection_models(
    frame: pd.DataFrame,
    complete_features: list[str],
    use_statsmodels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA, StandardScaler, RegularizedLDAProjection, dict[str, Any]]:
    projection = frame.loc[frame["projection_eligible"]].copy()
    baseline = projection.loc[projection["phase"].eq("baseline")].copy()
    if baseline.empty:
        raise ValueError("No projection-eligible baseline rows available.")

    scaler = StandardScaler()
    baseline_log = _log1p_matrix(baseline, complete_features)
    baseline_scaled = scaler.fit_transform(baseline_log)

    n_components = min(5, baseline_scaled.shape[0], baseline_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=DEFAULT_SEED)
    pca.fit(baseline_scaled)

    pca_scores = pca.transform(scaler.transform(_log1p_matrix(projection, complete_features)))
    pca_columns = [f"pca_{index + 1}" for index in range(pca_scores.shape[1])]
    pca_scores_df = projection[["mouse_id", "interval_start", "cage_id", "phase", "analysis_tier", "projection_eligible", "effect_eligible"]].copy()
    for index, column in enumerate(pca_columns):
        pca_scores_df[column] = pca_scores[:, index]

    pca_loadings = []
    for index, component in enumerate(pca.components_):
        for feature, loading in zip(complete_features, component):
            pca_loadings.append(
                {
                    "component": f"pca_{index + 1}",
                    "feature": feature,
                    "loading": float(loading),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                }
            )
    pca_loadings_df = pd.DataFrame(pca_loadings)

    validation = _temporal_validation(
        baseline,
        baseline_scaled,
        baseline["mouse_id"],
    )

    lda_train = RegularizedLDAProjection().fit(
        baseline_scaled,
        baseline["mouse_id"].to_numpy(),
    )
    lda_scores = lda_train.transform(scaler.transform(_log1p_matrix(projection, complete_features)))
    lda_columns = [f"id_{index + 1}" for index in range(lda_scores.shape[1])]

    projection_scores = projection[["mouse_id", "interval_start", "cage_id", "phase", "analysis_tier", "projection_eligible", "effect_eligible"]].copy()
    for index, column in enumerate(lda_columns):
        projection_scores[column] = lda_scores[:, index]
    projection_scores["identity_composite"] = (
        projection_scores[[c for c in lda_columns[:2] if c in projection_scores.columns]]
        .mean(axis=1)
        if lda_columns
        else np.nan
    )

    identity_loadings = []
    if lda_train.components_ is not None:
        for index, component in enumerate(lda_train.components_):
            for feature, loading in zip(complete_features, component):
                identity_loadings.append(
                    {
                        "component": f"id_{index + 1}",
                        "feature": feature,
                        "loading": float(loading),
                        "eigenvalue": float(lda_train.eigenvalues_[index]) if lda_train.eigenvalues_ is not None else np.nan,
                    }
                )
    identity_loadings_df = pd.DataFrame(identity_loadings)

    return pca_scores_df, pca_loadings_df, identity_loadings_df, pca, scaler, lda_train, validation


def _merge_scores(
    frame: pd.DataFrame,
    projection_scores: pd.DataFrame,
    pca_scores: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    out = out.merge(
        projection_scores.drop(columns=["projection_eligible", "effect_eligible"]),
        on=["mouse_id", "interval_start", "cage_id", "phase", "analysis_tier"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        pca_scores.drop(columns=["projection_eligible", "effect_eligible"]),
        on=["mouse_id", "interval_start", "cage_id", "phase", "analysis_tier"],
        how="left",
        suffixes=("", "_pca"),
        validate="many_to_one",
    )
    return out

