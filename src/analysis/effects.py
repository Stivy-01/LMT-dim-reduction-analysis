# -*- coding: utf-8 -*-
"""Effetti di modello/feature + correzione BH (ex BLOCCO 6 di thesis_analysis)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

try:  # Optional dependency.
    import statsmodels.api as sm  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover
    sm = None  # type: ignore
    smf = None  # type: ignore
    STATSMODELS_AVAILABLE = False

from src.analysis.projection import _fit_numpy_terms, _map_statsmodels_terms


# ==============================================================================
# BLOCCO 6 — Effetti di modello/feature + correzione BH (righe ~1044-1160)
# Stima effetti ID/feature per fase×trattamento, correzione FDR
# Benjamini-Hochberg, similarità/distanza tra vettori.
# -> futuro modulo: src/analysis/effects.py
# ==============================================================================
def _bh_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    valid = values[mask]
    if not len(valid):
        return adjusted
    order = np.argsort(valid)
    ranked = valid[order]
    n = len(ranked)
    q = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    adjusted[mask] = out
    return adjusted


def _fit_model_effects(
    frame: pd.DataFrame,
    response: str,
    use_statsmodels: bool,
) -> pd.DataFrame:
    data = frame.loc[
        frame["effect_eligible"]
        & frame[response].notna()
        & frame["phase"].isin(["baseline", "post_stress"])
        & frame["cage_id"].notna()
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=["response", "term", "estimate", "std_error", "p_value", "bh_q_value", "backend"])
    if not (use_statsmodels and STATSMODELS_AVAILABLE and smf is not None):
        raise RuntimeError(
            "The dimension-level models require statsmodels: the mixed model "
            "is the only specified estimator and no silent fallback is used."
        )
    # Mice are nested within cages, so the two random effects are only
    # separately identifiable when the *outer* level is the grouping factor
    # and the *inner* level is a variance component. Using mouse groups with a
    # cage variance component (as in earlier versions) put the two components
    # on a flat likelihood ridge, which produced singular Hessians and, in a
    # minority of runs, a silent fallback to robust OLS.
    if data["cage_id"].nunique() > 1 and data["mouse_id"].nunique() > 1:
        groups = data["cage_id"]
        vc_formula = {"mouse": "0 + C(mouse_id)"}
    else:
        groups = data["mouse_id"]
        vc_formula = None
    model = smf.mixedlm(
        f"{response} ~ C(phase) * C(treatment) + relative_night",
        data=data,
        groups=groups,
        vc_formula=vc_formula,
        re_formula="1",
    )
    try:
        fitted = model.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
    except Exception as exc:                     # pragma: no cover
        raise RuntimeError(
            "Mixed model for %s could not be fitted: %s: %s"
            % (response, type(exc).__name__, exc)
        ) from exc
    if not bool(getattr(fitted, "converged", True)):
        raise RuntimeError(
            "Mixed model for %s did not converge; refusing to substitute a "
            "different estimator." % response
        )
    terms = _map_statsmodels_terms(fitted, "statsmodels_mixedlm")
    terms.insert(0, "response", response)
    return terms


def _fit_feature_effects(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = frame.loc[
        frame["effect_eligible"] & frame["phase"].isin(["baseline", "post_stress"])
    ].copy()
    rows: list[pd.DataFrame] = []
    for feature in features:
        y = np.log1p(pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float))
        terms = _fit_numpy_terms(data, y)
        interaction = terms.loc[terms["term"].eq("phase_post_stress:treatment_stressed")].iloc[0]
        rows.append(
            pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "interaction_estimate": float(interaction["estimate"]),
                        "interaction_std_error": float(interaction["std_error"]),
                        "interaction_ci_low": float(interaction["ci_low"]),
                        "interaction_ci_high": float(interaction["ci_high"]),
                        "interaction_p_value": float(interaction["p_value"]),
                        "backend": interaction["backend"],
                        "n_rows": int(len(data)),
                    }
                ]
            )
        )
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not result.empty:
        result["interaction_q_value"] = _bh_adjust(result["interaction_p_value"])
    return result


def _pairwise_distance(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    distances = [float(np.linalg.norm(matrix[i] - matrix[j])) for i, j in combinations(range(len(matrix)), 2)]
    return float(np.mean(distances)) if distances else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)

