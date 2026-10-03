# -*- coding: utf-8 -*-
"""Metriche di gruppo + bootstrap/permutazioni (ex BLOCCO 7 di thesis_analysis)."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.effects import _cosine_similarity, _pairwise_distance
from src.analysis.text_parsing import _stable_seed


# ==============================================================================
# BLOCCO 7 — Metriche di gruppo + bootstrap/permutazioni (righe ~1161-1299)
# Aggrega metriche per gabbia/notte e calcola change-in-change con
# bootstrap e test di permutazione sign-flip.
# -> futuro modulo: src/analysis/group_stats.py
# ==============================================================================
def _compute_group_metrics(
    frame: pd.DataFrame,
    score_cols: list[str],
    complete_features: list[str],
) -> pd.DataFrame:
    data = frame.loc[frame["projection_eligible"]].copy()
    if data.empty:
        return pd.DataFrame()
    active_cols = [c for c in complete_features if "_active_" in c or c.endswith("_active_count")]
    passive_cols = [c for c in complete_features if "_passive_" in c or c.endswith("_passive_count")]
    rows: list[dict[str, Any]] = []
    for (cage_id, night_date), group in data.groupby(["cage_id", "night_date"], sort=True):
        group = group.sort_values(["interval_start", "mouse_id"])
        effect_group = group.loc[group["effect_eligible"]].copy()
        scores = group[score_cols].to_numpy(dtype=float)
        centroid = scores.mean(axis=0)
        baseline_scores = data.loc[
            (data["cage_id"].eq(cage_id)) & (data["night_date"] < night_date) & (data["phase"].eq("baseline")),
            score_cols,
        ].to_numpy(dtype=float)
        baseline_centroid = baseline_scores.mean(axis=0) if len(baseline_scores) else np.full(len(score_cols), np.nan)
        prior_change_rows = []
        prior_nights = sorted(data.loc[data["cage_id"].eq(cage_id), "night_date"].dropna().unique())
        if night_date in prior_nights and prior_nights.index(night_date) > 0:
            prev_night = prior_nights[prior_nights.index(night_date) - 1]
            current = group.set_index("mouse_id")[score_cols]
            previous = data.loc[
                (data["cage_id"].eq(cage_id)) & (data["night_date"].eq(prev_night)),
                ["mouse_id", *score_cols],
            ].set_index("mouse_id")
            common = current.index.intersection(previous.index)
            for mouse_id in common:
                prior_change_rows.append((current.loc[mouse_id] - previous.loc[mouse_id]).to_numpy(dtype=float))
        sync_similarity = np.nan
        if len(prior_change_rows) >= 2:
            sims = [ _cosine_similarity(a, b) for a, b in combinations(prior_change_rows, 2) ]
            sync_similarity = float(np.nanmean(sims))
        control = effect_group.loc[effect_group["treatment"].eq("control"), score_cols].to_numpy(dtype=float)
        stressed = effect_group.loc[effect_group["treatment"].eq("stressed"), score_cols].to_numpy(dtype=float)
        stressed_control_separation = (
            float(np.linalg.norm(stressed.mean(axis=0) - control.mean(axis=0)))
            if len(control) and len(stressed)
            else np.nan
        )
        control_baseline = data.loc[
            (data["cage_id"].eq(cage_id))
            & (data["night_date"] < night_date)
            & (data["treatment"].eq("control"))
            & (data["effect_eligible"]),
            score_cols,
        ].to_numpy(dtype=float)
        control_response = (
            float(np.linalg.norm(control.mean(axis=0) - control_baseline.mean(axis=0)))
            if len(control) and len(control_baseline)
            else np.nan
        )
        active = group[active_cols].sum(axis=1, numeric_only=True).to_numpy(dtype=float) if active_cols else np.zeros(len(group))
        passive = group[passive_cols].sum(axis=1, numeric_only=True).to_numpy(dtype=float) if passive_cols else np.zeros(len(group))
        denom = active + passive
        imbalance = np.where(denom > 0, (active - passive) / denom, np.nan)
        rows.append(
            {
                "cage_id": cage_id,
                "night_date": night_date,
                "relative_night": int(group["relative_night"].median()),
                "stress_aligned_night": int(group["stress_aligned_night"].median())
                if group["stress_aligned_night"].notna().any()
                else np.nan,
                "phase": group["phase"].mode().iloc[0],
                "mouse_count": int(group["mouse_id"].nunique()),
                "projection_rows": int(len(group)),
                "effect_rows": int(group["effect_eligible"].sum()),
                "mean_id_1": float(centroid[0]) if len(score_cols) > 0 else np.nan,
                "mean_id_2": float(centroid[1]) if len(score_cols) > 1 else np.nan,
                "mean_identity_composite": float(group["identity_composite"].mean()),
                "dispersion": float(np.mean(np.linalg.norm(scores - centroid, axis=1))) if len(scores) else np.nan,
                "pairwise_distance": _pairwise_distance(scores),
                "baseline_centroid_distance": float(np.linalg.norm(centroid - baseline_centroid)) if len(baseline_scores) else np.nan,
                "stressed_control_separation": stressed_control_separation,
                "active_passive_imbalance": float(np.nanmean(imbalance)),
                "control_response": control_response,
                "synchronization_similarity": sync_similarity,
            }
        )
    return pd.DataFrame(rows).sort_values(["cage_id", "night_date"]).reset_index(drop=True)


def _paired_cage_change_in_change(
    frame: pd.DataFrame,
    response: str,
    seed: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
) -> dict[str, Any]:
    data = frame.loc[
        frame["effect_eligible"]
        & frame["phase"].isin(["baseline", "post_stress"])
        & frame[response].notna()
    ].copy()
    cage_effects: list[float] = []
    for cage_id, group in data.groupby("cage_id", sort=False):
        by_phase = group.groupby(["phase", "treatment"])[response].mean().unstack("treatment")
        if {"baseline", "post_stress"} <= set(by_phase.index) and {"control", "stressed"} <= set(by_phase.columns):
            change = (
                (by_phase.loc["post_stress", "stressed"] - by_phase.loc["baseline", "stressed"])
                - (by_phase.loc["post_stress", "control"] - by_phase.loc["baseline", "control"])
            )
            if np.isfinite(change):
                cage_effects.append(float(change))
    if not cage_effects:
        return {
            "response": response,
            "n_cages": 0,
            "observed_change_in_change": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "permutation_p_value": np.nan,
            "backend": "paired_cage",
        }
    effects = np.asarray(cage_effects, dtype=float)
    observed = float(np.mean(effects))
    rng = np.random.default_rng(_stable_seed(response, seed))
    bootstrap = [float(np.mean(rng.choice(effects, size=len(effects), replace=True))) for _ in range(bootstrap_iterations)]
    flips = rng.choice([-1.0, 1.0], size=(permutation_iterations, len(effects)))
    permuted = np.mean(flips * effects, axis=1)
    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1)
    return {
        "response": response,
        "n_cages": int(len(effects)),
        "observed_change_in_change": observed,
        "bootstrap_mean": float(np.mean(bootstrap)),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "permutation_p_value": float(p_value),
        "backend": "paired_cage",
    }

