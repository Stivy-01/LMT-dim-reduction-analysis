# -*- coding: utf-8 -*-
"""Fasi, tier e frame QC (ex BLOCCO 4 di thesis_analysis)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.config import (
    DERIVED_COLUMNS,
    ID_COLUMNS,
    KNOWN_TREATMENTS,
    METADATA_COLUMNS,
)
from src.analysis.merge_dataset import _stress_interval_boundary


# ==============================================================================
# BLOCCO 4 — Fasi, tier e frame QC (righe ~661-873)
# Assegna fase baseline/post-stress, tier di analisi, seleziona feature
# globalmente complete, calcola attività di gabbia e costruisce il frame QC
# con flag di qualità ed eleggibilità.
# -> futuro modulo: src/analysis/quality_control.py
# ==============================================================================
def assign_phase(row: pd.Series) -> str:
    interval_start = row.get("interval_start")
    experiment_start = row.get("experiment_start")
    experiment_end = row.get("experiment_end")
    stress_start = row.get("stress_start")
    status = str(row.get("phase_assignment_status") or "").strip()

    if pd.isna(interval_start) or pd.isna(experiment_start) or pd.isna(experiment_end):
        return "date_anomaly"
    if interval_start < experiment_start or interval_start >= (experiment_end + pd.Timedelta(days=1)):
        return "outside_metadata_window"

    phase_known_statuses = {
        "explicit_and_usable",
        "explicit_stress_date_but_no_post_rows",
        "phase_known_treatment_unknown",
        "probable_project_phase_known_treatment_unknown",
    }
    if status in phase_known_statuses:
        if pd.isna(stress_start):
            return "date_anomaly"
        stress_boundary = _stress_interval_boundary(stress_start)
        return "baseline" if interval_start < stress_boundary else "post_stress"

    if status == "inferred_from_baseline_database_boundary":
        if pd.notna(stress_start):
            stress_boundary = _stress_interval_boundary(stress_start)
            return "baseline" if interval_start < stress_boundary else "post_stress"
        return "baseline" if interval_start <= experiment_end else "post_stress"

    return "date_anomaly"


def analysis_tier(row: pd.Series) -> str:
    status = str(row.get("phase_assignment_status") or "").strip()
    treatment = str(row.get("treatment") or "").strip()
    if treatment not in KNOWN_TREATMENTS:
        return "excluded"
    if status == "explicit_and_usable":
        return "primary"
    if status == "inferred_from_baseline_database_boundary":
        return "sensitivity"
    return "excluded"


def _numeric_candidate_columns(df: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | METADATA_COLUMNS | DERIVED_COLUMNS | {"rfid", "mapping_confidence"}
    candidates: list[str] = []
    for column in df.columns:
        if column in excluded or column.startswith("flag_") or column.startswith("pca_") or column.startswith("id_"):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            candidates.append(column)
    return candidates


def _select_global_complete_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = _numeric_candidate_columns(df)
    complete: list[str] = []
    for column in candidates:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all():
            if (numeric.to_numpy(dtype=float) >= 0).all():
                complete.append(column)
    extra = [column for column in candidates if column not in complete]
    return complete, extra


def _compute_cage_night_activity(
    df: pd.DataFrame, count_features: list[str]
) -> pd.DataFrame:
    out = df.copy()
    if not count_features:
        out["cage_night_activity"] = np.nan
        out["cage_night_prior_median_activity"] = np.nan
        out["flag_partial_final_night"] = False
        return out

    out["night_date"] = out["interval_start"].dt.normalize()
    cage_night = (
        out.groupby(["cage_id", "night_date"], dropna=False)[count_features]
        .sum(min_count=1)
        .sum(axis=1)
        .reset_index(name="cage_night_activity")
    )

    cage_night["cage_night_prior_median_activity"] = np.nan
    cage_night["flag_partial_final_night"] = False
    for cage_id, group in cage_night.groupby("cage_id", sort=False):
        ordered = group.sort_values("night_date")
        if len(ordered) < 3:
            continue
        prior = ordered.iloc[:-1]["cage_night_activity"].to_numpy(dtype=float)
        final_index = ordered.index[-1]
        median_prior = float(np.nanmedian(prior))
        final_activity = float(ordered.iloc[-1]["cage_night_activity"])
        cage_night.loc[final_index, "cage_night_prior_median_activity"] = median_prior
        if np.isfinite(median_prior) and median_prior > 0 and final_activity < 0.5 * median_prior:
            cage_night.loc[final_index, "flag_partial_final_night"] = True

    out = out.merge(
        cage_night,
        on=["cage_id", "night_date"],
        how="left",
        validate="many_to_one",
    )
    out["flag_partial_final_night"] = out["flag_partial_final_night"].fillna(False).astype(bool)
    return out


def _build_qc_frame(df: pd.DataFrame, complete_features: list[str], extra_features: list[str]) -> pd.DataFrame:
    qc = df.copy()
    qc["phase"] = qc.apply(assign_phase, axis=1)
    qc["analysis_tier"] = qc.apply(analysis_tier, axis=1)
    qc["treatment_verified"] = qc["treatment"].isin(KNOWN_TREATMENTS)
    qc["relative_night"] = (
        qc["interval_start"].dt.normalize() - qc["experiment_start"].dt.normalize()
    ).dt.days + 1
    stress_boundary = pd.to_datetime(qc["stress_start"].apply(_stress_interval_boundary), errors="coerce")
    stress_day_delta = (
        qc["interval_start"].dt.normalize() - stress_boundary.dt.normalize()
    ).dt.days
    qc["stress_aligned_night"] = np.where(
        qc["interval_start"].ge(stress_boundary),
        stress_day_delta + 1,
        stress_day_delta,
    )
    qc.loc[stress_boundary.isna() | qc["interval_start"].isna(), "stress_aligned_night"] = np.nan
    qc["flag_out_of_window"] = qc["phase"].eq("outside_metadata_window")
    qc["flag_date_anomaly"] = qc["phase"].eq("date_anomaly")
    qc["flag_manual_exclusion"] = qc.get(
        "manual_exclusion_applied",
        pd.Series(False, index=qc.index),
    ).fillna(False).astype(bool)
    protocol_nights = {-3, -2, -1, 1, 2, 3}
    qc["flag_out_of_protocol_window"] = (
        qc["phase"].isin(["baseline", "post_stress"])
        & ~qc["stress_aligned_night"].isin(protocol_nights)
    )

    feature_matrix = qc[complete_features].apply(pd.to_numeric, errors="coerce")
    extra_matrix = qc[extra_features].apply(pd.to_numeric, errors="coerce") if extra_features else pd.DataFrame(index=qc.index)
    qc["flag_structural_missingness"] = (
        extra_matrix.isna().any(axis=1) if not extra_matrix.empty else False
    )
    qc["complete_feature_count"] = feature_matrix.notna().sum(axis=1)
    qc["complete_feature_fraction"] = (
        qc["complete_feature_count"] / len(complete_features) if complete_features else 0.0
    )
    qc = _compute_cage_night_activity(qc, [c for c in complete_features if "count" in c.lower()])
    valid_window = (
        qc["cage_id"].notna()
        & qc["phase"].isin(["baseline", "post_stress"])
        & ~qc["flag_out_of_window"]
        & ~qc["flag_date_anomaly"]
        & ~qc["flag_manual_exclusion"]
    )
    first_night = qc.loc[valid_window].groupby("cage_id", dropna=False)["interval_start"].transform("min")
    qc["flag_first_recorded_night"] = False
    qc.loc[valid_window, "flag_first_recorded_night"] = qc.loc[valid_window, "interval_start"].eq(first_night)
    qc["projection_eligible"] = (
        qc["cage_id"].notna()
        & qc["phase"].isin(["baseline", "post_stress"])
        & ~qc["flag_out_of_window"]
        & ~qc["flag_date_anomaly"]
        & ~qc["flag_partial_final_night"]
        & ~qc["flag_first_recorded_night"]
        & ~qc["flag_manual_exclusion"]
        & ~qc["flag_out_of_protocol_window"]
        & qc[complete_features].notna().all(axis=1)
    )
    qc["effect_eligible"] = (
        qc["projection_eligible"]
        & qc["analysis_tier"].eq("primary")
        & qc["treatment_verified"]
    )
    qc["sensitivity_effect_eligible"] = (
        qc["projection_eligible"]
        & qc["analysis_tier"].isin(["primary", "sensitivity"])
        & qc["treatment_verified"]
    )
    qc["analysis_eligible"] = qc["effect_eligible"]
    flag_columns = [
        "flag_structural_missingness",
        "flag_out_of_window",
        "flag_date_anomaly",
        "flag_partial_final_night",
        "flag_first_recorded_night",
        "flag_manual_exclusion",
        "flag_out_of_protocol_window",
    ]
    qc["quality_reason"] = qc.apply(
        lambda row: ";".join(
            column.removeprefix("flag_")
            for column in flag_columns
            if bool(row[column])
        )
        or "ok",
        axis=1,
    )
    qc["quality_status"] = np.where(
        qc["projection_eligible"],
        "usable",
        "excluded_or_sensitivity_only",
    )
    return qc


def _log1p_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.log1p(df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))

