# -*- coding: utf-8 -*-
"""Merge dataset + correzioni date manuali (ex BLOCCO 3 di thesis_analysis)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.config import DEFAULT_DATE_CORRECTIONS
from src.analysis.recap_metadata import load_metadata_enriched
from src.analysis.text_parsing import _normalize_mouse_id


# ==============================================================================
# BLOCCO 3 — Merge dataset + correzioni date manuali (righe ~550-660)
# Unisce behavior CSV + metadati + recap + correzioni date manuali;
# applica esclusioni e allinea le notti stress.
# -> futuro modulo: src/analysis/merge_dataset.py
# ==============================================================================
def load_merged_dataset(
    input_path: Path,
    metadata_path: Path,
    reconstruction_path: Path | None = None,
    recap_path: Path | None = None,
    date_corrections_path: Path | None = DEFAULT_DATE_CORRECTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior = pd.read_csv(input_path)
    behavior["mouse_id"] = _normalize_mouse_id(behavior["mouse_id"])
    behavior["interval_start"] = pd.to_datetime(behavior["interval_start"], errors="coerce")

    metadata = load_metadata_enriched(metadata_path, reconstruction_path, recap_path, behavior)
    merged = behavior.merge(metadata, on="mouse_id", how="left", validate="many_to_one")
    merged = _apply_manual_date_corrections(merged, date_corrections_path)
    merged = merged.sort_values(["mouse_id", "interval_start"], na_position="last").reset_index(
        drop=True
    )
    return merged, metadata


def _is_blank(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _read_manual_date_corrections(path: Path | None) -> pd.DataFrame:
    required_columns = [
        "enabled",
        "cage_id",
        "mouse_id",
        "original_date",
        "corrected_date",
        "correction_source",
        "correction_reason",
    ]
    columns = required_columns + ["force_exclude", "exclusion_reason"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)

    corrections = pd.read_csv(path, dtype={"mouse_id": "string"})
    missing = set(required_columns) - set(corrections.columns)
    if missing:
        raise ValueError(f"Date correction file {path} is missing columns: {sorted(missing)}")
    if "force_exclude" not in corrections.columns:
        corrections["force_exclude"] = False
    if "exclusion_reason" not in corrections.columns:
        corrections["exclusion_reason"] = ""
    enabled = corrections["enabled"].fillna(True).astype(str).str.strip().str.lower()
    corrections = corrections.loc[~enabled.isin({"false", "0", "no", "n"})].copy()
    if corrections.empty:
        return corrections

    corrections["original_date"] = pd.to_datetime(
        corrections["original_date"], errors="coerce"
    ).dt.normalize()
    corrections["corrected_date"] = pd.to_datetime(
        corrections["corrected_date"], errors="coerce"
    ).dt.normalize()
    invalid = corrections["original_date"].isna() | corrections["corrected_date"].isna()
    if invalid.any():
        bad_rows = corrections.index[invalid].tolist()
        raise ValueError(f"Date correction file {path} has invalid dates at rows: {bad_rows}")
    corrections["force_exclude"] = corrections["force_exclude"].fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )
    return corrections


def _apply_manual_date_corrections(
    merged: pd.DataFrame,
    corrections_path: Path | None = DEFAULT_DATE_CORRECTIONS,
) -> pd.DataFrame:
    out = merged.copy()
    out["interval_start_original"] = out["interval_start"]
    out["date_correction_applied"] = False
    out["date_correction_source"] = ""
    out["date_correction_reason"] = ""
    out["manual_exclusion_applied"] = False
    out["manual_exclusion_reason"] = ""

    corrections = _read_manual_date_corrections(corrections_path)
    if corrections.empty:
        return out

    original_dates = out["interval_start_original"].dt.normalize()
    for correction in corrections.itertuples(index=False):
        mask = out["interval_start_original"].notna() & original_dates.eq(correction.original_date)
        if not _is_blank(correction.cage_id):
            mask &= out["cage_id"].astype(str).eq(str(correction.cage_id).strip())
        if not _is_blank(correction.mouse_id):
            mouse_id = int(float(str(correction.mouse_id).strip()))
            mask &= out["mouse_id"].eq(mouse_id)
        if not mask.any():
            continue

        time_of_day = out.loc[mask, "interval_start_original"] - original_dates.loc[mask]
        out.loc[mask, "interval_start"] = correction.corrected_date + time_of_day
        out.loc[mask, "date_correction_applied"] = True
        out.loc[mask, "date_correction_source"] = str(correction.correction_source or "").strip()
        out.loc[mask, "date_correction_reason"] = str(correction.correction_reason or "").strip()
        if bool(correction.force_exclude):
            out.loc[mask, "manual_exclusion_applied"] = True
            out.loc[mask, "manual_exclusion_reason"] = str(correction.exclusion_reason or "").strip()
    return out


def _stress_interval_boundary(stress_start: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(stress_start):
        return pd.NaT
    return pd.Timestamp(stress_start).floor("D") + pd.Timedelta(hours=19)

