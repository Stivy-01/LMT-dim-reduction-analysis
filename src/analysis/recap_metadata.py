# -*- coding: utf-8 -*-
"""Caricamento metadati recap (ex BLOCCO 2 di thesis_analysis)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.text_parsing import (
    _choose_experiment_range,
    _clean_text,
    _dates_corrected_to_experiment_window,
    _normalize_mouse_id,
    _normalize_rfid,
    _parse_recap_stress_datetime,
    _read_reconstruction_map,
)


# ==============================================================================
# BLOCCO 2 — Caricamento metadati recap (righe ~384-549)
# Legge il workbook recap (foglio Analysis_Metadata normalizzato),
# arricchisce i metadati dei topi (gabbia, sesso, trattamento, genotipo,
# RFID, finestre eksperimento/stress).
# -> futuro modulo: src/analysis/recap_metadata.py
# ==============================================================================
def load_recap_metadata(
    recap_path: Path,
    behavior: pd.DataFrame,
    prefer_normalized: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    workbook = pd.ExcelFile(recap_path)
    if prefer_normalized and "Analysis_Metadata" in workbook.sheet_names:
        metadata = pd.read_excel(recap_path, sheet_name="Analysis_Metadata")
        metadata = metadata.dropna(how="all")
        required = {
            "mouse_id",
            "cage_id",
            "source_sheet",
            "sex",
            "treatment",
            "genotype",
            "strain",
            "experiment_start",
            "experiment_end",
            "stress_start",
            "phase_assignment_status",
            "notes",
        }
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(f"Analysis_Metadata sheet is missing required columns: {missing}")
        return metadata
    for sheet_name in ["16p", "wt", "CD1", "CD del"]:
        if sheet_name not in workbook.sheet_names:
            continue
        table = pd.read_excel(recap_path, sheet_name=sheet_name)
        table = table[[column for column in table.columns if not str(column).startswith("Unnamed")]].copy()
        if "ID" not in table.columns or "DATA" not in table.columns:
            continue
        stress_column = "Stress datetime" if "Stress datetime" in table.columns else "Stress time"
        table["_group"] = table["DATA"].notna().cumsum()
        for column in [
            "DATA",
            "SEX",
            "GENOTYPE",
            "STRAIN",
            "Start time",
            stress_column,
            "Recorded baseline day",
        ]:
            if column in table.columns:
                table[column] = table.groupby("_group")[column].ffill()

        for _, group in table.groupby("_group", sort=False):
            mouse_rows = group.loc[group["ID"].notna()].copy()
            if mouse_rows.empty:
                continue
            mouse_ids = _normalize_mouse_id(mouse_rows["ID"])
            numeric_mouse_ids = pd.to_numeric(mouse_ids, errors="coerce").dropna().astype(int)
            observed = behavior.loc[
                behavior["mouse_id"].isin(numeric_mouse_ids),
                "interval_start",
            ]
            first = group.iloc[0]
            experiment_range = _choose_experiment_range(first.get("DATA"), observed)
            experiment_start = experiment_end = None
            if experiment_range is not None:
                experiment_start, experiment_end = experiment_range
            stress_start = _parse_recap_stress_datetime(first.get(stress_column), experiment_start)
            cage_id = f"{re.sub(r'[^0-9a-zA-Z]+', '', sheet_name).lower()}_{int(numeric_mouse_ids.iloc[0])}" if len(numeric_mouse_ids) else f"{sheet_name}_unknown"
            behavior_present = bool(len(observed))
            corrected_observed = _dates_corrected_to_experiment_window(
                observed,
                experiment_start,
                experiment_end,
            )
            stress_night = (
                stress_start.floor("D") + pd.Timedelta(hours=19)
                if stress_start is not None and pd.notna(stress_start)
                else None
            )
            has_post_stress = bool(
                stress_night is not None
                and corrected_observed.notna().any()
                and (corrected_observed >= stress_night).any()
            )

            for _, row in mouse_rows.iterrows():
                mouse_id = int(pd.to_numeric(pd.Series([row["ID"]]), errors="coerce").iloc[0])
                treatment_raw = _clean_text(row.get("EMOTION")).upper()
                treatment = {
                    "STRESS": "stressed",
                    "NEUTRAL": "control",
                }.get(treatment_raw, "")
                phase_status = "metadata_only_behavior_not_processed"
                if behavior_present:
                    if stress_start is not None:
                        phase_status = (
                            "explicit_and_usable"
                            if has_post_stress
                            else "explicit_stress_date_but_no_post_rows"
                        )
                    else:
                        phase_status = "treatment_known_phase_not_observed"
                notes = "; ".join(
                    part
                    for part in [
                        _clean_text(row.get("NOTES")),
                        _clean_text(row.get("Unnamed: 13")),
                    ]
                    if part
                )
                rows.append(
                    {
                        "mouse_id": mouse_id,
                        "cage_id": cage_id,
                        "source_sheet": sheet_name,
                        "sex": _clean_text(row.get("SEX")) or _clean_text(first.get("SEX")),
                        "treatment": treatment,
                        "genotype": _clean_text(row.get("GENOTYPE")),
                        "strain": _clean_text(row.get("STRAIN")) if "STRAIN" in mouse_rows.columns else "",
                        "experiment_start": experiment_start,
                        "experiment_end": experiment_end,
                        "stress_start": stress_start,
                        "phase_assignment_status": phase_status,
                        "notes": notes,
                        "rfid_from_recap": _normalize_rfid(row.get("RFID")),
                        "recap_data_range": _clean_text(row.get("DATA")),
                        "recap_stress_raw": _clean_text(row.get(stress_column)),
                    }
                )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f"No mouse metadata could be parsed from recap workbook: {recap_path}")
    metadata = metadata.drop_duplicates(subset=["mouse_id"], keep="first").sort_values("mouse_id")
    return metadata.reset_index(drop=True)


def load_metadata_enriched(
    metadata_path: Path,
    reconstruction_path: Path | None,
    recap_path: Path | None = None,
    behavior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if recap_path is not None:
        if behavior is None:
            raise ValueError("behavior data is required when loading metadata from a recap workbook")
        metadata = load_recap_metadata(recap_path, behavior)
    else:
        metadata = pd.read_csv(metadata_path)
    metadata["mouse_id"] = _normalize_mouse_id(metadata["mouse_id"])
    for column in ["experiment_start", "experiment_end", "stress_start"]:
        if column in metadata.columns:
            metadata[column] = pd.to_datetime(metadata[column], errors="coerce")
    mapping = _read_reconstruction_map(reconstruction_path)
    if mapping.empty:
        recap_rfid = metadata.get("rfid_from_recap", pd.Series(pd.NA, index=metadata.index))
        if "rfid" in metadata.columns:
            metadata["rfid"] = metadata["rfid"].fillna(recap_rfid)
        else:
            metadata["rfid"] = recap_rfid
        metadata["mapping_confidence"] = pd.NA
    else:
        metadata = metadata.merge(mapping, on="mouse_id", how="left", validate="many_to_one")
        if "rfid_from_recap" in metadata.columns:
            metadata["rfid"] = metadata["rfid"].fillna(metadata["rfid_from_recap"])
    return metadata

