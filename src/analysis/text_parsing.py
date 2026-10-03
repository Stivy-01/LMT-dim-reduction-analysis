# -*- coding: utf-8 -*-
"""Utilità di parsing testo/date/RFID (ex BLOCCO 1 di thesis_analysis)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.config import DEFAULT_SEED


def _normalize_mouse_id(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)
    return values


def _stable_seed(text: str, base: int = DEFAULT_SEED) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate(text))


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _date_tokens(value: Any) -> list[tuple[int, int]]:
    tokens: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*[./]\s*(\d{1,2})(?!\d)", _clean_text(value)):
        day, month = int(match.group(1)), int(match.group(2))
        if 1 <= day <= 31 and 1 <= month <= 12:
            tokens.append((day, month))
    return tokens


def _explicit_year_from_text(value: Any) -> int | None:
    parts = _range_date_parts(value)
    if parts is None:
        return None
    year = parts.get("start_year") or parts.get("end_year")
    if year is None:
        return None
    if year < 100:
        year += 2000
    return year


def _normalize_year(value: int | None) -> int | None:
    if value is None:
        return None
    return value + 2000 if value < 100 else value


def _range_date_parts(value: Any) -> dict[str, int | None] | None:
    text = _clean_text(value)
    first = re.search(
        r"(?<!\d)(\d{1,2})\s*[./]\s*(\d{1,2})(?:\s*[./]\s*(\d{2,4})(?!\s*[./]\s*\d))?(?!\d)",
        text,
    )
    if not first:
        return None
    start_day, start_month = int(first.group(1)), int(first.group(2))
    start_year = int(first.group(3)) if first.group(3) else None
    if not (1 <= start_day <= 31 and 1 <= start_month <= 12):
        return None
    tail = text[first.end() :]
    second = re.search(
        r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})(?:\s*[./]\s*(\d{2,4})(?!\s*[./]\s*\d))?(?!\d)",
        tail,
    )
    if second:
        end_day, end_month = int(second.group(1)), int(second.group(2))
        end_year = int(second.group(3)) if second.group(3) else None
        if not (1 <= end_day <= 31 and 1 <= end_month <= 12):
            end_day, end_month, end_year = start_day, start_month, start_year
    else:
        end_day, end_month, end_year = start_day, start_month, start_year
    return {
        "start_day": start_day,
        "start_month": start_month,
        "start_year": start_year,
        "end_day": end_day,
        "end_month": end_month,
        "end_year": end_year,
    }


def _experiment_range_for_year(value: Any, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    parts = _range_date_parts(value)
    if parts is None:
        return None
    start_day = int(parts["start_day"])
    start_month = int(parts["start_month"])
    end_day = int(parts["end_day"])
    end_month = int(parts["end_month"])
    start_year = int(_normalize_year(parts["start_year"]) or year)
    end_year = int(_normalize_year(parts["end_year"]) or start_year)
    if parts["end_year"] is None and end_month < start_month:
        end_year = start_year + 1
    try:
        return (
            pd.Timestamp(year=start_year, month=start_month, day=start_day),
            pd.Timestamp(year=end_year, month=end_month, day=end_day),
        )
    except ValueError:
        return None


def _choose_experiment_range(value: Any, observed_dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    explicit_year = _explicit_year_from_text(value)
    candidate_years: set[int] = set()
    if explicit_year is not None:
        candidate_years.add(explicit_year)
    if observed_dates.notna().any():
        years = observed_dates.dt.year.dropna().astype(int)
        for year in years.unique():
            candidate_years.update({year - 1, year, year + 1})
    candidate_years.update({2024, 2025})

    best: tuple[int, tuple[pd.Timestamp, pd.Timestamp]] | None = None
    for year in sorted(candidate_years):
        candidate = _experiment_range_for_year(value, year)
        if candidate is None:
            continue
        start, end = candidate
        if observed_dates.notna().any():
            normalized = observed_dates.dt.normalize()
            score = int(((normalized >= start) & (normalized <= end)).sum())
        else:
            score = 0
        if explicit_year is not None and year == explicit_year:
            score += 1000
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _parse_recap_stress_datetime(value: Any, experiment_start: pd.Timestamp | None) -> pd.Timestamp | None:
    text = _clean_text(value)
    if not text or experiment_start is None or pd.isna(experiment_start):
        return None
    date_match: re.Match[str] | None = None
    day = month = None
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*[./]\s*(\d{1,2})(?!\d)", text):
        candidate_day, candidate_month = int(match.group(1)), int(match.group(2))
        if 1 <= candidate_day <= 31 and 1 <= candidate_month <= 12:
            date_match = match
            day, month = candidate_day, candidate_month
            break
    if date_match is None or day is None or month is None:
        return None

    year = int(experiment_start.year)
    if month < int(experiment_start.month) - 6:
        year += 1
    hour = 19
    minute = 0
    after_date = text[date_match.end() :]
    for match in re.finditer(r"(?<!\d)(\d{1,2})(?:\s*[.:]\s*(\d{2}))?(?!\d)", after_date):
        candidate_hour = int(match.group(1))
        candidate_minute = int(match.group(2) or 0)
        if 0 <= candidate_hour <= 23 and 0 <= candidate_minute <= 59:
            hour, minute = candidate_hour, candidate_minute
            break
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    except ValueError:
        return None


def _normalize_rfid(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return str(int(numeric))
    return text


def _dates_corrected_to_experiment_window(
    dates: pd.Series,
    experiment_start: pd.Timestamp | None,
    experiment_end: pd.Timestamp | None,
) -> pd.Series:
    corrected = dates.copy()
    if experiment_start is None or experiment_end is None or pd.isna(experiment_start) or pd.isna(experiment_end):
        return corrected
    mask = (
        corrected.notna()
        & (
            (corrected < experiment_start)
            | (corrected >= experiment_end + pd.Timedelta(days=1))
        )
    )
    if not mask.any():
        return corrected
    shifted = corrected.loc[mask] + pd.DateOffset(years=1)
    in_window = (shifted >= experiment_start) & (shifted < experiment_end + pd.Timedelta(days=1))
    corrected.loc[shifted.loc[in_window].index] = shifted.loc[in_window]
    return corrected


def _read_reconstruction_map(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["mouse_id", "rfid", "mapping_confidence"])
    table = pd.read_csv(path)
    lower = {column.lower(): column for column in table.columns}
    required = ["rfid", "assigned_candidate_id", "assignment_classification"]
    missing = [name for name in required if name not in lower]
    if missing:
        raise ValueError(
            "Reconstruction CSV must contain columns rfid, assigned_candidate_id, assignment_classification."
        )
    table = table.rename(
        columns={
            lower["assigned_candidate_id"]: "mouse_id",
            lower["assignment_classification"]: "mapping_confidence",
            lower["rfid"]: "rfid",
        }
    )
    table["mouse_id"] = _normalize_mouse_id(table["mouse_id"])
    table["rfid"] = table["rfid"].astype(str)
    table = table[["mouse_id", "rfid", "mapping_confidence"]].drop_duplicates(
        subset=["mouse_id"], keep="first"
    )
    return table
