# -*- coding: utf-8 -*-
"""Configurazione e costanti della pipeline tesi (ex BLOCCO 0 di thesis_analysis)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "behavior_stats_intervals_to_analize"
    / "merged_analysis_behavior_stats_intervals.csv"
)
DEFAULT_METADATA = PROJECT_ROOT / "data" / "analysis_metadata.csv"
DEFAULT_DATE_CORRECTIONS = PROJECT_ROOT / "data" / "manual_date_corrections.csv"
DEFAULT_OUTPUT_ROOT = Path(r"D:\lmt_thesis_analysis\runs")
DEFAULT_SEED = 20240612

ID_COLUMNS = {"mouse_id", "interval_start"}
METADATA_COLUMNS = {
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
    "rfid_from_recap",
    "recap_data_range",
    "recap_stress_raw",
    "skip_first_night",
    "behavior_rows_present",
}
DERIVED_COLUMNS = {
    "phase",
    "analysis_tier",
    "analysis_eligible",
    "projection_eligible",
    "effect_eligible",
    "sensitivity_effect_eligible",
    "treatment_verified",
    "flag_structural_missingness",
    "flag_out_of_window",
    "flag_date_anomaly",
    "flag_partial_final_night",
    "flag_first_recorded_night",
    "flag_manual_exclusion",
    "flag_out_of_protocol_window",
    "date_correction_applied",
    "date_correction_source",
    "date_correction_reason",
    "manual_exclusion_applied",
    "manual_exclusion_reason",
    "interval_start_original",
    "complete_feature_count",
    "complete_feature_fraction",
    "cage_night_activity",
    "cage_night_prior_median_activity",
    "identity_composite",
    "stress_aligned_night",
}

KNOWN_TREATMENTS = {"control", "stressed"}


@dataclass(frozen=True)
class ThesisAnalysisConfig:
    input_path: Path = DEFAULT_INPUT
    metadata_path: Path = DEFAULT_METADATA
    recap_path: Path | None = None
    rfid_map_path: Path | None = None
    date_corrections_path: Path | None = DEFAULT_DATE_CORRECTIONS
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str | None = None
    use_statsmodels: bool = True
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    random_seed: int = DEFAULT_SEED


@dataclass
class ThesisAnalysisResult:
    output_dir: Path
    mouse_night: pd.DataFrame
    group_night: pd.DataFrame
    feature_columns: list[str]
    projection_rows: int
    effect_rows: int
    summary: dict[str, Any]
    model_effects: pd.DataFrame
    feature_effects: pd.DataFrame
    bootstrap_permutation: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> ThesisAnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--recap", type=Path, default=None)
    parser.add_argument("--rfid-map", type=Path, default=None)
    parser.add_argument("--date-corrections", type=Path, default=DEFAULT_DATE_CORRECTIONS)
    parser.add_argument("--no-date-corrections", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-statsmodels", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    return ThesisAnalysisConfig(
        input_path=args.input,
        metadata_path=args.metadata,
        recap_path=args.recap,
        rfid_map_path=args.rfid_map,
        date_corrections_path=None if args.no_date_corrections else args.date_corrections,
        output_root=args.output_root,
        run_name=args.run_name,
        use_statsmodels=not args.no_statsmodels,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        random_seed=args.seed,
    )
