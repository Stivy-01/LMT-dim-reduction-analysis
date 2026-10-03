from __future__ import annotations

import argparse
import json
import logging
import re
import stat
import textwrap
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import t as student_t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.getLogger("matplotlib").setLevel(logging.WARNING)

try:  # Optional dependency.
    import statsmodels.api as sm  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover
    sm = None  # type: ignore
    smf = None  # type: ignore
    STATSMODELS_AVAILABLE = False


# ==============================================================================
# BLOCCO 0 — Configurazione e costanti (righe ~41-161)
# Path di default, set di colonne (ID/METADATA/DERIVED), dataclass
# ThesisAnalysisConfig / ThesisAnalysisResult, parse_args.
# -> futuro modulo: src/analysis/config.py
# ==============================================================================
# BLOCCO 0 estratto in src/analysis/config.py (re-esportato per compatibilità).
from src.analysis.config import (
    DEFAULT_DATE_CORRECTIONS,
    DEFAULT_INPUT,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DERIVED_COLUMNS,
    ID_COLUMNS,
    KNOWN_TREATMENTS,
    METADATA_COLUMNS,
    PROJECT_ROOT,
    ThesisAnalysisConfig,
    ThesisAnalysisResult,
    parse_args,
)


# BLOCCO 1 estratto in src/analysis/text_parsing.py (re-esportato per compatibilità).
from src.analysis.text_parsing import (
    _choose_experiment_range,
    _clean_text,
    _date_tokens,
    _dates_corrected_to_experiment_window,
    _experiment_range_for_year,
    _explicit_year_from_text,
    _normalize_mouse_id,
    _normalize_rfid,
    _normalize_year,
    _parse_recap_stress_datetime,
    _range_date_parts,
    _read_reconstruction_map,
    _stable_seed,
)


# BLOCCO 2 estratto in src/analysis/recap_metadata.py (re-esportato per compatibilita).
from src.analysis.recap_metadata import (
    load_recap_metadata,
    load_metadata_enriched,
)


# BLOCCO 3 estratto in src/analysis/merge_dataset.py (re-esportato per compatibilita).
from src.analysis.merge_dataset import (
    load_merged_dataset,
    _is_blank,
    _read_manual_date_corrections,
    _apply_manual_date_corrections,
    _stress_interval_boundary,
)


# BLOCCO 4 estratto in src/analysis/quality_control.py (re-esportato per compatibilita).
from src.analysis.quality_control import (
    assign_phase,
    analysis_tier,
    _numeric_candidate_columns,
    _select_global_complete_features,
    _compute_cage_night_activity,
    _build_qc_frame,
    _log1p_matrix,
)


# BLOCCO 5 estratto in src/analysis/projection.py (re-esportato per compatibilita).
from src.analysis.projection import (
    RegularizedLDAProjection,
    _temporal_validation,
    _fit_numpy_terms,
    _map_statsmodels_terms,
)


# BLOCCO 6 estratto in src/analysis/effects.py (re-esportato per compatibilita).
from src.analysis.effects import (
    _bh_adjust,
    _fit_model_effects,
    _fit_feature_effects,
    _pairwise_distance,
    _cosine_similarity,
)


# BLOCCO 7 estratto in src/analysis/group_stats.py (re-esportato per compatibilita).
from src.analysis.group_stats import (
    _compute_group_metrics,
    _paired_cage_change_in_change,
)


# BLOCCO 8 estratto in src/analysis/fit_models.py (re-esportato per compatibilita).
from src.analysis.fit_models import (
    _fit_projection_models,
    _merge_scores,
)


# BLOCCO 9 estratto in src/analysis/diagnostics.py (re-esportato per compatibilita).
from src.analysis.diagnostics import (
    _figure_bundle,
    _figure_timeline_qc,
    _clean_feature_label,
    _figure_missingness_heatmap,
    _figure_trajectories,
    _figure_projection_arrows,
    _figure_forest_plot,
    _figure_group_dispersion,
    _figure_feature_heatmap,
)


# BLOCCO 10 estratto in src/analysis/reporting.py (re-esportato per compatibilità).
from src.analysis.reporting import (
    _make_run_directory,
    _render_report,
    _save_csv,
    _set_readonly,
)


# ==============================================================================
# BLOCCO 11 — Orchestratore pipeline (righe ~1948-2242)
# Chiama i blocchi 2-10 in ordine: merge -> QC -> proiezione -> effetti ->
# bootstrap -> salvataggi -> figure -> report -> manifest. Ultimo a essere
# smontato: diventerà un main sottile che importa i moduli dei blocchi.
# ==============================================================================
def build_pipeline(config: ThesisAnalysisConfig) -> ThesisAnalysisResult:
    merged, metadata_enriched = load_merged_dataset(
        config.input_path,
        config.metadata_path,
        config.rfid_map_path,
        config.recap_path,
        config.date_corrections_path,
    )
    complete_features, extra_features = _select_global_complete_features(merged)
    if not complete_features:
        raise ValueError("No globally complete features were found.")

    qc = _build_qc_frame(merged, complete_features, extra_features)
    projection_rows = int(qc["projection_eligible"].sum())
    effect_rows = int(qc["effect_eligible"].sum())

    pca_scores, pca_loadings, identity_loadings, pca_model, scaler, lda_model, validation = _fit_projection_models(
        qc, complete_features, config.use_statsmodels
    )
    score_merge_cols = [
        "mouse_id",
        "interval_start",
        "cage_id",
        "phase",
        "analysis_tier",
        "projection_eligible",
        "effect_eligible",
    ]
    qc = qc.merge(pca_scores, on=score_merge_cols, how="left", validate="one_to_one")

    projection = qc.loc[qc["projection_eligible"]].copy()
    baseline_projection = projection.loc[projection["phase"].eq("baseline")].copy()
    baseline_scaled = scaler.transform(_log1p_matrix(baseline_projection, complete_features))
    baseline_lda = RegularizedLDAProjection().fit(baseline_scaled, baseline_projection["mouse_id"].to_numpy())
    projection_scores = projection[score_merge_cols].copy()
    transformed = baseline_lda.transform(scaler.transform(_log1p_matrix(projection, complete_features)))
    id_columns = [f"id_{index + 1}" for index in range(transformed.shape[1])]
    for index, column in enumerate(id_columns):
        projection_scores[column] = transformed[:, index]
    projection_scores["identity_composite"] = (
        projection_scores[[c for c in id_columns[:2] if c in projection_scores.columns]].mean(axis=1)
        if id_columns
        else np.nan
    )
    identity_loadings = []
    if baseline_lda.components_ is not None:
        for index, component in enumerate(baseline_lda.components_):
            for feature, loading in zip(complete_features, component):
                identity_loadings.append(
                    {
                        "component": f"id_{index + 1}",
                        "feature": feature,
                        "loading": float(loading),
                        "eigenvalue": float(baseline_lda.eigenvalues_[index]) if baseline_lda.eigenvalues_ is not None else np.nan,
                    }
                )
    identity_loadings_df = pd.DataFrame(identity_loadings)

    qc = qc.merge(projection_scores, on=score_merge_cols, how="left", validate="one_to_one")
    qc["identity_composite"] = qc["identity_composite"].astype(float)

    # Put the primary analysis columns first and keep selected complete features.
    output_columns = (
        [
            "mouse_id",
            "rfid",
            "rfid_from_recap",
            "mapping_confidence",
            "interval_start",
            "interval_start_original",
            "date_correction_applied",
            "date_correction_source",
            "date_correction_reason",
            "manual_exclusion_applied",
            "manual_exclusion_reason",
            "cage_id",
            "source_sheet",
            "sex",
            "genotype",
            "strain",
            "night_date",
            "relative_night",
            "stress_aligned_night",
            "phase",
            "phase_assignment_status",
            "recap_data_range",
            "recap_stress_raw",
            "analysis_tier",
            "treatment",
            "treatment_verified",
            "analysis_eligible",
            "projection_eligible",
            "effect_eligible",
            "sensitivity_effect_eligible",
            "quality_status",
            "quality_reason",
        ]
        + [
            "flag_structural_missingness",
            "flag_out_of_window",
            "flag_date_anomaly",
            "flag_partial_final_night",
            "flag_first_recorded_night",
            "flag_manual_exclusion",
            "flag_out_of_protocol_window",
        ]
        + ["complete_feature_count", "complete_feature_fraction", "cage_night_activity", "cage_night_prior_median_activity"]
        + complete_features
        + [c for c in qc.columns if c.startswith("pca_") or c.startswith("id_") or c == "identity_composite"]
    )
    output_columns = [column for column in output_columns if column in qc.columns]
    mouse_night = qc[output_columns].copy()
    group_night = _compute_group_metrics(qc, id_columns[:2] if len(id_columns) >= 2 else id_columns[:1], complete_features)
    date_corrections_applied = qc.loc[
        qc["date_correction_applied"],
        [
            column
            for column in [
                "cage_id",
                "mouse_id",
                "interval_start_original",
                "interval_start",
                "phase",
                "relative_night",
                "date_correction_source",
                "date_correction_reason",
                "flag_manual_exclusion",
                "manual_exclusion_reason",
            ]
            if column in qc.columns
        ],
    ].drop_duplicates()

    responses = [
        column
        for column in ["id_1", "id_2", "identity_composite"]
        if column in qc.columns
    ]
    analysis_frames: list[tuple[str, pd.DataFrame]] = [("primary", qc)]

    sensitivity_qc = qc.copy()
    sensitivity_qc["effect_eligible"] = sensitivity_qc[
        "sensitivity_effect_eligible"
    ]
    if not sensitivity_qc["effect_eligible"].equals(qc["effect_eligible"]):
        analysis_frames.append(("phase_boundary_sensitivity", sensitivity_qc))

    outlier_qc = qc.copy()
    if outlier_qc["cage_id"].eq("wt_10132").any():
        outlier_qc["effect_eligible"] = (
            outlier_qc["effect_eligible"] & ~outlier_qc["cage_id"].eq("wt_10132")
        )
        if int(outlier_qc["effect_eligible"].sum()) < int(qc["effect_eligible"].sum()):
            analysis_frames.append(("exclude_wt_10132", outlier_qc))

    model_tables: list[pd.DataFrame] = []
    feature_tables: list[pd.DataFrame] = []
    bootstrap_tables: list[pd.DataFrame] = []
    for scope, scope_qc in analysis_frames:
        model_table = pd.concat(
            [
                _fit_model_effects(scope_qc, response, config.use_statsmodels)
                for response in responses
            ],
            ignore_index=True,
        )
        if not model_table.empty:
            model_table.insert(0, "analysis_scope", scope)
            model_tables.append(model_table)

        feature_table = _fit_feature_effects(scope_qc, complete_features)
        if not feature_table.empty:
            feature_table.insert(0, "analysis_scope", scope)
            feature_tables.append(feature_table)

        bootstrap_table = pd.DataFrame(
            [
                _paired_cage_change_in_change(
                    scope_qc,
                    response=response,
                    seed=config.random_seed,
                    bootstrap_iterations=config.bootstrap_iterations,
                    permutation_iterations=config.permutation_iterations,
                )
                for response in responses
            ]
        )
        if not bootstrap_table.empty:
            bootstrap_table.insert(0, "analysis_scope", scope)
            bootstrap_tables.append(bootstrap_table)

    model_effects = pd.concat(model_tables, ignore_index=True) if model_tables else pd.DataFrame()
    if not model_effects.empty:
        model_effects["bh_q_value"] = model_effects.groupby(
            "analysis_scope"
        )["p_value"].transform(_bh_adjust)

    feature_effects = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    bootstrap_rows = pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame()
    if not bootstrap_rows.empty:
        bootstrap_rows["bh_q_value"] = bootstrap_rows.groupby(
            "analysis_scope"
        )["permutation_p_value"].transform(_bh_adjust)

    output_dir = _make_run_directory(config.output_root, config.run_name)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(metadata_enriched, output_dir / "metadata_enriched.csv")
    _save_csv(date_corrections_applied, output_dir / "date_corrections_applied.csv")
    _save_csv(qc, output_dir / "quality_control.csv")
    _save_csv(mouse_night, output_dir / "mouse_night_analysis.csv")
    _save_csv(group_night, output_dir / "group_night_analysis.csv")
    _save_csv(pca_scores, output_dir / "pca_scores.csv")
    _save_csv(pca_loadings, output_dir / "pca_loadings.csv")
    _save_csv(identity_loadings_df, output_dir / "identity_loadings.csv")
    _save_csv(feature_effects, output_dir / "feature_effects.csv")
    _save_csv(model_effects, output_dir / "model_effects.csv")
    _save_csv(bootstrap_rows, output_dir / "bootstrap_permutation.csv")

    _figure_timeline_qc(qc, figures_dir)
    _figure_missingness_heatmap(qc, complete_features + extra_features, figures_dir)
    _figure_trajectories(qc, figures_dir)
    _figure_projection_arrows(qc, figures_dir)
    _figure_forest_plot(feature_effects, figures_dir)
    _figure_group_dispersion(group_night, figures_dir)
    _figure_feature_heatmap(pca_loadings, identity_loadings_df, figures_dir)

    report = _render_report(
        ThesisAnalysisResult(
            output_dir=output_dir,
            mouse_night=mouse_night,
            group_night=group_night,
            feature_columns=complete_features,
            projection_rows=projection_rows,
            effect_rows=effect_rows,
            summary={
                "temporal_validation": validation,
                "pca_components": int(pca_scores.filter(regex=r"^pca_").shape[1]),
            },
            model_effects=model_effects,
            feature_effects=feature_effects,
            bootstrap_permutation=bootstrap_rows,
        )
    )

    manifest = {
        "output_dir": str(output_dir),
        "report": str(report),
        "figures_dir": str(figures_dir),
        "complete_features": complete_features,
        "projection_rows": projection_rows,
        "effect_rows": effect_rows,
        "metadata_source": str(config.recap_path or config.metadata_path),
        "date_corrections_source": str(config.date_corrections_path) if config.date_corrections_path else None,
        "date_corrections_applied_rows": int(qc["date_correction_applied"].sum()),
        "analysis_scopes": [scope for scope, _ in analysis_frames],
        "model_formula": "response ~ C(phase) * C(treatment) + relative_night; random intercept by mouse_id; cage_id variance component when available",
        "temporal_validation": validation,
        "artifacts": {
            "metadata_enriched": str(output_dir / "metadata_enriched.csv"),
            "date_corrections_applied": str(output_dir / "date_corrections_applied.csv"),
            "quality_control": str(output_dir / "quality_control.csv"),
            "mouse_night_analysis": str(output_dir / "mouse_night_analysis.csv"),
            "group_night_analysis": str(output_dir / "group_night_analysis.csv"),
            "pca_scores": str(output_dir / "pca_scores.csv"),
            "pca_loadings": str(output_dir / "pca_loadings.csv"),
            "identity_loadings": str(output_dir / "identity_loadings.csv"),
            "feature_effects": str(output_dir / "feature_effects.csv"),
            "model_effects": str(output_dir / "model_effects.csv"),
            "bootstrap_permutation": str(output_dir / "bootstrap_permutation.csv"),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _set_readonly(manifest_path)
    return ThesisAnalysisResult(
        output_dir=output_dir,
        mouse_night=mouse_night,
        group_night=group_night,
        feature_columns=complete_features,
        projection_rows=projection_rows,
        effect_rows=effect_rows,
        summary={
            "temporal_validation": validation,
            "pca_components": int(pca_scores.filter(regex=r"^pca_").shape[1]),
            "metadata_source": str(config.recap_path or config.metadata_path),
            "date_corrections_source": str(config.date_corrections_path) if config.date_corrections_path else None,
            "date_corrections_applied_rows": int(qc["date_correction_applied"].sum()),
        },
        model_effects=model_effects,
        feature_effects=feature_effects,
        bootstrap_permutation=bootstrap_rows,
    )


# ==============================================================================
# BLOCCO 12 — Entry point (righe ~2245-2260)
# run_pipeline / main / __main__. Resta qui anche dopo lo split.
# ==============================================================================
def run_pipeline(config: ThesisAnalysisConfig) -> ThesisAnalysisResult:
    return build_pipeline(config)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    result = run_pipeline(config)
    print(f"Wrote analysis to {result.output_dir}")
    print(f"Selected complete features: {len(result.feature_columns)}")
    print(f"Projection eligible rows: {result.projection_rows}")
    print(f"Effect eligible rows: {result.effect_rows}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
