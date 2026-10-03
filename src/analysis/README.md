# Thesis analysis pipeline (`src.analysis`)

This package implements the thesis analysis: it measures how chronic stress reshapes
mouse behavior and group structure in Live Mouse Tracker (LMT) data, using
dimensionality reduction (PCA + regularized LDA identity space) and
phase-by-treatment effect models.

Design principles:

- Treatment labels (control / stressed) always come from metadata, never inferred
  from behavioral similarity.
- The known outlier cage `wt_10132` is analyzed in a dedicated scope, never silently
  dropped from the primary scope.
- With few cages, confidence intervals and permutation results are more informative
  than isolated p-values (BH-FDR corrected throughout).

## Pipeline stages

`thesis_analysis.py` only orchestrates (`build_pipeline`); each stage lives in its
own module:

| Stage | Module | What it does |
|---|---|---|
| 0. Config | `config.py` | Default paths, column sets, `ThesisAnalysisConfig` / `ThesisAnalysisResult`, CLI args |
| 1. Parsing | `text_parsing.py` | Normalizes mouse IDs, recap dates, years, RFID; stable seeds |
| 2. Recap metadata | `recap_metadata.py` | Reads the recap workbook (`Analysis_Metadata` sheet) into mouse metadata |
| 3. Merge | `merge_dataset.py` | Joins behavior CSV + metadata + recap + manual date corrections |
| 4. QC | `quality_control.py` | Assigns baseline/post-stress phase, analysis tiers, eligibility flags, complete features |
| 5. Projection | `projection.py` | Regularized LDA identity space + temporal validation |
| 6. Effects | `effects.py` | Phase×treatment model/feature effects, BH correction |
| 7. Group stats | `group_stats.py` | Cage-level metrics, bootstrap + sign-flip permutation change-in-change |
| 8. Fit models | `fit_models.py` | Fits PCA/LDA on log1p features, merges scores back into QC frame |
| 9. Diagnostics | `diagnostics.py` | 7 internal QC figures (timeline, missingness, trajectories, arrows, forest, dispersion, heatmap) |
| 10. Reporting | `reporting.py` | HTML report, run directory, CSV writers |
| 11. Orchestration | `thesis_analysis.py` | `build_pipeline` + `run_pipeline` + `main` |

Analysis scopes (when the sample changes, results are exported per scope):

- `primary` — all effect-eligible rows.
- `phase_boundary_sensitivity` — sensitivity eligibility definition.
- `exclude_wt_10132` — primary sample without the outlier cage.

## Inputs

- `data/behavior_stats_intervals_to_analize/merged_analysis_behavior_stats_intervals.csv`
- `data/analysis_metadata.csv`
- `data/LMT RECAP ALL EXPERIMENTS.xlsx` (normalized into `Analysis_Metadata`)
- `data/manual_date_corrections.csv`

## Outputs (per run directory)

Tables: `mouse_night_analysis.csv`, `group_night_analysis.csv`, `pca_scores.csv`,
`pca_loadings.csv`, `identity_loadings.csv`, `model_effects.csv`, `feature_effects.csv`,
`bootstrap_permutation.csv`, `metadata_enriched.csv`, `quality_control.csv`,
`date_corrections_applied.csv`, plus `thesis_analysis_report.html` and
`run_manifest.json`.

The CSVs consumed by the publication figures are mirrored under
`src/visualization/data/` (all figure inputs live there, including the
`longitudinal_window/` values for S10); the figures themselves under
`src/visualization/output/`. See `src/visualization/README.md` for the
per-figure module layout (group entry `build_all.py`).

## Statistical methods

Deep dive: [`STATISTICS.md`](STATISTICS.md) — what was used, where it lives in the
pipeline, and why (PCA/LDA, mixed models, BH-FDR, bootstrap, sign-flip tests).

## Run it

```bash
python -m src.analysis.thesis_analysis --help
python -m src.analysis.thesis_analysis --recap "data/LMT RECAP ALL EXPERIMENTS.xlsx"
```

or via the wrapper (also builds the workbook):

```powershell
scripts/run_lmt_pipeline.ps1
```
