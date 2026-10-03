# LMT RFID Reconstruction and Thesis Analysis

## Scope

The repository now contains two independent pipelines:

1. `src.reconstruction` reads raw SQLite databases only to reconstruct RFID-to-mouse identity.
2. `src.analysis.thesis_analysis` analyzes the existing merged mouse-night CSV with verified metadata.

Raw behavioral events are not appended to the thesis dataset. Stress/control labels always come from metadata and are never inferred from behavioral similarity.

## Current validated outputs

RFID reconstruction:

`D:\lmt_rfid_reconstruction\runs\reconstruction_v1_20260612_192455`

Thesis analysis:

`D:\lmt_thesis_analysis\runs\thesis_final_v2_20260612`

The known-positive validation report is:

`D:\lmt_rfid_reconstruction\known_positive_control_validation.json`

It reproduced 128/128 count cells exactly. Duration differences had a median absolute error of 0.000104 seconds and a maximum of 0.005 seconds.

## Reconstructed identities

| Project | RFID | Mouse ID | Confidence |
|---|---:|---:|---|
| WT 6 February 2025 | 001043416716 | 10132 | high_behavioral |
| WT 6 February 2025 | 001040990306 | 10135 | high_behavioral |
| WT 6 February 2025 | 001040995418 | 10136 | high_behavioral |
| WT 6 February 2025 | 001040993034 | 10137 | high_behavioral |
| WT 21-28 February 2025 | 001040993517 | 10205 | probable |
| WT 21-28 February 2025 | RFID_D | 10206 | probable |
| WT 21-28 February 2025 | 001040990573 | 10207 | probable |
| WT 21-28 February 2025 | 001040990459 | 10208 | probable |
| CD-del 21-28 February 2025 | 091078191757 | 10276 | probable |
| CD-del 21-28 February 2025 | 091078191792 | 10277 | probable |
| CD-del 21-28 February 2025 | 091078191823 | 10278 | probable |
| CD-del 21-28 February 2025 | 091078191826 | 10279 | probable |

`high_behavioral` is not equivalent to independently verified identity. The February 21 mappings are exploratory because their behavioral assignment margins and empirical null tests were insufficient for the stricter category. Groups 9621-9624 and 9905-9908 remain unresolved.

## Analysis rules

- Unit: one mouse during one 12-hour active interval, 19:00-07:00.
- Primary feature set: 126 features complete across all rows.
- Counts and durations: `log1p`.
- Scaling and PCA axes: fitted only on eligible baseline nights.
- Identity domains: regularized LDA fitted only on baseline.
- No mean imputation.
- Partial final nights, out-of-window rows, and date anomalies are excluded from primary inference.
- Primary stress inference uses only metadata-explicit cages.
- Inferred phase boundaries are reported separately as sensitivity analysis.
- Feature-level tests use Benjamini-Hochberg correction.
- Cage-level change-in-change uses cage bootstrap and sign-flip permutation.

The current primary inferential sample contains 40 mouse-night rows from two metadata-explicit cages. This is too small for strong causal claims. Confidence intervals and permutation results are the main interpretation, not isolated nominal p-values.

## Run commands

Normalize the recap workbook metadata before running thesis analysis:

```powershell
python .\scripts\update_recap_analysis_metadata.py `
  "C:\Users\andre\Downloads\LMT RECAP ALL EXPERIMENTS.xlsx"
```

This creates or updates the `Analysis_Metadata` sheet inside the recap workbook. The analysis pipeline reads that sheet directly when `--recap` is supplied, so downstream runs do not depend on the free-form layout of the human-editable sheets.

`CD1` rows are excluded by default because they are out of thesis scope. Use `--include-cd1` only if those animals become part of a later analysis.

Known upstream date-processing errors are stored in:

```text
data/manual_date_corrections.csv
```

The raw behavioral CSV is not modified. The analysis applies those corrections only to the operational dataset and records the original date, corrected date, source, and reason in `mouse_night_analysis.csv`, `quality_control.csv`, and `date_corrections_applied.csv`. If a corrected night is known to be an extra or invalid acquisition, the same table can set `force_exclude=true`; the row remains visible but is removed from projection/effect/group analyses by `flag_manual_exclusion`.

Run everything:

```powershell
.\scripts\run_lmt_pipeline.ps1 `
  -Recap "C:\Users\andre\Downloads\LMT RECAP ALL EXPERIMENTS.xlsx"
```

Reuse the existing RFID mapping and rerun only analysis/workbook:

```powershell
.\scripts\run_lmt_pipeline.ps1 `
  -SkipRfid `
  -RfidMapping "D:\lmt_rfid_reconstruction\runs\reconstruction_v1_20260612_192455\rfid_id_mapping.csv" `
  -Recap "C:\Users\andre\Downloads\LMT RECAP ALL EXPERIMENTS.xlsx"
```

Run analysis directly:

```powershell
$env:LMT_DEBUG = "False"
python -m src.analysis.thesis_analysis `
  --recap "C:\Users\andre\Downloads\LMT RECAP ALL EXPERIMENTS.xlsx" `
  --date-corrections "C:\Users\andre\Desktop\iit\lmt toolkit-cleaned\data\manual_date_corrections.csv" `
  --rfid-map "D:\lmt_rfid_reconstruction\runs\reconstruction_v1_20260612_192455\rfid_id_mapping.csv" `
  --output-root "D:\lmt_thesis_analysis\runs" `
  --bootstrap-iterations 1000 `
  --permutation-iterations 10000 `
  --seed 20240612
```

Build exploratory genotype figures from an existing thesis run:

```powershell
python .\scripts\build_genotype_figures.py `
  "C:\Users\andre\Desktop\iit\lmt toolkit-cleaned\outputs\thesis_full_runs\thesis_analysis_20260620_013502"
```

These figures use paired `TG - WT` contrasts within the same cage, night, and treatment. They are intended for exploratory genotype interpretation, especially separating `16p` and `CD del` rather than pooling them as a single non-WT group.

Build exploratory group-dynamics figures using the quartetto as the experimental unit:

```powershell
python .\scripts\build_group_dynamics_figures.py `
  "C:\Users\andre\Desktop\iit\lmt toolkit-cleaned\outputs\thesis_full_runs\thesis_analysis_20260620_013502"
```

These figures compare `16p`, `CD del`, and `WT` at cage-night level. They use only cages with both baseline and post-stress group metrics for stress-response summaries, so the current complete-cage counts are small: `16p n=2`, `CD del n=1`, `WT n=6`.

Validate a known processed database:

```powershell
python -m src.reconstruction.validate_known `
  --output "D:\lmt_rfid_reconstruction\known_positive_control_validation.json"
```

## Deliverables

Each reconstruction run contains:

- `rfid_id_mapping.csv`
- `mapping_cost_matrices.csv`
- `mapping_validation_report.html`
- `source_manifest.json`
- resumable checkpoints

Each analysis run contains:

- `mouse_night_analysis.csv`
- `group_night_analysis.csv`
- `metadata_enriched.csv`
- `date_corrections_applied.csv`
- `quality_control.csv`
- PCA and identity loadings/scores
- dimension, feature, bootstrap, and permutation tables
- seven figure families in PNG, SVG, and PDF
- `thesis_analysis_report.html`
- `LMT_thesis_analysis.xlsx`

The workbook keeps original values, QC flags, transformed analysis outputs, model effects, resampling results, and figures in separate sheets.

## Interpretation boundary

The CSV does not retain the partner identity for social interactions. It supports individual and whole-quartet summaries, but not mouse-to-mouse network claims. Without fully unmanipulated cages, changes in control mice should be described as associated with group manipulation rather than as definitive causal evidence of social stress.
