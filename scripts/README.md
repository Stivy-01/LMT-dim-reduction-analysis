# Helper scripts (`scripts/`)

Entry points and validation tools. Figure builders live in
`src/visualization/` (group entry `build_all.py`); these scripts either drive
the pipelines or implement analyses whose outputs are not committed figures.

| Script | What it does |
|---|---|
| `run_lmt_pipeline.ps1` | End-to-end wrapper: thesis analysis (`python -m src.analysis.thesis_analysis`) + Excel workbook. Defaults: recap `data/LMT RECAP ALL EXPERIMENTS.xlsx`, corrections `data/manual_date_corrections.csv`. Flags: `-Recap`, `-DateCorrections`, `-SkipWorkbook`, `-AnalysisOutputRoot`, bootstrap/permutation iterations, `-Seed`. |
| `update_recap_analysis_metadata.py` | Normalizes the external recap workbook: builds the `Analysis_Metadata` sheet from behavior + recap (usage: `python scripts/update_recap_analysis_metadata.py <recap> [--behavior ...]`). Backs up the workbook (`.bak`) before writing. |
| `build_thesis_workbook.mjs` | Builds `LMT_thesis_analysis.xlsx` from a run directory (run manifest + tables + `figures/`). Needs Node.js; called automatically by the wrapper. |
| `build_window_comparison_report.py` | Window-method validation: compares approved (−1 vs +1) vs extended (−2/−1 vs +1/+2/+3) analysis; writes an uncommitted PDF report. Also provides `FAMILIES`, `load_inputs`, `per_mouse_pct`, imported by supplementary S8/S9/S10 and results Fig 17. Run with an explicit run dir. |
| `longitudinal_window_comparison.py` | Per-mouse ratios across nights (`POST_NIGHTS`, `PRE_NIGHTS`, `per_mouse_ratios`, `window_values`); library for the window report. |
| `build_configuration_assets.py` | Renders `svg_configurations/*.svg` → PNG pictogram configurations (needs Node.js). Default output `src/visualization/configurations/` (assets used by Fig 9 icons). Re-run only when the SVG sources change. |
