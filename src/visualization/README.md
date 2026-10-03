# Figure builders (`src/visualization`)

One module per figure; `build_all.py` rebuilds everything as a group:

```bash
python -m src.visualization.build_all
python -m src.visualization.build_all --input-dir <csv-dir> --output-dir <fig-dir>
```

Every `build(input_dir, out_dir)` defaults to `data/` → `output/`, so a fresh
clone reproduces all figures with no arguments. Each module is also runnable
standalone (`python -m src.visualization.results.fig_03_group_trajectories`).

## Layout

- `methods/` — Fig 1 (setup) e Fig 3 (manipulation), auto-disegnate con
  `draw_kit`; Fig 2 ufficiale = `figure_methods_02_repertoire_source.png`
  (artwork manuale, NON toccata dai rebuild). `pictograms.py` e' standalone
  (varianti programmate, non usate dentro Fig 1/3).
- `results/` — Fig 4–17: `fig_01`…`fig_09`, `fig_per_feature_changes`,
  `fig_11_within_cage`, `fig_17_longitudinal`, `fig_genotype`,
  `fig_group_dynamics`, `fig_captions`; shared `icons`, `figure_data`.
- `supplementary/` — S1–S10 (`fig_s01`…`fig_s10`, shared `common`),
  `s03_treatment_split`, `treatment_profiles` (2 paper-style PDFs),
  standalone bulk tools `pages.py` / `pdfs.py` (outputs not committed).
- `paper_style.py`, `figure_data.py` — style, bundles and stats shared by all.
- `data/` — all figure inputs (21 CSVs + manifest + `longitudinal_window/` for S10).
- `output/` — committed figures: `paper/`, `paper_style/`, `supplementary/`,
  `methods/`, `genotype/`, `group_dynamics/`.
- `configurations/` — 65 pictogram assets used by Fig 9 icons.

Helpers that stay in `scripts/`: `build_window_comparison_report.py`,
`longitudinal_window_comparison.py` (window-method validation, run with
explicit paths), `build_configuration_assets.py` (renders
`svg_configurations/` → PNG).

Analysis result tables not needed by any figure (bootstrap, QC report,
p-values, scores) live in `src/results/` instead of a duplicated `outputs/`
tree.
