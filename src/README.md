# Source layout (`src/`)

End-to-end flow: raw LMT databases → analysis tables → figures.

| Package / file | Role |
|---|---|
| `behavior/` | Processors turning raw LMT events into behavior statistics (daily, 12 h night intervals, hourly). |
| `preprocessing/` | Data preprocessing: event filtering (`event_filtered.py`), social feature ratios (`feature_ratios.py`). Kept as-is from the base pipeline. |
| `database/` | Merges per-experiment databases into analysis-ready datasets (`lda_database_creator.py`). |
| `analysis/` | Thesis pipeline, one module per stage (see `analysis/README.md`): merge → QC → PCA/LDA projection → effects → group stats → diagnostics → report. Group entry `thesis_analysis.py`. Statistics deep dive in `analysis/STATISTICS.md`. |
| `visualization/` | One module per thesis figure, grouped in `methods/` (Fig 1–3), `results/` (Fig 4–17), `supplementary/` (S1–S10); group entry `build_all.py`. Inputs in `visualization/data/`, figures in `visualization/output/`. See `visualization/README.md`. |
| `results/` | Final run tables not consumed by any figure (kept as record). See `results/README.md`. |
| `config/` | Shared settings (`settings.yaml`). |
| `utils/` | DB helpers, ID updates, CSV converters, GUI file pickers. |
| `bootstrap.py` | Package bootstrap / logging init. |

Data flow: `data/*.csv` + recap workbook → `analysis/` → result tables
(`results/`, `visualization/data/`) → `visualization/` → committed figures
(`visualization/output/`). Helper entry points in `scripts/` (see
`scripts/README.md`).
