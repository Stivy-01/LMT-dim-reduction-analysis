# LMT Thesis Analysis — Setup Guide

Reproduce the thesis analysis and figures from a fresh clone.

## Prerequisites

- Python 3.8+ (3.12 used for the thesis runs)
- Git
- The input data committed under `data/`:
  `merged_analysis_behavior_stats_intervals.csv`,
  `manual_date_corrections.csv`, `LMT RECAP ALL EXPERIMENTS.xlsx`
- Node.js — only needed to rebuild the Excel workbook
  (`scripts/build_thesis_workbook.mjs`); skip otherwise

## 1. Get the code

```bash
git clone https://github.com/Stivy-01/LMT-dim-reduction-analysis.git
cd LMT-dim-reduction-analysis
git checkout thesis-clean
```

## 2. Environment

```bash
python -m venv lmt_env
# Windows:
.\lmt_env\Scripts\activate
# Unix/macOS:
source lmt_env/bin/activate
```

## 3. Install dependencies

```bash
# 1. numpy first, then scipy (order matters for some platforms)
pip install "numpy==1.23.5"
pip install "scipy>=1.9.0"

# 2. everything else
pip install -r docs/requirements.txt
pip install -e .   # development mode (recommended)
```

## 4. Verify installation

```bash
python -c "import src.analysis.thesis_analysis; print('ok')"
python -m src.analysis.thesis_analysis --help
python -m src.visualization.build_all --help
```

## 5. Run the analysis

```powershell
# Full pipeline (analysis + workbook); writes a new run directory
scripts/run_lmt_pipeline.ps1
```

or step by step:

```bash
# 1. normalize the recap workbook into Analysis_Metadata
python scripts/update_recap_analysis_metadata.py "data/LMT RECAP ALL EXPERIMENTS.xlsx"
# 2. thesis analysis -> result tables + run_manifest.json
python -m src.analysis.thesis_analysis --recap "data/LMT RECAP ALL EXPERIMENTS.xlsx"
# 3. all thesis figures (inputs in src/visualization/data/)
python -m src.visualization.build_all
```

Single figures are also runnable standalone, e.g.
`python -m src.visualization.results.fig_03_group_trajectories --help`.
See `src/README.md`, `src/analysis/README.md` (+ `STATISTICS.md`) and
`src/visualization/README.md` for module details.
