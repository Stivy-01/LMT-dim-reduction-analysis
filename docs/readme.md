# LMT Thesis Analysis Toolkit

Reproducible thesis pipeline + all 27 thesis figures (17 main + 10 supplementary
+ methods) from Live Mouse Tracker group-housed mouse data.

> 📚 **New to this toolkit?** Start from the root [`README.md`](../README.md)
> and the [`SETUP_GUIDE.md`](../SETUP_GUIDE.md). Module details live in
> `src/README.md`, `src/analysis/README.md` (+ `STATISTICS.md`),
> `src/visualization/README.md` and `scripts/README.md`.

## Overview

Two layers, one repo:

1. **Base preprocessing** (kept from the original toolkit): event filtering,
   behavior processors (daily / 12 h night / hourly), database unification,
   feature ratios.
2. **Thesis layer** (new): metadata-driven analysis
   (`src/analysis/thesis_analysis.py`, split into one module per stage),
   result tables (`src/results/`, `src/visualization/data/`) and one module
   per figure (`src/visualization/`, group entry `build_all.py`).

Design rules: treatment labels come only from metadata (never inferred);
`wt_10132` is handled in a dedicated analysis scope; committed figures rebuild
bit-identically (PNG/CSV) via `build_all`.

## Prerequisites

- Python 3.8+ (3.12 used for the thesis runs) with packages in
  `requirements.txt` (this folder)
- Committed inputs under `data/`: behavior CSV, manual date corrections,
  `LMT RECAP ALL EXPERIMENTS.xlsx`
- SQLite databases from LMT experiments (only to re-run the base preprocessing)
- DB Browser for SQLite (optional, for manual checks)
- Node.js — only to rebuild the Excel workbook

## Installation

```bash
# 1. numpy first, then scipy (order matters on some platforms)
pip install "numpy==1.23.5"
pip install "scipy>=1.9.0"

# 2. everything else + package
pip install -r docs/requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import src.analysis.thesis_analysis; print('ok')"
python -m src.analysis.thesis_analysis --help
python -m src.visualization.build_all --help
```

## Project Structure

```
.
├── data/                     # committed inputs (behavior CSV, corrections, recap)
├── src/
│   ├── analysis/             # thesis pipeline, one module per stage
│   ├── behavior/             # behavior processors (daily / interval / hourly)
│   ├── preprocessing/        # event filtering, feature ratios
│   ├── database/             # database unification
│   ├── visualization/        # figure builders (methods/ results/ supplementary/)
│   │   ├── data/             # all figure inputs
│   │   └── output/           # committed figures
│   ├── results/              # final run tables not consumed by figures
│   ├── config/               # settings.yaml
│   └── utils/                # db_selector, database_utils, id_update, csv converter
├── scripts/                  # entry points (pipeline wrapper, recap, workbook, validation)
├── docs/                     # this documentation
└── SETUP_GUIDE.md            # end-to-end setup
```

## Pipeline Workflow

### 1. Event Filtering

Script: `src/preprocessing/event_filtered.py`

Purpose: Create cleaned event data with timestamps.

```python
from src.preprocessing.event_filtered import create_event_filtered_table, process_events

# The script will show a GUI for:
# 1. Database selection
# 2. Experiment start time selection

# It automatically:
# - Creates EVENT_FILTERED table
# - Excludes non-behavioral events (e.g., RFID errors, brief detections)
# - Merges adjacent events (<1 sec apart)
# - Adds duration and timestamp information
```

### 2. Behavioral Feature Extraction

Choose processor based on temporal resolution:

| Script | Resolution | Use Case | Output Tables |
|--------|------------|----------|---------------|
| behavior_processor.py | Per-experiment | Daily totals | BEHAVIOR_STATS, MULTI_MOUSE_EVENTS |
| behavior_processor_interval.py | 12h intervals (7PM-7AM) | Night-cycle analysis | behavior_stats_intervals |
| behavior_processor_hourly.py | Hourly chunks | Flexible temporal analysis | behavior_hourly, group_events_hourly |

Example usage:

```python
# Daily Totals
from src.behavior.behavior_processor import BehaviorProcessor
processor = BehaviorProcessor(db_path)
processor.process_events()

# 12-hour Intervals (Night Cycle)
from src.behavior.behavior_processor_interval import IntervalBehaviorProcessor
processor = IntervalBehaviorProcessor(db_path)
processor.process_intervals()

# Hourly Analysis
from src.behavior.behavior_processor_hourly import HourlyBehaviorProcessor
processor = HourlyBehaviorProcessor(db_path)
processor.process_hourly()
```

Key Features:
- Separates dyadic interactions into active/passive counts
- Tracks group behaviors (≥3 mice)
- Imputes missing data using mouse-specific medians

### 3. Database Unification

Script: `src/database/lda_database_creator.py`

Purpose: Merge multiple experiments into one analysis-ready dataset
(select source databases via GUI, choose output path). Tip: create separate
merged DBs for different temporal resolutions.

### 4. Thesis analysis

```bash
python scripts/update_recap_analysis_metadata.py "data/LMT RECAP ALL EXPERIMENTS.xlsx"
python -m src.analysis.thesis_analysis --recap "data/LMT RECAP ALL EXPERIMENTS.xlsx"
```

Stages (`src/analysis/`, one module each — see `src/analysis/README.md`):
config → parsing → recap metadata → merge → QC → PCA/LDA projection →
model/feature effects → group stats → diagnostics → report.
Statistics deep dive: `src/analysis/STATISTICS.md`.

### 5. Thesis figures

```bash
python -m src.visualization.build_all
```

One module per figure under `src/visualization/methods|results|supplementary/`
(see `src/visualization/README.md`); each is also runnable standalone.

# Complete Behavior Statistics Table Column Structure

## 🔑 Primary Keys

- `mouse_id` (INTEGER)
- `date` (TEXT) - Format: 'YYYY-MM-DD'

## 🤝 Social Behaviors (Pairwise)

### Approach & Social Approach

- `approach_active_count`
- `approach_passive_count`
- `approach_total_duration`
- `approach_mean_duration`
- `approach_median_duration`
- `approach_std_duration`

- `social_approach_active_count`
- `social_approach_passive_count`
- `social_approach_total_duration`
- `social_approach_mean_duration`
- `social_approach_median_duration`
- `social_approach_std_duration`

### Investigation

- `oral_genital_contact_active_count`
- `oral_genital_contact_passive_count`
- `oral_genital_contact_total_duration`
- `oral_genital_contact_mean_duration`
- `oral_genital_contact_median_duration`
- `oral_genital_contact_std_duration`

- `oral_oral_contact_active_count`
- `oral_oral_contact_passive_count`
- `oral_oral_contact_total_duration`
- `oral_oral_contact_mean_duration`
- `oral_oral_contact_median_duration`
- `oral_oral_contact_std_duration`

### Contact & Movement

- `contact_active_count`
- `contact_passive_count`
- `contact_total_duration`
- `contact_mean_duration`
- `contact_median_duration`
- `contact_std_duration`

- `move_in_contact_active_count`
- `move_in_contact_passive_count`
- `move_in_contact_total_duration`
- `move_in_contact_mean_duration`
- `move_in_contact_median_duration`
- `move_in_contact_std_duration`

- `stop_in_contact_active_count`
- `stop_in_contact_passive_count`
- `stop_in_contact_total_duration`
- `stop_in_contact_mean_duration`
- `stop_in_contact_median_duration`
- `stop_in_contact_std_duration`

### Social Response/Avoidance

- `social_escape_active_count`
- `social_escape_passive_count`
- `social_escape_total_duration`
- `social_escape_mean_duration`
- `social_escape_median_duration`
- `social_escape_std_duration`

- `get_away_active_count`
- `get_away_passive_count`
- `get_away_total_duration`
- `get_away_mean_duration`
- `get_away_median_duration`
- `get_away_std_duration`

## 🚶 Individual Behaviors

### Exploration & Movement

- `rearing_count`
- `rearing_total_duration`
- `rearing_mean_duration`
- `rearing_median_duration`
- `rearing_std_duration`

- `rear_in_centerwindow_count`
- `rear_in_centerwindow_total_duration`
- `rear_in_centerwindow_mean_duration`
- `rear_in_centerwindow_median_duration`
- `rear_in_centerwindow_std_duration`

- `rear_at_periphery_count`
- `rear_at_periphery_total_duration`
- `rear_at_periphery_mean_duration`
- `rear_at_periphery_median_duration`
- `rear_at_periphery_std_duration`

### Zone Occupation

- `center_zone_count`
- `center_zone_total_duration`
- `center_zone_mean_duration`
- `center_zone_median_duration`
- `center_zone_std_duration`

- `periphery_zone_count`
- `periphery_zone_total_duration`
- `periphery_zone_mean_duration`
- `periphery_zone_median_duration`
- `periphery_zone_std_duration`

### Anxiety-Related

- `walljump_count`
- `walljump_total_duration`
- `walljump_mean_duration`
- `walljump_median_duration`
- `walljump_std_duration`

- `sap_count` (Stretched Attend Posture)
- `sap_total_duration`
- `sap_mean_duration`
- `sap_median_duration`
- `sap_std_duration`

- `huddling_count`
- `huddling_total_duration`
- `huddling_mean_duration`
- `huddling_median_duration`
- `huddling_std_duration`

### Isolation

- `isolated_count`
- `isolated_total_duration`
- `isolated_mean_duration`
- `isolated_median_duration`
- `isolated_std_duration`

### Movement (Solo)

- `move_isolated_count`
- `move_isolated_total_duration`
- `move_isolated_mean_duration`
- `move_isolated_median_duration`
- `move_isolated_std_duration`

## 📊 Column Properties

- Count columns: INTEGER, default 0
- Duration columns: REAL, default 0
- Time columns: TEXT
- ID columns: INTEGER

## 🔍 Important Notes

1. All behavior names are sanitized:
   - Spaces/hyphens → underscores
   - Special characters removed
   - Non-alpha first characters prefixed with 'b_'

2. Social behaviors have:
   - Active/passive distinction
   - Separate count columns for initiator/receiver
   - Shared duration statistics

3. Duration statistics for each behavior:
   - Total duration (sum)
   - Mean duration (average)
   - Median duration (middle value)
   - Standard deviation (variability)

## Key Script Utilities

- `id_update.py`: Safely modify mouse IDs across tables
- `database_utils.py`: DB backup/verification functions
- `db_selector.py`: GUI-based database selection (tkcalendar date picker)

## Best Practices

- Backup databases before running processors
- Validate outputs using DB Browser
- Keep committed inputs (`data/`) immutable; new runs write to fresh run
  directories
- Rebuild figures with `python -m src.visualization.build_all` and check
  `git status` shows no unexpected diffs on committed PNG/CSV

## Configuration

- `src/config/settings.yaml`: package paths
- `LMT_ENV`: Set environment (development/production)
- `LMT_DEBUG`: Enable/disable debug mode

## License

This project is licensed under the GNU General Public License v3.0 - see the
[LICENSE](LICENSE) file for details.

## Contact

For support, contributions, or inquiries, please contact:
Andrea Stivala (andreastivala.as@gmail.com)

## Citation

If you use this toolkit in your research, please cite:
Andrea Stivala, 2025

## Acknowledgments

- Live Mouse Tracker project for inspiring this toolkit
- Contributors and maintainers of the core Python packages used in this project
- Forkosh et al. 2019 for the original dimensionality reduction pipeline this project is based on
