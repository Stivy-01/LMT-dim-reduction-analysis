# LMT Dimensionality Reduction Analysis Toolkit
A Python-based pipeline for behavioral identity analysis using LDA/PCA.

## Overview

This toolkit processes rodent behavioral data from LMT experiments to perform identity domain analysis via dimensionality reduction techniques (LDA/PCA). Currently supports baseline analysis with three temporal resolutions:

- 12-hour intervals (night cycle)
- 4-hour chunks (circadian phases)
- Hourly resolution

## Prerequisites
- Python 3.8+ with packages: pandas, numpy, scipy, scikit-learn, sqlite3, tkinter
- SQLite databases containing raw event/animal data
- DB Browser for SQLite (recommended for manual checks)

## Installation

The installation requires a specific order due to dependencies:

```bash
# 1. Install numpy first
pip install numpy==1.23.5

# 2. Install scipy before scikit-learn
pip install scipy>=1.9.0

# 3. Install remaining packages
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/                          # Data and visualization outputs
│   └── behavior_stats_intervals_analyzed/  # Interactive HTML visualizations
├── src/
│   ├── analysis/                  # Analysis scripts
│   ├── behavior/                  # Behavior processing modules
│   ├── preprocessing/             # Data preprocessing tools
│   ├── visualization/             # Visualization utilities
│   └── utils/                     # Utility functions
├── docs/                          # Documentation
└── tools/                         # Utility scripts
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

Purpose: Merge multiple experiments into one analysis-ready dataset.

Steps:
- Select source databases (GUI)
- Choose output path

Output:
- Merged SQLite database
- Corresponding CSV file

Tip: Create separate merged DBs for different temporal resolutions

### 4. Dimensionality Reduction Analysis

| Script | Method | Temporal Resolution | Key Features |
|--------|--------|-------------------|--------------|
| analysis_demo.py | LDA | 12h intervals | Identity domain stability scoring |
| analysis_demo_4h.py | LDA | 4h chunks | Circadian phase analysis |
| analysis_demo_pca.py | PCA | 4h chunks | 3D feature space projection |

Example usage:
```python
# LDA Analysis
from src.analysis.analysis_demo import run_analysis

# PCA Analysis
from src.analysis.analysis_demo_pca import run_pca_analysis
```

The analysis includes:
- Feature preprocessing
- Variance thresholding
- Correlation filtering (>0.95 correlated features removed)
- Standardization (Z-score normalization)

### 5. Visualization

The package includes interactive visualization capabilities:

```python
from src.visualization.identity_space_plotter import plot_identity_space

# Creates interactive HTML visualizations
# Output location: data/behavior_stats_(resolution)_analyzed/
```

Visualization outputs can be found in:
- `data/behavior_stats_(resolution)_analyzed/*.html`
- Open these files in any web browser to view interactive plots

## Key Script Utilities
- `id_update.py`: Safely modify mouse IDs across tables
- `database_utils.py`: DB backup/verification functions
- `db_selector.py`: GUI-based database selection

## Best Practices
- Backup databases before running processors
- Validate outputs using DB Browser
- For 4h analysis: Use merged_for_lda_hourly.csv
- For 12h analysis: Use merged_for_lda_intervals.csv

## Configuration

The package can be configured using environment variables:
- `LMT_ENV`: Set environment (development/production)
- `LMT_DEBUG`: Enable/disable debug mode

## Recent Updates

### Project Organization
- Added dedicated `tools` directory for utility scripts:
  - `fix_encoding.py`: Tool for fixing file encodings to UTF-8
- Consolidated all configuration files in `docs` folder

### New Dependencies
Added support for:
- `scipy>=1.9.0`: Linear algebra operations for analysis
- `scikit-learn>=1.0.0`: PCA and preprocessing functionality
- `plotly>=5.13.0`: Advanced visualization capabilities
- `tkcalendar>=1.6.1`: Enhanced date selection UI
- `seaborn>=0.12.2`: Statistical data visualization

### Encoding Management
- Added UTF-8 encoding support across all source files
- Implemented automatic encoding detection and conversion
- Fixed potential encoding-related issues in data processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For support or inquiries, please contact Andrea Stivala at andreastivala.as@gmail.com.