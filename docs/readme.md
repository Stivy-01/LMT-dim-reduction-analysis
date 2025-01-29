# LMT Analysis Package

## Overview

The LMT Analysis Package provides tools for analyzing Laboratory Mouse Tracker (LMT) data. It includes modules for data preprocessing, behavior analysis, visualization, and database management.

## Installation

To install the package, run:

```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the package:

```python
from src.bootstrap import initialize_app

initialize_app()
```

## Configuration

The package can be configured using the following environment variables:

- `LMT_ENV`: Set the environment (e.g., `development`, `production`).
- `LMT_DEBUG`: Enable or disable debug mode (`True` or `False`).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## Contact

For support or inquiries, please contact Andrea Stivala at andreastivala.as@gmail.com.

LMT Stress Experiment Analysis Toolkit
A Python-based pipeline for behavioral identity analysis using LDA/PCA.

Overview
This toolkit processes rodent behavioral data from LMT experiments to perform identity domain analysis via dimensionality reduction techniques (LDA/PCA). Currently supports baseline analysis with three temporal resolutions:

- 12-hour intervals (night cycle)
- 4-hour chunks (circadian phases)
- Hourly resolution

Prerequisites
- Python 3.8+ with packages: pandas, numpy, scipy, scikit-learn, sqlite3, tkinter
- SQLite databases containing raw event/animal data
- DB Browser for SQLite (recommended for manual checks)

Pipeline Workflow

1. Event Filtering
Script: Event_filtered.py

Purpose: Create cleaned event data with timestamps.

Steps:
- Run script, select database via GUI
- Set experiment start time using calendar prompt

Output:
- Creates EVENT_FILTERED table
- Excludes non-behavioral events (e.g., RFID errors, brief detections)
- Merges adjacent events (<1 sec apart)
- Adds duration/frame metrics

2. Behavioral Feature Extraction
Choose processor based on temporal resolution:

| Script | Resolution | Use Case | Output Tables |
|--------|------------|----------|---------------|
| behavior_processor.py | Per-experiment | Daily totals | BEHAVIOR_STATS, MULTI_MOUSE_EVENTS |
| behavior_processor_interval.py | 12h intervals (7PM-7AM) | Night-cycle analysis | behavior_stats_intervals |
| behavior_processor_hourly.py | Hourly chunks | Flexible temporal analysis | behavior_hourly, group_events_hourly |

Key Features:
- Separates dyadic interactions into active/passive counts
- Tracks group behaviors (≥3 mice)
- Imputes missing data using mouse-specific medians

3. Database Unification
Script: lda_database_creator.py

Purpose: Merge multiple experiments into one analysis-ready dataset.

Steps:
- Select source databases (GUI)
- Choose output path

Output:
- Merged SQLite database
- Corresponding CSV file

Tip: Create separate merged DBs for different temporal resolutions

4. Dimensionality Reduction Analysis
Scripts:

| Script | Method | Temporal Resolution | Key Features |
|--------|--------|-------------------|--------------|
| analysis_demo.py | LDA | 12h intervals | Identity domain stability scoring |
| analysis_demo_4h.py | LDA | 4h chunks | Circadian phase analysis |
| analysis_demo_pca.py | PCA | 4h chunks | 3D feature space projection |

Workflow:
- Input merged CSV
- Automated preprocessing:
  - Variance thresholding (remove low-variance features)
  - Correlation filtering (>0.95 correlated features removed)
  - Standardization (Z-score normalization)

Output:
- Stability scores (LDA)
- Projection coordinates in identity space

5. Post-Processing & Visualization
To be implemented

Suggested tools:
- Plotly/Dash for interactive 3D plots
- Seaborn for stability score heatmaps

Key Script Utilities
- id_update.py: Safely modify mouse IDs across tables
- database_utils.py: DB backup/verification functions
- db_selector.py: GUI-based database selection

Best Practices
- Backup databases before running processors
- Validate outputs using DB Browser
- For 4h analysis: Use merged_for_lda_hourly.csv
- For 12h analysis: Use merged_for_lda_intervals.csv

Based on Forkosh's Identity Domains concept (GitHub).

For support, contact [andreastivala.as@gmail.com/email].

## Recent Updates

### Project Organization
- Added dedicated `tools` directory for utility scripts:
  - `fix_encoding.py`: Tool for fixing file encodings to UTF-8
- Consolidated all configuration files in `docs` folder:
  - Updated `requirements.txt` with complete dependency list
  - Added specific version constraints for core packages
  - Improved installation order to handle dependencies correctly

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

### Installation Updates
The installation process has been improved with a specific order:
1. Install numpy (base dependency)
2. Install scipy (required for scikit-learn)
3. Install remaining packages

For the complete list of dependencies and installation instructions, see the updated `requirements.txt` in the `docs` folder.