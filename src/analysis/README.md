# Analysis Module Documentation

## Overview
This directory contains the core analysis modules for the LMT behavioral analysis pipeline. Each module is designed to handle specific aspects of the analysis workflow.

## Module Descriptions

### 1. analysis_pipeline.py
The main orchestrator module that coordinates the complete analysis workflow.
- **Key Class**: `AnalysisPipeline`
- **Main Functions**:
  - `run_analysis()`: Executes the complete analysis pipeline
- **Features**:
  - Coordinates data preprocessing
  - Runs sequential and parallel analyses
  - Handles statistical analysis
  - Generates comprehensive reports
- **Dependencies**: preprocessing, sequential_analysis, parallel_pca, parallel_lda, statistics, reporting

### 2. preprocessing.py
Handles data preprocessing and feature selection.
- **Key Class**: `BehaviorPreprocessor`
- **Main Functions**:
  - `fit_transform()`: Preprocesses behavioral data
  - `_handle_datetime()`: Processes temporal data
  - `_normalize_by_interval()`: Normalizes features within intervals
  - `_clean_data()`: Cleans and prepares data
  - `_remove_correlated_features()`: Filters highly correlated features
- **Features**:
  - Data cleaning and normalization
  - Feature selection
  - Correlation filtering
  - Standardization
  - Cross-validation split creation

### 3. parallel_pca.py
Implements PCA part of Forkosh's parallel approach.
- **Key Function**: `analyze_pca_parallel()`
- **Features**:
  - Data preprocessing with standardization
  - Low variance feature removal
  - Quantile normalization
  - Component significance testing
  - Distribution overlap analysis
  - Stability scoring
- **Returns**: Dictionary with PCA results including transformed space, components, eigenvalues, etc.

### 4. parallel_lda.py
Implements LDA part of Forkosh's parallel approach.
- **Key Function**: `analyze_lda_parallel()`
- **Features**:
  - Data preprocessing
  - Identity domain analysis
  - Component significance testing
  - Distribution overlap analysis
  - Discriminative power calculation
- **Returns**: Dictionary with LDA results including transformed space, components, eigenvalues, etc.

### 5. statistics.py
Implements statistical analysis methods.
- **Key Functions**:
  - `perform_permutation_test()`: Stability assessment
  - `perform_variance_ttest()`: Variance explained testing
  - `perform_regression_analysis()`: Linear regression analysis
  - `perform_anova()`: ANOVA across components
  - `perform_correlation_analysis()`: Feature correlation analysis
  - `calculate_stability_scores()`: Stability scoring
- **Features**:
  - Comprehensive statistical testing
  - Multiple test types supported
  - Cross-validation
  - Significance testing

### 6. reporting.py
Handles generation of analysis reports.
- **Key Class**: `AnalysisReport`
- **Main Functions**:
  - `generate_markdown()`: Creates markdown report
  - `generate_html()`: Creates HTML report
- **Features**:
  - Comprehensive reporting of:
    - Analysis overview
    - Sequential and parallel results
    - Statistical test results
    - Feature significance
    - Component structure
  - Multiple output formats (Markdown, HTML)

### 7. sequential_analysis.py
Implements sequential analysis approach.
- **Key Function**: `analyze_identity_domain_sequential()`
- **Features**:
  - PCA followed by LDA
  - Stability scoring
  - Variance explained calculation

### 8. identity_domain.py
Implements identity domain analysis.
- **Key Class**: `IdentityDomainAnalyzer`
- **Features**:
  - Scatter matrix computation
  - Eigendecomposition
  - Identity domain transformation

### 9. gui.py
Provides graphical user interface for analysis configuration.
- **Key Class**: `AnalysisGUI`
- **Features**:
  - Analysis type selection
  - File selection
  - Configuration settings
  - User-friendly interface

### 10. analysis_demo.py
Demonstrates usage of analysis pipeline.
- **Features**:
  - Complete analysis workflow example
  - Statistical testing demonstration
  - Result visualization
  - Feature importance analysis

### 11. analysis_demo_pca.py
Focused demonstration of PCA analysis.
- **Features**:
  - PCA-specific workflow
  - Component analysis
  - Result visualization

### 12. analysis_demo_4h.py
Demonstration with 4-hour interval analysis.
- **Features**:
  - Time-based analysis example
  - Interval-specific processing
  - Result visualization

## Usage
The main entry point is `run_analysis.py` in the src directory, which:
1. Shows GUI for configuration
2. Creates and runs analysis pipeline
3. Generates comprehensive reports and results

## Dependencies
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib (for visualizations)
- markdown (for report generation) 