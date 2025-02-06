"""
Reporting Module
==============

Handles generation of analysis reports in both Markdown and HTML formats.
Includes comprehensive reporting of:
- Analysis overview
- Sequential and parallel analysis results
- Statistical test results
- Feature significance analysis
- Component structure analysis
"""

import pandas as pd
import numpy as np
import markdown
from pathlib import Path

class AnalysisReport:
    """
    Generates comprehensive analysis reports in multiple formats.
    """
    
    def __init__(self, results_dict, significance_summary, feature_names=None):
        """
        Initialize report generator.
        
        Args:
            results_dict: Dictionary containing all analysis results
            significance_summary: Summary of statistical significance tests
            feature_names: List of feature names (optional)
        """
        self.results = results_dict
        self.significance = significance_summary
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(100)]
        
    def _generate_overview_section(self):
        """Generate analysis overview section."""
        return f"""# Behavioral Identity Domain Analysis Report

## 1. Analysis Overview
- Original dimensions: {self.results['original_dims']}
- Final PCA dimensions: {self.results['pca_dims']}
- Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _generate_sequential_section(self):
        """Generate sequential analysis results section."""
        return f"""
## 2. Sequential Analysis Results
- Stability score: {self.results['sequential_stability']:.3f}
- PCA variance explained: {self.results['sequential_pca_var']:.3f}
"""

    def _generate_parallel_section(self):
        """Generate parallel analysis results section."""
        return f"""
## 3. Parallel Analysis Results (Forkosh Approach)
### PCA Results
- Stability score: {self.results['parallel_pca_stability']:.3f}
- Variance explained: {self.results['parallel_pca_var']:.3f}
- Number of significant components: {self.significance['PCA']['n_significant_components']}
- Component p-values:
{pd.DataFrame({
    'Component': range(1, len(self.significance['PCA']['component_p_values'])+1),
    'p-value': self.significance['PCA']['component_p_values']
}).to_markdown(index=False)}

### LDA Results
- Stability score: {self.results['parallel_lda_stability']:.3f}
- Number of significant components: {self.significance['LDA']['n_significant_components']}
- Component p-values:
{pd.DataFrame({
    'Component': range(1, len(self.significance['LDA']['component_p_values'])+1),
    'p-value': self.significance['LDA']['component_p_values']
}).to_markdown(index=False)}
"""

    def _generate_feature_analysis_section(self):
        """Generate feature significance analysis section."""
        section = "\n## 4. Significant Features Analysis\n### PCA Components\n"
        
        # Add PCA significant features
        for comp, features in self.significance['PCA']['significant_features'].items():
            section += f"\n#### {comp}\n"
            for feat in features:
                weight = self.results['parallel_pca_feature_significance'][comp][feat]['weight']
                ci_lower = self.results['parallel_pca_feature_significance'][comp][feat]['ci_lower']
                ci_upper = self.results['parallel_pca_feature_significance'][comp][feat]['ci_upper']
                section += f"- {feat}: {weight:.3f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}])\n"
        
        section += "\n### LDA Components\n"
        
        # Add LDA significant features
        for comp, features in self.significance['LDA']['significant_features'].items():
            section += f"\n#### {comp}\n"
            for feat in features:
                weight = self.results['parallel_lda_feature_significance'][comp][feat]['weight']
                ci_lower = self.results['parallel_lda_feature_significance'][comp][feat]['ci_lower']
                ci_upper = self.results['parallel_lda_feature_significance'][comp][feat]['ci_upper']
                section += f"- {feat}: {weight:.3f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}])\n"
        
        return section

    def _generate_statistics_section(self):
        """Generate statistical analysis section."""
        stats = self.results['forkosh_statistics']
        
        section = f"""
## 5. Statistical Analysis Results
### Permutation Tests
- PCA Stability: {stats['permutation_tests']['pca']['observed']:.3f}
  - p-value: {stats['permutation_tests']['pca']['p_value']:.3f}
- LDA Stability: {stats['permutation_tests']['lda']['observed']:.3f}
  - p-value: {stats['permutation_tests']['lda']['p_value']:.3f}

### Variance Explained (t-tests)
#### PCA Components
{pd.DataFrame({
    'Component': range(1, len(stats['ttest_results']['PCA']['var_explained'])+1),
    'Variance': stats['ttest_results']['PCA']['var_explained'],
    'p-value': stats['ttest_results']['PCA']['p_values']
}).to_markdown(index=False)}

#### LDA Components
{pd.DataFrame({
    'Component': range(1, len(stats['ttest_results']['LDA']['var_explained'])+1),
    'Variance': stats['ttest_results']['LDA']['var_explained'],
    'p-value': stats['ttest_results']['LDA']['p_values']
}).to_markdown(index=False)}

### Linear Regression (R² Analysis)
- PCA R² scores: {stats['regression_results']['PCA']['r2_scores']}
- LDA R² scores: {stats['regression_results']['LDA']['r2_scores']}

### ANOVA Results
#### PCA Components
{pd.DataFrame({
    'Component': range(1, len(stats['anova_results']['PCA']['f_statistics'])+1),
    'F-statistic': stats['anova_results']['PCA']['f_statistics'],
    'p-value': stats['anova_results']['PCA']['p_values']
}).to_markdown(index=False)}

#### LDA Components
{pd.DataFrame({
    'Component': range(1, len(stats['anova_results']['LDA']['f_statistics'])+1),
    'F-statistic': stats['anova_results']['LDA']['f_statistics'],
    'p-value': stats['anova_results']['LDA']['p_values']
}).to_markdown(index=False)}
"""
        return section

    def _generate_correlation_section(self):
        """Generate correlation analysis section."""
        section = "\n### Feature Correlations\nSignificant correlations with behavioral features (p < 0.05):\n"
        
        stats = self.results['forkosh_statistics']
        for space_name in ['PCA', 'LDA']:
            section += f"\n#### {space_name} Components\n"
            corr_results = stats['correlation_results'][space_name]
            for comp_idx in range(corr_results['correlations'].shape[0]):
                significant_idx = corr_results['p_values'][comp_idx] < 0.05
                if np.any(significant_idx):
                    section += f"\nComponent {comp_idx + 1}:\n"
                    for feat_idx in np.where(significant_idx)[0]:
                        section += (f"- {self.feature_names[feat_idx]}: "
                                  f"r = {corr_results['correlations'][comp_idx, feat_idx]:.3f} "
                                  f"(p = {corr_results['p_values'][comp_idx, feat_idx]:.3f})\n")
        
        return section

    def _generate_files_section(self):
        """Generate files summary section."""
        return """
## 6. Generated Files
The following files contain detailed analysis results:
- `*_metrics.csv`: Summary metrics for both approaches
- `*_sequential_space.csv`: Identity space from sequential approach
- `*_parallel_pca_space.csv`: PCA space from parallel approach
- `*_parallel_lda_space.csv`: LDA space from parallel approach
"""

    def generate_markdown(self):
        """Generate complete markdown report."""
        sections = [
            self._generate_overview_section(),
            self._generate_sequential_section(),
            self._generate_parallel_section(),
            self._generate_feature_analysis_section(),
            self._generate_statistics_section(),
            self._generate_correlation_section(),
            self._generate_files_section()
        ]
        
        return '\n'.join(sections)

    def generate_html(self):
        """Generate HTML report with styling."""
        md_content = self.generate_markdown()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #2c3e50;
                }
                h2 {
                    color: #34495e;
                    margin-top: 30px;
                }
                h3 {
                    color: #7f8c8d;
                }
                h4 {
                    color: #95a5a6;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f5f6fa;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .feature-list {
                    margin-left: 20px;
                }
                .confidence-interval {
                    color: #7f8c8d;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
        {content}
        </body>
        </html>
        """
        
        return html_template.format(content=markdown.markdown(md_content))

def generate_report(results_dict, significance_summary, output_dir, base_filename, feature_names=None):
    """
    Generate and save analysis reports.
    
    Args:
        results_dict: Dictionary containing all analysis results
        significance_summary: Summary of statistical significance tests
        output_dir: Directory to save reports
        base_filename: Base name for report files
        feature_names: List of feature names (optional)
    
    Returns:
        tuple: (markdown_path, html_path)
    """
    report = AnalysisReport(results_dict, significance_summary, feature_names)
    
    # Save markdown report
    md_path = output_dir / f"{base_filename}_analysis_report.md"
    with open(md_path, 'w') as f:
        f.write(report.generate_markdown())
    
    # Save HTML report
    html_path = output_dir / f"{base_filename}_analysis_report.html"
    with open(html_path, 'w') as f:
        f.write(report.generate_html())
    
    return md_path, html_path 