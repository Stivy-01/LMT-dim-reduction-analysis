# analysis_demo.py (OPTIMIZED VERSION)
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
import os
import sys
import markdown
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_analysis_type():
    """Get analysis type from user via GUI."""
    root = tk.Tk()
    root.withdraw()
    
    table_type = simpledialog.askinteger(
        "Analysis Type",
        "Choose analysis type (enter a number):\n\n" +
        "1: Hourly Analysis\n" +
        "   - behavior_hourly\n\n" +
        "2: Interval Analysis\n" +
        "   - behavior_stats_intervals\n\n" +
        "3: Daily Analysis\n" +
        "   - BEHAVIOR_STATS",
        minvalue=1, maxvalue=3
    )
    
    if not table_type:
        raise ValueError("No analysis type selected")
    
    # Map analysis type to table name
    table_mapping = {
        1: 'behavior_hourly',
        2: 'behavior_stats_intervals',
        3: 'BEHAVIOR_STATS'
    }
    
    return table_type, table_mapping[table_type]

def select_csv_file(table_name):
    """Select CSV file from the appropriate analysis directory."""
    data_dir = project_root / 'data'
    analysis_dir = data_dir / f"{table_name}_to_analyze"
    
    if not analysis_dir.exists():
        print(f"\n❌ Error: Analysis directory not found at {analysis_dir}")
        print("Please ensure the CSV file exists in the correct directory.")
        return None
        
    # Check for the specific file first
    default_file = analysis_dir / f"merged_analysis_{table_name}.csv"
    if default_file.exists():
        return str(default_file)
    
    # If specific file not found, show file dialog
    root = tk.Tk()
    root.withdraw()
    
    csv_path = filedialog.askopenfilename(
        title="Select CSV file to analyze",
        initialdir=analysis_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    return csv_path if csv_path else None

def preprocess_data(df):
    # Handle datetime
    if 'interval_start' in df.columns:
        df['interval_start'] = pd.to_datetime(df['interval_start'])
        df['timestamp'] = df['interval_start'].astype('int64') // 10**9
        df = df.drop('interval_start', axis=1)
    
    # Convert to numeric and clean
    non_target_cols = [col for col in df.columns if col not in ['mouse_id', 'date']]
    df[non_target_cols] = df[non_target_cols].apply(pd.to_numeric, errors='coerce')
    
    X = df.drop(['mouse_id', 'date'], axis=1, errors='ignore').dropna(axis=1, how='all')
    X = X.fillna(X.mean())
    
    # Initial feature selection
    selector = VarianceThreshold(threshold=0.1)
    return selector.fit_transform(X), df['mouse_id']

class IdentityDomainAnalyzer:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.components_ = None

    def _compute_scatter_matrices(self, X, y):
        unique_classes = np.unique(y)
        n_features = X.shape[1]
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        overall_mean = np.mean(X, axis=0)
        
        for cls in unique_classes:
            X_cls = X[y == cls]
            if len(X_cls) > 1:
                Sw += np.cov(X_cls.T) * (len(X_cls)-1)
            if len(X_cls) > 0:
                mean_diff = (np.mean(X_cls, axis=0) - overall_mean).reshape(-1, 1)
                Sb += len(X_cls) * (mean_diff @ mean_diff.T)
                
        return Sw, Sb

    def fit(self, X, y):
        Sw, Sb = self._compute_scatter_matrices(X, y)
        
        # Enhanced regularization
        Sw_reg = (Sw + Sw.T) / 2 + 1e-3 * np.eye(Sw.shape[0])  # Increased regularization
        Sb = (Sb + Sb.T) / 2
        
        # Eigen decomposition
        eig_vals, eig_vecs = eigh(Sb, Sw_reg, check_finite=False)
        order = np.argsort(eig_vals)[::-1]
        self.components_ = eig_vecs[:, order[:self.n_components]].T
        return self

    def transform(self, X):
        return X @ self.components_.T

def test_component_significance(X, components, method='permutation', n_permutations=1000):
    """Test statistical significance of components using permutation testing"""
    n_samples, n_features = X.shape
    original_variance = np.var(X @ components.T, axis=0)
    p_values = np.zeros(len(components))
    
    for i in range(n_permutations):
        # Permute each feature independently
        X_perm = np.copy(X)
        for j in range(n_features):
            np.random.shuffle(X_perm[:, j])
        
        # Project permuted data
        perm_variance = np.var(X_perm @ components.T, axis=0)
        p_values += (perm_variance >= original_variance).astype(int)
    
    p_values = p_values / n_permutations
    
    return {
        'p_values': p_values,
        'significant_components': p_values < 0.05,
        'n_significant': sum(p_values < 0.05)
    }

def test_feature_contributions(X, components, feature_names, n_bootstraps=1000):
    """Test significance of feature contributions using bootstrap confidence intervals"""
    n_samples = X.shape[0]
    n_components = len(components)
    bootstrap_weights = np.zeros((n_bootstraps, n_components, len(feature_names)))
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        
        # Calculate weights for bootstrapped sample
        if n_components == 1:
            bootstrap_weights[i] = np.abs(components)
        else:
            bootstrap_weights[i] = np.abs(components)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_weights, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_weights, 97.5, axis=0)
    mean_weights = np.mean(bootstrap_weights, axis=0)
    
    # Determine significant features (CI doesn't include 0)
    significant_features = (ci_lower > 0) | (ci_upper < 0)
    
    results = {}
    for comp_idx in range(n_components):
        comp_results = {
            feature_names[i]: {
                'weight': mean_weights[comp_idx, i],
                'ci_lower': ci_lower[comp_idx, i],
                'ci_upper': ci_upper[comp_idx, i],
                'significant': significant_features[comp_idx, i]
            }
            for i in range(len(feature_names))
        }
        results[f'Component_{comp_idx+1}'] = comp_results
    
    return results

def summarize_significance_results(results):
    """Create a summary of significance testing results"""
    summary = {
        'PCA': {
            'n_significant_components': results['pca_significance']['n_significant'],
            'component_p_values': results['pca_significance']['p_values'],
            'significant_features': {
                f'Component_{i+1}': [
                    feat for feat, data in comp_data.items() 
                    if data['significant']
                ]
                for i, comp_data in enumerate(results['pca_feature_significance'].values())
            }
        },
        'LDA': {
            'n_significant_components': results['lda_significance']['n_significant'],
            'component_p_values': results['lda_significance']['p_values'],
            'significant_features': {
                f'Component_{i+1}': [
                    feat for feat, data in comp_data.items() 
                    if data['significant']
                ]
                for i, comp_data in enumerate(results['lda_feature_significance'].values())
            }
        }
    }
    return summary

def analyze_identity_domain_sequential(X, y):
    """Current approach: PCA followed by LDA"""
    # First reduce dimensions with PCA
    pca = PCA(n_components=0.80)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
    
    # Then apply LDA
    ida = IdentityDomainAnalyzer(n_components=4)
    ida.fit(X_pca, y)
    X_ids = ida.transform(X_pca)
    
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_ids[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return np.mean(stability_scores), X_ids, pca.explained_variance_ratio_.sum()

def perform_forkosh_statistics(X, y, X_pca, X_lda, feature_names):
    """
    Implement Forkosh's statistical analysis methods
    1. Permutation Tests
    2. One-Sample t-Tests
    3. Linear Regression and R² Analysis
    4. ANOVA
    5. Pearson's Correlation
    """
    from scipy import stats
    
    stats_results = {
        'permutation_tests': {},
        'ttest_results': {},
        'regression_results': {},
        'anova_results': {},
        'correlation_results': {}
    }
    
    # 1. Permutation Tests for ID stability
    print("Running permutation tests...")
    n_permutations = 1000
    observed_stability_pca = calculate_stability_scores(X_pca, y)
    observed_stability_lda = calculate_stability_scores(X_lda, y)
    
    perm_stability_pca = np.zeros(n_permutations)
    perm_stability_lda = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_stability_pca[i] = calculate_stability_scores(X_pca, y_perm)
        perm_stability_lda[i] = calculate_stability_scores(X_lda, y_perm)
    
    stats_results['permutation_tests'] = {
        'pca': {
            'observed': observed_stability_pca,
            'permuted_mean': np.mean(perm_stability_pca),
            'p_value': np.mean(perm_stability_pca >= observed_stability_pca)
        },
        'lda': {
            'observed': observed_stability_lda,
            'permuted_mean': np.mean(perm_stability_lda),
            'p_value': np.mean(perm_stability_lda >= observed_stability_lda)
        }
    }
    
    # 2. One-Sample t-Tests for variance explained
    print("Performing t-tests...")
    for space_name, space_data in [('PCA', X_pca), ('LDA', X_lda)]:
        var_explained = np.var(space_data, axis=0)
        null_vars = np.zeros((1000, space_data.shape[1]))
        
        for i in range(1000):
            X_shuffled = np.copy(X)
            for j in range(X.shape[1]):
                np.random.shuffle(X_shuffled[:, j])
            if space_name == 'PCA':
                null_vars[i] = np.var(X_shuffled @ space_data.T, axis=0)
            else:
                null_vars[i] = np.var(space_data, axis=0)
        
        t_stats = []
        p_values = []
        for comp_idx in range(space_data.shape[1]):
            t_stat, p_val = stats.ttest_1samp(null_vars[:, comp_idx], var_explained[comp_idx])
            t_stats.append(t_stat)
            p_values.append(p_val)
        
        stats_results['ttest_results'][space_name] = {
            'var_explained': var_explained,
            't_statistics': t_stats,
            'p_values': p_values
        }
    
    # 3. Linear Regression Analysis
    print("Performing regression analysis...")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    
    for space_name, space_data in [('PCA', X_pca), ('LDA', X_lda)]:
        r2_scores = []
        for comp_idx in range(space_data.shape[1]):
            reg = LinearRegression()
            scores = cross_val_score(reg, X, space_data[:, comp_idx], cv=5, scoring='r2')
            r2_scores.append(np.mean(scores))
        
        stats_results['regression_results'][space_name] = {
            'r2_scores': r2_scores
        }
    
    # 4. ANOVA across components
    print("Performing ANOVA...")
    for space_name, space_data in [('PCA', X_pca), ('LDA', X_lda)]:
        f_stats = []
        p_values = []
        for comp_idx in range(space_data.shape[1]):
            groups = [space_data[y == mouse_id, comp_idx] for mouse_id in np.unique(y)]
            f_stat, p_val = stats.f_oneway(*groups)
            f_stats.append(f_stat)
            p_values.append(p_val)
        
        stats_results['anova_results'][space_name] = {
            'f_statistics': f_stats,
            'p_values': p_values
        }
    
    # 5. Pearson's Correlation
    print("Computing correlations...")
    for space_name, space_data in [('PCA', X_pca), ('LDA', X_lda)]:
        corr_matrix = np.zeros((space_data.shape[1], len(feature_names)))
        p_values = np.zeros((space_data.shape[1], len(feature_names)))
        
        for i in range(space_data.shape[1]):
            for j, feat in enumerate(feature_names):
                corr, p_val = stats.pearsonr(space_data[:, i], X[:, j])
                corr_matrix[i, j] = corr
                p_values[i, j] = p_val
        
        stats_results['correlation_results'][space_name] = {
            'correlations': corr_matrix,
            'p_values': p_values
        }
    
    return stats_results

def analyze_identity_domain_parallel(X, y):
    """Forkosh approach: PCA and LDA in parallel with comprehensive statistical testing"""
    # Data preprocessing with quantile normalization
    qt = QuantileTransformer(output_distribution='normal')
    X_transformed = qt.fit_transform(X)
    
    # Get feature names for significance testing
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Original analysis
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_transformed)
    
    ida = IdentityDomainAnalyzer(n_components=4)
    ida.fit(X_transformed, y)
    X_lda = ida.transform(X_transformed)
    
    # Original statistical tests
    pca_significance = test_component_significance(X_transformed, pca.components_)
    lda_significance = test_component_significance(X_transformed, ida.components_)
    pca_feature_significance = test_feature_contributions(X_transformed, pca.components_, feature_names)
    lda_feature_significance = test_feature_contributions(X_transformed, ida.components_, feature_names)
    
    # Calculate stability scores
    pca_stability = calculate_stability_scores(X_pca, y)
    lda_stability = calculate_stability_scores(X_lda, y)
    
    # Perform Forkosh's statistical analyses
    forkosh_stats = perform_forkosh_statistics(X_transformed, y, X_pca, X_lda, feature_names)
    
    return {
        'pca_stability': pca_stability,
        'lda_stability': lda_stability,
        'pca_space': X_pca,
        'lda_space': X_lda,
        'pca_explained_var': pca.explained_variance_ratio_.sum(),
        'pca_significance': pca_significance,
        'lda_significance': lda_significance,
        'pca_feature_significance': pca_feature_significance,
        'lda_feature_significance': lda_feature_significance,
        'forkosh_statistics': forkosh_stats
    }

def generate_analysis_report(results_dict, significance_summary, output_dir, base_filename):
    """Generate a comprehensive analysis report including Forkosh statistics"""
    report_md = f"""# Behavioral Identity Domain Analysis Report
Generated for: {base_filename}

## 1. Analysis Overview
- Original dimensions: {results_dict['original_dims']}
- Final PCA dimensions: {results_dict['pca_dims']}
- Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. Sequential Analysis Results
- Stability score: {results_dict['sequential_stability']:.3f}
- PCA variance explained: {results_dict['sequential_pca_var']:.3f}

## 3. Parallel Analysis Results (Forkosh Approach)
### PCA Results
- Stability score: {results_dict['parallel_pca_stability']:.3f}
- Variance explained: {results_dict['parallel_pca_var']:.3f}
- Number of significant components: {significance_summary['PCA']['n_significant_components']}
- Component p-values:
{pd.DataFrame({'Component': range(1, len(significance_summary['PCA']['component_p_values'])+1),
               'p-value': significance_summary['PCA']['component_p_values']}).to_markdown(index=False)}

### LDA Results
- Stability score: {results_dict['parallel_lda_stability']:.3f}
- Number of significant components: {significance_summary['LDA']['n_significant_components']}
- Component p-values:
{pd.DataFrame({'Component': range(1, len(significance_summary['LDA']['component_p_values'])+1),
               'p-value': significance_summary['LDA']['component_p_values']}).to_markdown(index=False)}

## 4. Significant Features Analysis
### PCA Components
"""
    
    # Add PCA significant features
    for comp, features in significance_summary['PCA']['significant_features'].items():
        report_md += f"\n#### {comp}\n"
        for feat in features:
            weight = results_dict['parallel_pca_feature_significance'][comp][feat]['weight']
            ci_lower = results_dict['parallel_pca_feature_significance'][comp][feat]['ci_lower']
            ci_upper = results_dict['parallel_pca_feature_significance'][comp][feat]['ci_upper']
            report_md += f"- {feat}: {weight:.3f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}])\n"
    
    report_md += "\n### LDA Components\n"
    
    # Add LDA significant features
    for comp, features in significance_summary['LDA']['significant_features'].items():
        report_md += f"\n#### {comp}\n"
        for feat in features:
            weight = results_dict['parallel_lda_feature_significance'][comp][feat]['weight']
            ci_lower = results_dict['parallel_lda_feature_significance'][comp][feat]['ci_lower']
            ci_upper = results_dict['parallel_lda_feature_significance'][comp][feat]['ci_upper']
            report_md += f"- {feat}: {weight:.3f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}])\n"
    
    report_md += """
## 5. Files Generated
The following files contain detailed analysis results:
- `*_metrics.csv`: Summary metrics for both approaches
- `*_sequential_space.csv`: Identity space from sequential approach
- `*_parallel_pca_space.csv`: PCA space from parallel approach
- `*_parallel_lda_space.csv`: LDA space from parallel approach

## 6. Forkosh Statistical Analysis
### Permutation Tests
- PCA Stability: {results_dict['forkosh_statistics']['permutation_tests']['pca']['observed']:.3f}
  - p-value: {results_dict['forkosh_statistics']['permutation_tests']['pca']['p_value']:.3f}
- LDA Stability: {results_dict['forkosh_statistics']['permutation_tests']['lda']['observed']:.3f}
  - p-value: {results_dict['forkosh_statistics']['permutation_tests']['lda']['p_value']:.3f}

### Variance Explained (t-tests)
#### PCA Components
{pd.DataFrame({
    'Component': range(1, len(results_dict['forkosh_statistics']['ttest_results']['PCA']['var_explained'])+1),
    'Variance': results_dict['forkosh_statistics']['ttest_results']['PCA']['var_explained'],
    'p-value': results_dict['forkosh_statistics']['ttest_results']['PCA']['p_values']
}).to_markdown(index=False)}

#### LDA Components
{pd.DataFrame({
    'Component': range(1, len(results_dict['forkosh_statistics']['ttest_results']['LDA']['var_explained'])+1),
    'Variance': results_dict['forkosh_statistics']['ttest_results']['LDA']['var_explained'],
    'p-value': results_dict['forkosh_statistics']['ttest_results']['LDA']['p_values']
}).to_markdown(index=False)}

### Linear Regression (R² Analysis)
- PCA R² scores: {results_dict['forkosh_statistics']['regression_results']['PCA']['r2_scores']}
- LDA R² scores: {results_dict['forkosh_statistics']['regression_results']['LDA']['r2_scores']}

### ANOVA Results
#### PCA Components
{pd.DataFrame({
    'Component': range(1, len(results_dict['forkosh_statistics']['anova_results']['PCA']['f_statistics'])+1),
    'F-statistic': results_dict['forkosh_statistics']['anova_results']['PCA']['f_statistics'],
    'p-value': results_dict['forkosh_statistics']['anova_results']['PCA']['p_values']
}).to_markdown(index=False)}

#### LDA Components
{pd.DataFrame({
    'Component': range(1, len(results_dict['forkosh_statistics']['anova_results']['LDA']['f_statistics'])+1),
    'F-statistic': results_dict['forkosh_statistics']['anova_results']['LDA']['f_statistics'],
    'p-value': results_dict['forkosh_statistics']['anova_results']['LDA']['p_values']
}).to_markdown(index=False)}

### Feature Correlations
Significant correlations with behavioral features (p < 0.05):
"""
    
    # Add correlation analysis
    for space_name in ['PCA', 'LDA']:
        report_md += f"\n#### {space_name} Components\n"
        corr_results = results_dict['forkosh_statistics']['correlation_results'][space_name]
        for comp_idx in range(corr_results['correlations'].shape[0]):
            significant_idx = corr_results['p_values'][comp_idx] < 0.05
            if np.any(significant_idx):
                report_md += f"\nComponent {comp_idx + 1}:\n"
                for feat_idx in np.where(significant_idx)[0]:
                    report_md += f"- {feature_names[feat_idx]}: r = {corr_results['correlations'][comp_idx, feat_idx]:.3f} (p = {corr_results['p_values'][comp_idx, feat_idx]:.3f})\n"
    
    report_md += """
## 7. Files Generated
The following files contain detailed analysis results:
- `*_metrics.csv`: Summary metrics for both approaches
- `*_sequential_space.csv`: Identity space from sequential approach
- `*_parallel_pca_space.csv`: PCA space from parallel approach
- `*_parallel_lda_space.csv`: LDA space from parallel approach
"""
    
    # Save markdown report
    report_path = output_dir / f"{base_filename}_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    # Convert to HTML with basic styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Report - {base_filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; }}
            h4 {{ color: #95a5a6; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f6fa; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .feature-list {{ margin-left: 20px; }}
            .confidence-interval {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
    {markdown.markdown(report_md)}
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = output_dir / f"{base_filename}_analysis_report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return report_path, html_path

def save_results(results_dict, output_dir, base_filename):
    """Save analysis results and generate comprehensive report."""
    # Original metrics saving
    results_df = pd.DataFrame({
        'Metric': [
            'Sequential approach stability',
            'Sequential PCA variance explained',
            'Parallel PCA stability',
            'Parallel LDA stability',
            'Parallel PCA variance explained',
            'Original dimensions',
            'Sequential PCA dimensions'
        ],
        'Value': [
            results_dict['sequential_stability'],
            results_dict['sequential_pca_var'],
            results_dict['parallel_pca_stability'],
            results_dict['parallel_lda_stability'],
            results_dict['parallel_pca_var'],
            results_dict['original_dims'],
            results_dict['pca_dims']
        ]
    })
    results_df.to_csv(output_dir / f"{base_filename}_metrics.csv", index=False)
    
    # Save identity spaces
    for space_name, space_data in [
        ('sequential', results_dict['sequential_space']),
        ('parallel_pca', results_dict['parallel_pca_space']),
        ('parallel_lda', results_dict['parallel_lda_space'])
    ]:
        space_df = pd.DataFrame(
            space_data,
            columns=[f'Component_{i+1}' for i in range(space_data.shape[1])]
        )
        space_df.insert(0, 'mouse_id', results_dict['mouse_ids'])
        space_df.to_csv(output_dir / f"{base_filename}_{space_name}_space.csv", index=False)
    
    # Generate and save comprehensive report
    report_path, html_path = generate_analysis_report(
        results_dict,
        results_dict['significance_summary'],
        output_dir,
        base_filename
    )
    
    return report_path, html_path

# Main execution
if __name__ == "__main__":
    try:
        # Get analysis type and corresponding table name
        analysis_type, table_name = get_analysis_type()
        
        # Select CSV file to analyze
        print("\nSelect CSV file to analyze...")
        csv_path = select_csv_file(table_name)
        if not csv_path:
            sys.exit(1)
            
        # Create output directory
        output_dir = project_root / 'data' / f"{table_name}_analyzed"
        output_dir.mkdir(exist_ok=True)
        base_filename = Path(csv_path).stem
        
        # Read and process data
        print(f"Reading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        X, y = preprocess_data(df)
        
        # Standardization pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove highly correlated features
        corr_matrix = pd.DataFrame(X_scaled).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_filtered = np.delete(X_scaled, to_drop, axis=1)
        print(f"Removed {len(to_drop)} highly correlated features")
        
        # Run both analysis approaches
        print("\nRunning sequential analysis (current approach)...")
        seq_stability, seq_space, seq_pca_var = analyze_identity_domain_sequential(X_filtered, y)
        
        print("\nRunning parallel analysis (Forkosh approach)...")
        parallel_results = analyze_identity_domain_parallel(X_filtered, y)
        
        # Generate significance summary
        significance_summary = summarize_significance_results(parallel_results)
        
        print("\nSignificance Testing Results:")
        print("\nPCA Components:")
        print(f"Number of significant components: {significance_summary['PCA']['n_significant_components']}")
        print("Component p-values:", significance_summary['PCA']['component_p_values'])
        
        print("\nLDA Components:")
        print(f"Number of significant components: {significance_summary['LDA']['n_significant_components']}")
        print("Component p-values:", significance_summary['LDA']['component_p_values'])
        
        print("\nSignificant Features Summary:")
        for method in ['PCA', 'LDA']:
            print(f"\n{method} Significant Features:")
            for comp, features in significance_summary[method]['significant_features'].items():
                print(f"\n{comp}:")
                for feat in features:
                    print(f"- {feat}")
        
        # Save results
        results = {
            'sequential_stability': seq_stability,
            'sequential_space': seq_space,
            'sequential_pca_var': seq_pca_var,
            'parallel_pca_stability': parallel_results['pca_stability'],
            'parallel_lda_stability': parallel_results['lda_stability'],
            'parallel_pca_space': parallel_results['pca_space'],
            'parallel_lda_space': parallel_results['lda_space'],
            'parallel_pca_var': parallel_results['pca_explained_var'],
            'original_dims': X_filtered.shape[1],
            'pca_dims': seq_space.shape[1],
            'mouse_ids': y,
            'parallel_pca_feature_significance': parallel_results['pca_feature_significance'],
            'parallel_lda_feature_significance': parallel_results['lda_feature_significance'],
            'significance_summary': significance_summary,
            'forkosh_statistics': parallel_results['forkosh_statistics']
        }
        
        report_path, html_path = save_results(results, output_dir, base_filename)
        
        print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
        print("\nGenerated report files:")
        print(f"- Markdown report: {report_path}")
        print(f"- HTML report: {html_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
