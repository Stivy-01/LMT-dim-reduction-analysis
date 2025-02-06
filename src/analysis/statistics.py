"""
Statistical Analysis Module
Implements Forkosh's statistical methods
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def perform_permutation_test(X, y, transformed_space, n_permutations=1000):
    """
    Perform permutation test for stability assessment
    """
    observed_stability = calculate_stability_scores(transformed_space, y)
    perm_stability = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_stability[i] = calculate_stability_scores(transformed_space, y_perm)
    
    return {
        'observed': observed_stability,
        'permuted_mean': np.mean(perm_stability),
        'p_value': np.mean(perm_stability >= observed_stability)
    }

def perform_variance_ttest(X, transformed_space, n_iterations=1000):
    """
    Perform one-sample t-test for variance explained
    """
    var_explained = np.var(transformed_space, axis=0)
    null_vars = np.zeros((n_iterations, transformed_space.shape[1]))
    
    for i in range(n_iterations):
        X_shuffled = np.copy(X)
        for j in range(X.shape[1]):
            np.random.shuffle(X_shuffled[:, j])
        null_vars[i] = np.var(X_shuffled @ transformed_space.T, axis=0)
    
    t_stats = []
    p_values = []
    for comp_idx in range(transformed_space.shape[1]):
        t_stat, p_val = stats.ttest_1samp(null_vars[:, comp_idx], var_explained[comp_idx])
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    return {
        'var_explained': var_explained,
        't_statistics': t_stats,
        'p_values': p_values
    }

def perform_regression_analysis(X, transformed_space):
    """
    Perform linear regression analysis with cross-validation
    """
    r2_scores = []
    for comp_idx in range(transformed_space.shape[1]):
        reg = LinearRegression()
        scores = cross_val_score(reg, X, transformed_space[:, comp_idx], cv=5, scoring='r2')
        r2_scores.append(np.mean(scores))
    
    return {'r2_scores': r2_scores}

def perform_anova(transformed_space, y):
    """
    Perform ANOVA across components
    """
    f_stats = []
    p_values = []
    for comp_idx in range(transformed_space.shape[1]):
        groups = [transformed_space[y == mouse_id, comp_idx] for mouse_id in np.unique(y)]
        f_stat, p_val = stats.f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_val)
    
    return {
        'f_statistics': f_stats,
        'p_values': p_values
    }

def perform_correlation_analysis(transformed_space, X, feature_names):
    """
    Perform Pearson correlation analysis
    """
    corr_matrix = np.zeros((transformed_space.shape[1], len(feature_names)))
    p_values = np.zeros((transformed_space.shape[1], len(feature_names)))
    
    for i in range(transformed_space.shape[1]):
        for j, feat in enumerate(feature_names):
            corr, p_val = stats.pearsonr(transformed_space[:, i], X[:, j])
            corr_matrix[i, j] = corr
            p_values[i, j] = p_val
    
    return {
        'correlations': corr_matrix,
        'p_values': p_values
    }

def calculate_stability_scores(X, y):
    """
    Calculate stability scores for a given transformed space
    """
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    return np.mean(stability_scores)

def perform_complete_statistical_analysis(X, y, transformed_space, feature_names):
    """
    Perform all statistical analyses
    
    Args:
        X: Original feature matrix
        y: Target labels (mouse IDs)
        transformed_space: Transformed data (PCA or LDA space)
        feature_names: List of feature names
        
    Returns:
        dict: Complete statistical analysis results
    """
    return {
        'permutation_tests': perform_permutation_test(X, y, transformed_space),
        'ttest_results': perform_variance_ttest(X, transformed_space),
        'regression_results': perform_regression_analysis(X, transformed_space),
        'anova_results': perform_anova(transformed_space, y),
        'correlation_results': perform_correlation_analysis(transformed_space, X, feature_names)
    } 