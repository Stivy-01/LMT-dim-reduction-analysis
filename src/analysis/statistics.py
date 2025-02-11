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
    
    Args:
        X: Original feature matrix (n_samples, n_features)
        transformed_space: Transformed data matrix (n_samples, n_components)
        n_iterations: Number of permutation iterations
    """
    # Ensure inputs are numeric arrays
    X = np.asarray(X, dtype=np.float64)
    transformed_space = np.asarray(transformed_space, dtype=np.float64)
    
    # Calculate observed variance
    var_explained = np.var(transformed_space, axis=0)
    null_vars = np.zeros((n_iterations, transformed_space.shape[1]))
    
    # For null distribution, we shuffle each feature independently
    for i in range(n_iterations):
        # Shuffle rows of X
        X_shuffled = np.copy(X)
        for j in range(X.shape[1]):
            np.random.shuffle(X_shuffled[:, j])
            
        # Calculate variance of shuffled data projected onto components
        try:
            # For PCA, we can project directly
            if transformed_space.shape[1] <= X.shape[1]:
                # Estimate components using SVD
                U, S, Vt = np.linalg.svd(transformed_space, full_matrices=False)
                components = Vt.T[:, :transformed_space.shape[1]]
                null_projection = X_shuffled @ components
                null_vars[i] = np.var(null_projection, axis=0)
            else:
                # For other cases, just calculate variance of shuffled transformed space
                null_vars[i] = np.var(transformed_space[np.random.permutation(len(transformed_space))], axis=0)
        except ValueError as e:
            print(f"Warning: Error in iteration {i}: {e}")
            null_vars[i] = np.var(transformed_space, axis=0)
    
    # Perform t-test
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
    
    Args:
        transformed_space: Transformed data matrix (n_samples, n_components)
        y: Array of mouse IDs (n_samples,)
    """
    # Ensure inputs are numeric arrays
    transformed_space = np.asarray(transformed_space, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    
    if transformed_space.ndim == 1:
        transformed_space = transformed_space.reshape(-1, 1)
    
    f_stats = []
    p_values = []
    
    # Ensure we have enough samples and components
    if transformed_space.shape[1] == 0:
        return {'f_statistics': np.array([]), 'p_values': np.array([])}
    
    for comp_idx in range(transformed_space.shape[1]):
        try:
            # Extract the component data
            comp_data = transformed_space[:, comp_idx]
            
            # Create groups for each mouse ID
            unique_mice = np.unique(y)
            groups = []
            for mouse_id in unique_mice:
                mouse_data = comp_data[y == mouse_id]
                if len(mouse_data) > 0:  # Include all data points
                    groups.append(mouse_data)
            
            if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                # Perform one-way ANOVA
                f_stat, p_val = stats.f_oneway(*groups)
                if not np.isnan(f_stat) and not np.isnan(p_val):
                    f_stats.append(f_stat)
                    p_values.append(p_val)
                else:
                    f_stats.append(0.0)
                    p_values.append(1.0)
            else:
                f_stats.append(0.0)
                p_values.append(1.0)
        except ValueError as e:
            print(f"Warning: Could not perform ANOVA for component {comp_idx}. Error: {e}")
            f_stats.append(0.0)
            p_values.append(1.0)
    
    return {
        'f_statistics': np.array(f_stats),
        'p_values': np.array(p_values)
    }

def perform_correlation_analysis(transformed_space, X, feature_names):
    """
    Perform Pearson correlation analysis
    
    Args:
        transformed_space: Transformed data matrix (n_samples, n_components)
        X: Original feature matrix (n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        dict: Correlation analysis results
    """
    # Ensure inputs are numeric arrays
    transformed_space = np.asarray(transformed_space, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    # Validate feature_names
    if feature_names is None:
        print("Warning: No feature names provided")
        return {
            'correlations': np.array([]),
            'p_values': np.array([])
        }
    
    # Ensure feature_names is a list of strings
    feature_names = [str(name) for name in feature_names]
    
    # Ensure feature_names matches X shape
    if len(feature_names) > X.shape[1]:
        feature_names = feature_names[:X.shape[1]]
        print(f"Warning: Truncating feature_names to match X shape ({X.shape[1]} features)")
    elif len(feature_names) < X.shape[1]:
        print(f"Warning: feature_names length ({len(feature_names)}) doesn't match X shape ({X.shape[1]})")
        return {
            'correlations': np.array([]),
            'p_values': np.array([])
        }
    
    try:
        corr_matrix = np.zeros((transformed_space.shape[1], len(feature_names)))
        p_values = np.zeros((transformed_space.shape[1], len(feature_names)))
        
        for i in range(transformed_space.shape[1]):
            for j in range(len(feature_names)):
                try:
                    # Ensure we're working with numeric data
                    x = transformed_space[:, i].astype(np.float64)
                    y = X[:, j].astype(np.float64)
                    
                    # Check for invalid values
                    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or \
                       np.any(np.isinf(x)) or np.any(np.isinf(y)):
                        print(f"Warning: Invalid values found in component {i} or feature {feature_names[j]}")
                        corr_matrix[i, j] = 0
                        p_values[i, j] = 1.0
                        continue
                    
                    corr, p_val = stats.pearsonr(x, y)
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                    p_values[i, j] = p_val if not np.isnan(p_val) else 1.0
                except ValueError as e:
                    print(f"Warning: Could not compute correlation for component {i} and feature {feature_names[j]}: {e}")
                    corr_matrix[i, j] = 0
                    p_values[i, j] = 1.0
        
        return {
            'correlations': corr_matrix,
            'p_values': p_values
        }
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        return {
            'correlations': np.array([]),
            'p_values': np.array([])
        }

def calculate_stability_scores(X, y):
    """
    Calculate stability scores for a given transformed space
    
    Args:
        X: Transformed data matrix (n_samples, n_features)
        y: Array of mouse IDs (n_samples,)
    """
    # Ensure inputs are numeric arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    stability_scores = []
    unique_mice = np.unique(y)
    
    # If we only have one sample per mouse, return 0
    if all(np.sum(y == mouse_id) == 1 for mouse_id in unique_mice):
        return 0.0
    
    for mouse in unique_mice:
        mouse_data = X[y == mouse]
        if mouse_data.shape[0] > 1:  # Only calculate if we have more than one sample
            try:
                # Calculate correlation matrix
                corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
                # Handle potential NaN values
                if np.any(np.isnan(corr_matrix)):
                    corr_matrix = np.nan_to_num(corr_matrix, 0)
                np.fill_diagonal(corr_matrix, np.nan)
                score = np.nanmean(corr_matrix)
                if not np.isnan(score):
                    stability_scores.append(score)
            except ValueError as e:
                print(f"Warning: Could not calculate correlation for mouse {mouse}. Error: {e}")
                continue
    
    return np.mean(stability_scores) if stability_scores else 0.0

def perform_complete_statistical_analysis(X, y, pca_space, lda_space=None, feature_names=None):
    """
    Perform all statistical analyses
    
    Args:
        X: Original feature matrix
        y: Target labels (mouse IDs)
        pca_space: PCA transformed data
        lda_space: LDA transformed data (optional)
        feature_names: List of feature names (optional)
        
    Returns:
        dict: Complete statistical analysis results
    """
    results = {}
    
    # PCA Statistics
    if pca_space is not None:
        results['pca'] = {
            'permutation_tests': perform_permutation_test(X, y, pca_space),
            'ttest_results': perform_variance_ttest(X, pca_space),
            'regression_results': perform_regression_analysis(X, pca_space),
            'anova_results': perform_anova(pca_space, y)
        }
        if feature_names is not None:
            results['pca']['correlation_results'] = perform_correlation_analysis(pca_space, X, feature_names)
    
    # LDA Statistics
    if lda_space is not None:
        results['lda'] = {
            'permutation_tests': perform_permutation_test(X, y, lda_space),
            'ttest_results': perform_variance_ttest(X, lda_space),
            'regression_results': perform_regression_analysis(X, lda_space),
            'anova_results': perform_anova(lda_space, y)
        }
        if feature_names is not None:
            results['lda']['correlation_results'] = perform_correlation_analysis(lda_space, X, feature_names)
    
    return results 