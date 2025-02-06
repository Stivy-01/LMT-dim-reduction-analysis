"""
Parallel PCA Analysis Module
Implements PCA part of Forkosh's parallel approach
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold

def analyze_pca_parallel(X, y):
    """
    Parallel PCA analysis following Forkosh approach
    
    Args:
        X: Input features matrix
        y: Target labels (mouse IDs)
        
    Returns:
        dict: PCA analysis results containing:
            - transformed_space: Data in PCA space
            - components: Principal components
            - eigenvalues: Corresponding eigenvalues
            - explained_variance_ratio: Explained variance ratio per component
            - stability_score: Stability of components
            - n_components: Number of significant components found
            - component_overlaps: Distribution overlap for each component
            - significant_components: Indices of components with < 5% overlap
            - feature_mask: Mask of selected features
    """
    # 1. Data preprocessing
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.1)
    X_filtered = selector.fit_transform(X_scaled)
    
    # Quantile normalization for Gaussian-like distributions
    qt = QuantileTransformer(output_distribution='normal')
    X_transformed = qt.fit_transform(X_filtered)
    
    # 2. Initial PCA with all possible components
    pca = PCA()  # No component limit - let the data decide
    X_pca = pca.fit_transform(X_transformed)
    
    # 3. Determine significant components based on distribution overlap
    overlaps = []
    for comp_idx in range(X_pca.shape[1]):
        comp_overlaps = []
        for mouse1_idx, mouse1 in enumerate(np.unique(y)):
            for mouse2 in np.unique(y)[mouse1_idx+1:]:
                data1 = X_pca[y == mouse1, comp_idx]
                data2 = X_pca[y == mouse2, comp_idx]
                
                # Calculate distribution overlap
                hist1, bins = np.histogram(data1, bins=50, density=True)
                hist2, _ = np.histogram(data2, bins=bins, density=True)
                overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
                comp_overlaps.append(overlap)
        
        overlaps.append(np.mean(comp_overlaps))
    
    # Keep components with < 5% overlap
    significant_components = np.where(np.array(overlaps) < 0.05)[0]
    n_stable = len(significant_components)
    
    if n_stable == 0:
        # If no components meet the overlap criterion, use components with lowest overlap
        # that explain at least 80% of variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        min_components = np.where(cumsum >= 0.80)[0][0] + 1
        significant_components = np.argsort(overlaps)[:min_components]
        n_stable = len(significant_components)
    
    # 4. Final PCA with selected components
    X_pca_final = X_pca[:, significant_components]
    components_final = pca.components_[significant_components]
    eigenvalues_final = pca.explained_variance_[significant_components]
    explained_var_final = pca.explained_variance_ratio_[significant_components]
    
    # 5. Calculate stability scores
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_pca_final[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return {
        'transformed_space': X_pca_final,
        'components': components_final,
        'eigenvalues': eigenvalues_final,
        'explained_variance_ratio': explained_var_final,
        'stability_score': np.mean(stability_scores),
        'n_components': n_stable,
        'component_overlaps': overlaps,
        'significant_components': significant_components,
        'feature_mask': selector.get_support()  # For tracking which features were kept
    } 