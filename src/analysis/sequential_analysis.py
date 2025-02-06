"""
Sequential Analysis Module
Implements PCA followed by LDA approach
"""
import numpy as np
from sklearn.decomposition import PCA
from .identity_domain import IdentityDomainAnalyzer

def analyze_identity_domain_sequential(X, y, variance_threshold=0.80):
    """
    Current approach: PCA followed by LDA
    
    Args:
        X: Input features matrix
        y: Target labels (mouse IDs)
        variance_threshold: PCA variance threshold (default: 0.80)
        
    Returns:
        tuple: (stability_score, transformed_space, pca_variance_explained)
    """
    # First reduce dimensions with PCA
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
    
    # Then apply LDA
    ida = IdentityDomainAnalyzer()
    ida.fit(X_pca, y)
    
    # Determine number of significant components based on eigenvalues
    # Use Kaiser criterion (eigenvalues > 1) as initial filter
    significant_components = np.where(ida.eigenvalues_ > 1)[0]
    if len(significant_components) == 0:
        n_components = min(4, len(ida.eigenvalues_))
    else:
        n_components = len(significant_components)
    
    X_ids = ida.transform(X_pca, n_components)
    
    # Calculate stability scores
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_ids[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return np.mean(stability_scores), X_ids, pca.explained_variance_ratio_.sum() 