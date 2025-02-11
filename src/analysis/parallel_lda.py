"""
Parallel LDA Analysis Module
Implements LDA part of Forkosh's parallel approach
"""
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from .identity_domain import IdentityDomainAnalyzer

def analyze_lda_parallel(X, y):
    """
    Parallel LDA analysis following Forkosh approach
    
    Args:
        X: Input features matrix
        y: Target labels (mouse IDs)
        
    Returns:
        dict: LDA analysis results containing:
            - transformed_space: Data in identity domain space
            - components: Identity domain components
            - eigenvalues: Corresponding eigenvalues
            - stability_score: Stability of identity domains
            - n_components: Number of significant components found
            - component_overlaps: Distribution overlap for each component
            - significant_components: Indices of components with < 5% overlap
            - discriminative_power: Between/within class variance ratio
            - feature_mask: Mask of selected features
    """
    # Print analysis info
    unique_mice = np.unique(y)
    print("\nLDA Analysis Info:")
    print(f"Input data shape: {X.shape}")
    print(f"Number of unique mice: {len(unique_mice)}")
    print("Samples per mouse:")
    for mouse in unique_mice:
        print(f"Mouse {mouse}: {np.sum(y == mouse)} samples")
    print()
    
    # 1. Data preprocessing
    # Standardization
   # scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    X_scaled = X
    # Remove low variance features
    #selector = VarianceThreshold(threshold=0.1)
    #X_filtered = selector.fit_transform(X_scaled)
    X_filtered = X_scaled
    # Quantile normalization for Gaussian-like distributions
    #qt = QuantileTransformer(output_distribution='normal')
    #X_transformed = qt.fit_transform(X_filtered)
    X_transformed = X_filtered
    # 2. Initial LDA with all possible components
    ida = IdentityDomainAnalyzer()
    # Maximum possible components is min(n_features, n_classes - 1)
    max_components = min(X_transformed.shape[1], len(unique_mice) - 1)
    print(f"Maximum possible LDA components: {max_components}")
    
    ida.fit(X_transformed, y, max_components=max_components)
    X_lda = ida.transform(X_transformed)  # Use all components initially
    
    # 3. Determine significant components based on distribution overlap
    overlaps = []
    for comp_idx in range(X_lda.shape[1]):
        comp_overlaps = []
        for mouse1_idx, mouse1 in enumerate(unique_mice):
            for mouse2 in unique_mice[mouse1_idx+1:]:
                data1 = X_lda[y == mouse1, comp_idx]
                data2 = X_lda[y == mouse2, comp_idx]
                
                # Calculate distribution overlap
                hist1, bins = np.histogram(data1, bins=50, density=True)
                hist2, _ = np.histogram(data2, bins=50, density=True)
                overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
                comp_overlaps.append(overlap)
        
        if comp_overlaps:
            overlaps.append(np.mean(comp_overlaps))
        else:
            overlaps.append(1.0)  # Maximum overlap if we can't compute
    
    # Keep components with < 5% overlap
    overlaps = np.array(overlaps)  # Convert to numpy array
    significant_components = np.where(overlaps < 0.05)[0]
    n_stable = len(significant_components)
    
    if n_stable == 0:
        # If no components meet the overlap criterion, use components with lowest overlap
        # and highest eigenvalues
        eigenvalue_ratio = ida.eigenvalues_[:len(overlaps)] / ida.eigenvalues_[:len(overlaps)].sum()
        cumsum = np.cumsum(eigenvalue_ratio)
        min_components = np.where(cumsum >= 0.80)[0][0] + 1
        # Combine overlap and eigenvalue information
        component_scores = overlaps / eigenvalue_ratio[:len(overlaps)]
        significant_components = np.argsort(component_scores)[:min_components]
        n_stable = len(significant_components)
    
    print(f"Found {n_stable} significant LDA components")
    
    # 4. Final LDA with selected components
    X_lda_final = X_lda[:, significant_components]
    components_final = ida.components_[significant_components]
    eigenvalues_final = ida.eigenvalues_[significant_components]
    
    # 5. Calculate stability scores
    stability_scores = []
    for mouse in unique_mice:
        mouse_data = X_lda_final[y == mouse]
        if len(mouse_data) > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    # 6. Calculate discriminative power
    between_class_var = np.zeros(n_stable)
    within_class_var = np.zeros(n_stable)
    
    for comp_idx in range(n_stable):
        class_means = [np.mean(X_lda_final[y == cls, comp_idx]) for cls in unique_mice]
        overall_mean = np.mean(X_lda_final[:, comp_idx])
        
        # Between-class variance
        between_class_var[comp_idx] = np.sum([(mean - overall_mean) ** 2 for mean in class_means])
        
        # Within-class variance
        within_class_var[comp_idx] = np.sum([
            np.sum((X_lda_final[y == cls, comp_idx] - class_means[i]) ** 2)
            for i, cls in enumerate(unique_mice)
        ])
    
    discriminative_power = between_class_var / (within_class_var + 1e-10)
    
    return {
        'transformed_space': X_lda_final,
        'components': components_final,
        'eigenvalues': eigenvalues_final,
        'stability_score': np.mean(stability_scores) if stability_scores else 0.0,
        'n_components': n_stable,
        'component_overlaps': overlaps,
        'significant_components': significant_components,
        'discriminative_power': discriminative_power,
        #'feature_mask': selector.get_support()  # For tracking which features were kept
    } 