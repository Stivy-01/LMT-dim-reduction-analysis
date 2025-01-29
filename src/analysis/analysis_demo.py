# analysis_demo.py (OPTIMIZED VERSION)
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

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

def analyze_identity_domain(X, y):
    ida = IdentityDomainAnalyzer(n_components=4)
    ida.fit(X, y)
    X_ids = ida.transform(X)
    
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_ids[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return np.mean(stability_scores), X_ids

if __name__ == "__main__":
    df = pd.read_csv('merged_for_lda_intervals.csv')
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
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_filtered)
    print(f"PCA reduced dimensions from {X_filtered.shape[1]} to {X_pca.shape[1]}")
    
    # Final analysis
    avg_stability, identity_space = analyze_identity_domain(X_pca, y)
    
    print(f"\nAverage ID stability score: {avg_stability:.3f}")
    print("First 5 rows in identity space:")
    print(identity_space[:5].round(2))  # Rounded for readability
