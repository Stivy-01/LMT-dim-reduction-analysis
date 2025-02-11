from sklearn.decomposition import PCA
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.means_ = None
        self.scalings_ = None
        self.classes_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Compute class means and overall mean
        mean_overall = np.mean(X, axis=0)
        self.means_ = {cls: X[y == cls].mean(axis=0) for cls in self.classes_}
        
        # Compute between-class scatter matrix (Sb)
        Sb = np.zeros((n_features, n_features))
        for cls in self.classes_:
            n_cls = np.sum(y == cls)
            mean_cls = self.means_[cls].reshape(n_features, 1)
            mean_diff = mean_cls - mean_overall.reshape(n_features, 1)
            Sb += n_cls * (mean_diff @ mean_diff.T)

        # Compute within-class scatter matrix (Sw)
        Sw = np.zeros((n_features, n_features))
        for cls in self.classes_:
            X_cls = X[y == cls] - self.means_[cls]
            Sw += X_cls.T @ X_cls

        # Add small regularization term to Sw to make it invertible
        Sw += np.eye(Sw.shape[0]) * 1e-6

        # Solve the generalized eigenvalue problem: (Sw^-1 Sb) v = lambda v
        eigvals, eigvecs = scipy.linalg.eigh(Sb, Sw)
        
        # Sort eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[sorted_indices], eigvecs[:, sorted_indices]
        print(eigvals.shape)
        # Select top components
        self.n_components = min(n_classes - 1, X.shape[1]) if self.n_components is None else self.n_components
        self.scalings_ = eigvecs[:, :self.n_components]

    def transform(self, X):
        return X @ self.scalings_

# Generate synthetic data: 77 samples, 204 features, 28 classes
X, y = make_classification(n_samples=77, n_features=204, n_informative=50, 
                           n_redundant=50, n_classes=28, n_clusters_per_class=1, random_state=42)

# # Step 1: Use PCA to reduce dimensions (e.g., to 50 components)
# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(X)

# Step 2: Fit LDA on the reduced data
lda = LDA(n_components=None)
lda.fit(X, y)
X_lda = lda.transform(X)

# Plot the transformed data (first 2 LDA components)
plt.figure(figsize=(10, 7))
for cls in np.unique(y):
    plt.scatter(X_lda[y == cls, 0], X_lda[y == cls, 1], label=f'Class {cls}', alpha=0.6)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA Projection with PCA Preprocessing')
plt.legend(markerscale=0.5, bbox_to_anchor=(1, 1))
plt.show()
