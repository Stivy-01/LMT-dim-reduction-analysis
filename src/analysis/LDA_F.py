import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Data Input and Preprocessing
# a. Load the Behavioral Data
data = pd.read_csv(r"C:\Users\andre\Desktop\LMT dim reduction toolkit\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv")

# Organize data into a dictionary where for each mouse m and day d we have a behavioral vector x[m][d]
x = {}  # This will store our behavioral vectors

# First, identify all unique mouse IDs and days
mouse_ids = data['mouse_id'].unique()
days = data['time_interval'].unique()

# Create behavioral vectors for each mouse and day
for mouse_id in mouse_ids:
    x[mouse_id] = {}
    mouse_data = data[data['mouse_id'] == mouse_id]
    
    for day in days:
        day_data = mouse_data[mouse_data['time_interval'] == day]
        if len(day_data) > 0:
            # Extract the 204 behavioral features
            behavior_cols = [f'behavior_{i+1}' for i in range(204)]
            behavior_vector = day_data[behavior_cols].values[0]  # 1x204 vector
            x[mouse_id][day] = behavior_vector

# Print some basic information about the data
print(f"Number of mice: {len(mouse_ids)}")
print(f"Number of days: {len(days)}")
print(f"Shape of each behavioral vector: {x[mouse_ids[0]][days[0]].shape}")

# Step 2: LDA Mathematical Formulation
# The objective is to find projection vector w that maximizes:
# w = argmax_w (w^T Σ_b w)/(w^T Σ_w w)
# This leads to the eigenvalue problem: Σ_b w = λ Σ_w w
# We solve this by computing eigenvectors of Σ_w^(-1) Σ_b

# Step 3: Compute Variability Matrices
def compute_mean_vector(mouse_data):
    """Compute mean behavior vector for a mouse over all days"""
    return np.mean(list(mouse_data.values()), axis=0)

# Step 3a: Compute Within-Individual Variability (Σ_w)
Sigma_w = np.zeros((204, 204))  # Initialize 204x204 matrix

# For each mouse, compute deviation from its mean behavior
for mouse_id in x:
    mouse_mean = compute_mean_vector(x[mouse_id])
    
    # For each day, compute difference from mean and outer product
    for day in x[mouse_id]:
        diff = x[mouse_id][day] - mouse_mean
        diff = diff.reshape(-1, 1)  # Make column vector
        Sigma_w += diff @ diff.T  # Outer product

# Step 3b: Compute Between-Individual Variability (Σ_b)
# First compute global mean
all_vectors = []
for mouse_id in x:
    all_vectors.extend(list(x[mouse_id].values()))
global_mean = np.mean(all_vectors, axis=0)

# Now compute Σ_b
Sigma_b = np.zeros((204, 204))
D = len(days)  # Number of time intervals

for mouse_id in x:
    # Sum of mouse's behavioral vectors
    mouse_sum = sum(x[mouse_id].values())
    # Difference from global mean
    diff = mouse_sum - global_mean
    diff = diff.reshape(-1, 1)  # Make column vector
    Sigma_b += D * (diff @ diff.T)  # Scale by D and add outer product

# Step 4: Solve the Eigenvalue Problem
# We need to solve Σ_w^(-1) Σ_b w = λw
print("\nSolving eigenvalue problem...")

# Add small regularization to ensure Sigma_w is invertible
epsilon = 1e-10
Sigma_w_reg = Sigma_w + epsilon * np.eye(204)

# Compute inverse of Sigma_w
try:
    # Try using more stable solver first
    Sigma_w_inv = linalg.solve(Sigma_w_reg, np.eye(204), assume_a='pos')
except np.linalg.LinAlgError:
    print("Warning: Using pseudo-inverse due to singular matrix")
    Sigma_w_inv = np.linalg.pinv(Sigma_w_reg)

# Compute the matrix for eigendecomposition
mat = Sigma_w_inv @ Sigma_b

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(mat)

# Sort by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Select Significant Identity Domains
def compute_overlap(eigenvector, data_dict):
    """
    Compute overlap percentage for an eigenvector.
    Lower overlap indicates more distinct individual differences.
    """
    projections = []
    for mouse_id, mouse_data in data_dict.items():
        # Project each day's data onto the eigenvector
        for day_vector in mouse_data.values():
            projection = day_vector @ eigenvector
            projections.append(projection)
    
    projections = np.array(projections)
    
    # Compute variance of projections
    projection_variance = np.var(projections)
    
    # Compute overlap (lower is better)
    overlap = 1.0 / (1.0 + projection_variance)
    return overlap * 100  # Convert to percentage

# Select significant eigenvectors (IDs) based on overlap criterion
overlap_threshold = 5.0  # 5% threshold as mentioned in roadmap
significant_eigenvectors = []
overlaps = []

print("\nSelecting significant Identity Domains...")
for i in range(len(eigenvalues)):
    overlap = compute_overlap(eigenvectors[:, i], x)
    overlaps.append(overlap)
    if overlap < overlap_threshold:
        significant_eigenvectors.append(eigenvectors[:, i])

# Convert to numpy array for easier manipulation
EID = np.array(significant_eigenvectors)  # Each row is a significant ID

# Step 6: Projection - Compute ID Scores for Each Mouse
print("\nComputing ID scores for each mouse...")

# First compute mean behavioral vector for each mouse
mouse_mean_vectors = {}
for mouse_id in x:
    mouse_mean_vectors[mouse_id] = compute_mean_vector(x[mouse_id])

# Project onto the EID matrix to get ID scores
id_scores = {}
for mouse_id, mean_vector in mouse_mean_vectors.items():
    # Project mean vector onto EID matrix
    # EID: (K × 204) and mean_vector: (204,), resulting in a (K,) vector
    score = EID @ mean_vector
    id_scores[mouse_id] = score

# Save results to CSV
results_df = pd.DataFrame.from_dict(id_scores, orient='index')
# Name columns as ID1, ID2, etc.
results_df.columns = [f'ID{i+1}' for i in range(results_df.shape[1])]
results_df.index.name = 'mouse_id'
results_df.to_csv('data/identity_scores.csv')

# Step 7: Analyze Relationships between Mice and IDs
print("\nAnalyzing relationships between mice and their Identity Domains...")

def analyze_id_relationships(id_scores_df, eigenvectors, behavior_cols):
    """Analyze and visualize relationships between IDs and behaviors"""
    
    # Create directory for plots if it doesn't exist
    import os
    os.makedirs('data/id_analysis', exist_ok=True)
    
    # 7.1: Correlation Analysis between IDs
    plt.figure(figsize=(10, 8))
    id_correlations = id_scores_df.corr()
    sns.heatmap(id_correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlations between Identity Domains')
    plt.tight_layout()
    plt.savefig('data/id_analysis/id_correlations.png')
    plt.close()
    
    # 7.2: Hierarchical Clustering of Mice based on ID scores
    plt.figure(figsize=(12, 8))
    linkage_matrix = linkage(id_scores_df, method='ward')
    dendrogram(linkage_matrix, labels=id_scores_df.index)
    plt.title('Hierarchical Clustering of Mice based on Identity Domains')
    plt.xlabel('Mouse ID')
    plt.ylabel('Distance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('data/id_analysis/mouse_clustering.png')
    plt.close()
    
    # 7.3: Analyze behavioral components of each ID
    id_components = pd.DataFrame(
        EID,
        columns=behavior_cols,
        index=[f'ID{i+1}' for i in range(len(EID))]
    )
    
    # Find top contributing behaviors for each ID
    n_top_behaviors = 10
    top_behaviors = {}
    for id_name in id_components.index:
        components = id_components.loc[id_name].abs()
        top_idx = components.nlargest(n_top_behaviors).index
        top_behaviors[id_name] = list(zip(top_idx, components[top_idx]))
    
    # Save behavioral components analysis
    with open('data/id_analysis/id_behavioral_components.txt', 'w') as f:
        f.write("Top Contributing Behaviors for each Identity Domain:\n\n")
        for id_name, behaviors in top_behaviors.items():
            f.write(f"\n{id_name}:\n")
            for behavior, weight in behaviors:
                f.write(f"  {behavior}: {weight:.4f}\n")
    
    # 7.4: Visualize distribution of ID scores
    plt.figure(figsize=(12, 6))
    id_scores_df.boxplot()
    plt.title('Distribution of Identity Domain Scores')
    plt.xlabel('Identity Domain')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/id_analysis/id_score_distributions.png')
    plt.close()
    
    return id_correlations, top_behaviors

if __name__ == "__main__":
    # Print diagnostic information
    print("\nMatrix Condition Numbers:")
    print(f"Sigma_w condition number: {np.linalg.cond(Sigma_w)}")
    print(f"Sigma_b condition number: {np.linalg.cond(Sigma_b)}")
    
    print("\nMatrix Ranks:")
    print(f"Sigma_w rank: {np.linalg.matrix_rank(Sigma_w)}")
    print(f"Sigma_b rank: {np.linalg.matrix_rank(Sigma_b)}")
    
    print(f"\nNumber of significant Identity Domains found: {len(EID)}")
    print("\nTop 10 eigenvalues:")
    print(eigenvalues[:10])
    
    print("\nOverlap percentages for top 10 eigenvectors:")
    print(np.array(overlaps[:10]))
    
    # Print information about ID scores
    print("\nID Scores Summary:")
    print(f"Number of mice: {len(id_scores)}")
    print(f"Number of IDs per mouse: {len(EID)}")
    print("\nFirst few rows of ID scores:")
    print(results_df.head())
    
    # Basic statistical summary of ID scores
    print("\nID scores statistical summary:")
    print(results_df.describe())
    
    # Run relationship analysis
    print("\nAnalyzing ID relationships and generating visualizations...")
    id_correlations, top_behaviors = analyze_id_relationships(
        results_df, 
        EID,
        [f'behavior_{i+1}' for i in range(204)]
    )
    
    # Print additional analysis results
    print("\nID Correlations Summary:")
    print("Average absolute correlation between IDs:", 
          np.mean(np.abs(id_correlations.values - np.eye(len(EID)))))
    
    print("\nTop 5 most correlated ID pairs:")
    corr_pairs = []
    for i in range(len(EID)):
        for j in range(i+1, len(EID)):
            corr_pairs.append((f'ID{i+1}', f'ID{j+1}', id_correlations.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for id1, id2, corr in corr_pairs[:5]:
        print(f"{id1} - {id2}: {corr:.3f}")
    
    print("\nAnalysis results have been saved to 'data/id_analysis/' directory")
    pass 