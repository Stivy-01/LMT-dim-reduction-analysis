import pandas as pd
import numpy as np
import random
from scipy.stats import gaussian_kde
from itertools import combinations

# -----------------------------
# Step 1: Data Input (No Normalization)
# -----------------------------

# Load the behavioral data from a CSV file.
# Assumes the CSV file "behavior_data.csv" contains columns:
# 'mouse_id', 'interval_start', and 204 behavior columns named "behavior_1", "behavior_2", ..., "behavior_204".
data = pd.read_csv(r"C:\Users\andre\Desktop\LMT dim reduction toolkit\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv")

# Define the list of behavior column names.
behavior_cols = [col for col in data.columns if col not in ['mouse_id', 'interval_start']]
    
# Create two dictionaries:
#   - data_dict: mapping each mouse_id to its daily 204-dimensional behavior vectors.
#   - n_days: mapping each mouse_id to the number of unique days (from "interval_start")
data_dict = {}
n_days = {}
for mouse_id, group in data.groupby('mouse_id'):
    # Convert the behavior columns into a numpy array for each mouse.
    vectors = group[behavior_cols].to_numpy()  # shape: (num_days, 204)
    data_dict[mouse_id] = vectors
    # Determine the number of unique days using the "interval_start" column if it exists.
    if 'interval_start' in group.columns:
        n_days[mouse_id] = group['interval_start'].nunique()
    else:
        n_days[mouse_id] = len(group)

# -----------------------------
# Step 2: Compute the Variability Matrices
# -----------------------------

dim = 204  # Dimension of the behavioral vectors

# a. Compute Within-Individual Variability (Sigma_w)
# Sigma_w = sum_m sum_d (x_m,d - mean_x_m) (x_m,d - mean_x_m)^T
Sigma_w = np.zeros((dim, dim))
for mouse_id, day_vectors in data_dict.items():
    mean_vector = np.mean(day_vectors, axis=0)
    for vector in day_vectors:
        diff = (vector - mean_vector).reshape(dim, 1)
        Sigma_w += diff @ diff.T

# b. Compute Between-Individual Variability (Sigma_b)
# Standard LDA uses a between-class scatter matrix:
# Sigma_b = sum_m n_m * (mean_m - mu)(mean_m - mu)^T,
# where n_m is the number of days for mouse m.
# First, compute the global mean (mu) over all mice and all days.
all_vectors = np.concatenate(list(data_dict.values()), axis=0)
mu = np.mean(all_vectors, axis=0)

Sigma_b = np.zeros((dim, dim))
for mouse_id, day_vectors in data_dict.items():
    n_m = n_days[mouse_id]  # number of unique days for this mouse
    mean_vector = np.mean(day_vectors, axis=0)
    diff = (mean_vector - mu).reshape(dim, 1)
    Sigma_b += n_m * (diff @ diff.T)

# -----------------------------
# Step 3: Implement LDA: Solve the Eigenvalue Problem
# -----------------------------

# Solve the generalized eigenvalue problem: Sigma_w^{-1} Sigma_b
Sigma_w_inv = np.linalg.inv(Sigma_w)
mat = Sigma_w_inv @ Sigma_b

# Compute eigenvalues and eigenvectors using eigh for Hermitian matrices
eigenvalues, eigenvectors = np.linalg.eigh(mat)
# Sort the eigenvectors in descending order by eigenvalue
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices].real  # ensure real eigenvalues
eigenvectors = eigenvectors[:, sorted_indices].real  # ensure real eigenvectors

# -----------------------------
# Step 4: Selection of Significant Identity Domains (IDs)
# -----------------------------

# Define a function to compute the overlap percentage.
def compute_overlap(eigenvector, data_dict):
    """
    Compute the average percentage overlap for a given eigenvector (ID) across all mice.
    
    Args:
        eigenvector (numpy array): The eigenvector representing a potential ID.
        data_dict (dict): Dictionary containing behavioral data for each mouse across multiple days.
    
    Returns:
        float: The average percentage overlap between mouse score distributions.
    """
    
    # Step 1: Project behavior vectors onto the eigenvector for each mouse
    scores = {}
    for mouse_id, behavior_vectors in data_dict.items():
        # Project each behavior vector onto the eigenvector
        scores[mouse_id] = np.dot(behavior_vectors, eigenvector)
    
    # Step 2: Compute pairwise overlaps between score distributions
    overlaps = []
    mouse_pairs = list(combinations(scores.keys(), 2))
    
    for mouse1, mouse2 in mouse_pairs:
        # Estimate the density of the score distributions using Gaussian KDE
        kde1 = gaussian_kde(scores[mouse1])
        kde2 = gaussian_kde(scores[mouse2])
        
        # Create a range of values over which to evaluate the KDEs
        score_range = np.linspace(
            min(np.min(scores[mouse1]), np.min(scores[mouse2])),
            max(np.max(scores[mouse1]), np.max(scores[mouse2])),
            1000
        )
        
        # Evaluate the KDEs on the score range
        density1 = kde1(score_range)
        density2 = kde2(score_range)
        
        # Calculate the overlap as the integral of the minimum of the two densities
        overlap = np.trapz(np.minimum(density1, density2), score_range)
        overlaps.append(overlap)
    
    # Step 3: Calculate the average percentage overlap across all pairs
    average_overlap = np.mean(overlaps) * 100  # Convert to percentage
    
    return average_overlap

overlap_threshold = 20 # Only eigenvectors with an overlap < 5% are considered significant.
significant_eigenvectors = []
for i in range(eigenvectors.shape[1]):
    eigvec = eigenvectors[:, i]
    overlap = compute_overlap(eigvec, data_dict)
    if overlap < overlap_threshold:
        significant_eigenvectors.append(eigvec)
        print(f"Found significant ID {len(significant_eigenvectors)} with overlap: {overlap:.2f}%")

# Form the EID matrix.
# Its shape is (number_of_significant_IDs, 204) and the number is determined by the overlap criterion.
EID = np.array(significant_eigenvectors)

# -----------------------------
# Step 5: Projection: Compute ID Scores for Each Mouse
# -----------------------------

# a. Compute the mean behavioral vector for each mouse.
mouse_mean_vectors = {}
for mouse_id, day_vectors in data_dict.items():
    mouse_mean_vectors[mouse_id] = np.mean(day_vectors, axis=0)  # shape: (204,)

# b. Project each mouse's mean behavior vector onto the EID matrix.
# The resulting score vector's dimension equals the number of significant IDs.
id_scores = {}
for mouse_id, mean_vector in mouse_mean_vectors.items():
    score = EID @ mean_vector  # (K, 204) x (204,) => (K,), where K = number of significant IDs.
    id_scores[mouse_id] = score

# -----------------------------
# Step 6: Output: Relationship of Each Mouse with the IDs
# -----------------------------

print("Number of significant Identity Domains (IDs):", EID.shape[0])

# -----------------------------
# Step 7: Analyze Behavior Contributions to IDs
# -----------------------------

# Function to get top contributing behaviors for an ID
def get_top_behaviors(eigenvector, behavior_names, n_top=10):
    """Get the behaviors that contribute most to an ID."""
    # Get absolute values of coefficients
    abs_coeffs = np.abs(eigenvector)
    # Get indices of top contributors
    top_indices = np.argsort(abs_coeffs)[::-1][:n_top]
    # Get the corresponding behaviors and coefficients
    top_behaviors = []
    for idx in top_indices:
        behavior = behavior_names[idx]
        coeff = eigenvector[idx]
        top_behaviors.append((behavior, coeff))
    return top_behaviors

# Save behavior contributions to a text file
behavior_output_path = 'data/identity_domains_behaviors.txt'
with open(behavior_output_path, 'w') as f:
    f.write("Analysis of Behavior Contributions to Identity Domains\n")
    f.write("=" * 50 + "\n\n")
    
    for id_num, eigenvector in enumerate(significant_eigenvectors, 1):
        f.write(f"Identity Domain {id_num}\n")
        f.write("-" * 20 + "\n")
        
        # Get top contributing behaviors
        top_behaviors = get_top_behaviors(eigenvector, behavior_cols)
        
        # Write positive and negative contributions separately
        f.write("\nTop Contributing Behaviors:\n")
        for behavior, coeff in top_behaviors:
            sign = '+' if coeff > 0 else '-'
            f.write(f"{sign} {behavior}: {abs(coeff):.4f}\n")
        f.write("\n")

print(f"\nBehavior analysis saved to: {behavior_output_path}")

# -----------------------------
# Step 8: Save Results to CSV
# -----------------------------

# Create a DataFrame from the ID scores
# First, determine the number of IDs to create column names
n_ids = len(next(iter(id_scores.values())))  # Get length of first score vector
id_columns = [f'ID_{i+1}' for i in range(n_ids)]

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame.from_dict(id_scores, orient='index', columns=id_columns)
results_df.index.name = 'mouse_id'
results_df.reset_index(inplace=True)

# Save to CSV
output_path = 'data/identity_space_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Save the EID matrix for future use
np.save('data/EID_matrix.npy', EID)
print(f"\nEID matrix saved to: data/EID_matrix.npy")

# -----------------------------
# Function to apply existing IDs to new data
# -----------------------------

def apply_ids_to_new_data(new_data_path, eid_matrix_path, behavior_cols):
    """
    Apply existing Identity Domains to a new dataset.
    
    Args:
        new_data_path: Path to the new (stress) dataset CSV
        eid_matrix_path: Path to the saved EID matrix
        behavior_cols: List of behavior columns to use
    
    Returns:
        DataFrame with ID scores for the new dataset
    """
    # Load the new data
    new_data = pd.read_csv(new_data_path)
    
    # Load the existing EID matrix
    EID = np.load(eid_matrix_path)
    
    # Create dictionary for the new data
    new_data_dict = {}
    for mouse_id, group in new_data.groupby('mouse_id'):
        vectors = group[behavior_cols].to_numpy()
        new_data_dict[mouse_id] = vectors
    
    # Compute mean vectors for each mouse
    new_mouse_mean_vectors = {}
    for mouse_id, day_vectors in new_data_dict.items():
        new_mouse_mean_vectors[mouse_id] = np.mean(day_vectors, axis=0)
    
    # Project onto existing IDs
    new_id_scores = {}
    for mouse_id, mean_vector in new_mouse_mean_vectors.items():
        score = EID @ mean_vector
        new_id_scores[mouse_id] = score
    
    # Create DataFrame with results
    n_ids = len(next(iter(new_id_scores.values())))
    id_columns = [f'ID_{i+1}' for i in range(n_ids)]
    results_df = pd.DataFrame.from_dict(new_id_scores, orient='index', columns=id_columns)
    results_df.index.name = 'mouse_id'
    results_df.reset_index(inplace=True)
    
    return results_df

# Example usage for stress dataset (commented out)
"""
# To apply to stress dataset:
stress_data_path = 'path_to_stress_dataset.csv'
stress_results = apply_ids_to_new_data(
    stress_data_path,
    'data/EID_matrix.npy',
    behavior_cols
)
stress_results.to_csv('data/stress_identity_space_results.csv', index=False)
"""

# -----------------------------
# Step 9: Calculate ID Stability Scores
# -----------------------------

def calculate_id_stability(data_dict, eigenvector):
    """
    Calculate stability score for an Identity Domain across time points.
    
    Args:
        data_dict: Dictionary containing behavioral data for each mouse
        eigenvector: The eigenvector representing the ID
        
    Returns:
        float: Stability score (0-1), where 1 is most stable
    """
    # Calculate scores for each mouse at each timepoint
    mouse_timepoint_scores = {}
    for mouse_id, behavior_vectors in data_dict.items():
        # Project each time point onto the ID
        scores = np.dot(behavior_vectors, eigenvector)
        mouse_timepoint_scores[mouse_id] = scores
    
    # Calculate within-mouse stability
    stability_scores = []
    for mouse_id, scores in mouse_timepoint_scores.items():
        if len(scores) > 1:  # Need at least 2 timepoints
            # Calculate coefficient of variation (inverse stability)
            cv = np.std(scores) / np.abs(np.mean(scores)) if np.mean(scores) != 0 else np.inf
            # Convert to stability score (1 - normalized cv)
            stability = 1 / (1 + cv)
            stability_scores.append(stability)
    
    # Return average stability across all mice
    return np.mean(stability_scores)

# Calculate stability for each ID
id_stability_scores = []
print("\nCalculating ID Stability Scores...")
for i, eigenvector in enumerate(significant_eigenvectors, 1):
    stability = calculate_id_stability(data_dict, eigenvector)
    id_stability_scores.append(stability)
    print(f"ID {i} Stability Score: {stability:.3f}")

# Save stability scores
stability_df = pd.DataFrame({
    'ID': [f'ID_{i+1}' for i in range(len(id_stability_scores))],
    'Stability_Score': id_stability_scores
})
stability_df.to_csv('data/id_stability_scores.csv', index=False)
print(f"\nStability scores saved to: data/id_stability_scores.csv")

def compare_id_stability_across_conditions(normal_data_dict, stress_data_dict, EID):
    """
    Compare ID stability between normal and stress conditions.
    
    Args:
        normal_data_dict: Dictionary containing normal condition data
        stress_data_dict: Dictionary containing stress condition data
        EID: Matrix of Identity Domain eigenvectors
    
    Returns:
        DataFrame with stability scores for both conditions
    """
    results = []
    for i, eigenvector in enumerate(EID, 1):
        normal_stability = calculate_id_stability(normal_data_dict, eigenvector)
        stress_stability = calculate_id_stability(stress_data_dict, eigenvector)
        
        results.append({
            'ID': f'ID_{i}',
            'Normal_Stability': normal_stability,
            'Stress_Stability': stress_stability,
            'Stability_Change': stress_stability - normal_stability
        })
    
    return pd.DataFrame(results)

# Example usage for comparing conditions (commented out)
"""
# After loading stress data:
stability_comparison = compare_id_stability_across_conditions(
    data_dict,  # normal condition
    stress_data_dict,  # stress condition
    EID
)
stability_comparison.to_csv('data/id_stability_comparison.csv', index=False)
"""
