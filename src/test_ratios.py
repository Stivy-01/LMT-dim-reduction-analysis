# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.feature_ratios import (
    calculate_behavioral_ratios,
    calculate_composite_scores,
    prepare_pca_features,
    get_feature_descriptions
)

def load_data():
    """Load behavioral data from the database."""
    conn = sqlite3.connect(r"C:\Users\andre\Desktop\exps\lda_database.sqlite")
    df = pd.read_sql_query("SELECT * FROM merged_data", conn)
    conn.close()
    return df

def plot_ratio_distributions(ratios_df, output_dir="ratio_plots", n_bins=20):
    """Plot distributions of all ratios to check for outliers and patterns."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature descriptions
    descriptions = get_feature_descriptions()
    
    # Set up a color palette for different mice
    unique_mice = sorted(ratios_df['mouse_id'].unique())  # Sort mouse IDs
    n_mice = len(unique_mice)
    
    # Define highly distinct colors
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
    ][:n_mice]  # Take only as many colors as we have mice
    
    mouse_colors = dict(zip(unique_mice, distinct_colors))
    
    # Plot ratios
    for col in ratios_df.columns:
        if col in ['mouse_id', 'date', 'interval_start']:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Create subplot layout
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1])
        ax_hist = plt.subplot(gs[0])  # Main histogram
        ax_box = plt.subplot(gs[1])   # Boxplot below
        
        # Calculate common bin edges for all mice
        all_data = ratios_df[col].dropna()
        if len(all_data) == 0:
            continue
            
        # Create common bin edges with fixed number of bins
        data_min = all_data.min()
        data_max = all_data.max()
        bin_edges = np.linspace(data_min, data_max, n_bins + 1)
        
        # Dictionary to store histogram data for overlap detection
        hist_data = {}
        
        # First pass: collect histogram data for each mouse
        for mouse_id in unique_mice:
            mouse_data = ratios_df[ratios_df['mouse_id'] == mouse_id][col].dropna()
            if not mouse_data.empty:
                counts, _ = np.histogram(mouse_data, bins=bin_edges)
                hist_data[mouse_id] = counts
        
        # Find overlaps
        overlaps = {}
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        for bin_idx in range(len(bin_centers)):
            mice_in_bin = []
            for mouse_id, counts in hist_data.items():
                if counts[bin_idx] > 0:
                    mice_in_bin.append(mouse_id)
            if len(mice_in_bin) > 1:
                overlaps[bin_idx] = mice_in_bin
        
        # Plot individual mouse histograms
        max_height = 0
        for mouse_id in reversed(unique_mice):
            mouse_data = ratios_df[ratios_df['mouse_id'] == mouse_id][col].dropna()
            if not mouse_data.empty:
                # Plot histogram
                n, _, patches = ax_hist.hist(mouse_data, bins=bin_edges, alpha=0.4,
                                          label=f'Mouse {mouse_id}',
                                          color=mouse_colors[mouse_id],
                                          edgecolor='black',
                                          linewidth=1)
                max_height = max(max_height, n.max())
                
                # Add boxplot
                bp = ax_box.boxplot(mouse_data, positions=[mouse_id], 
                                  widths=0.7,
                                  patch_artist=True,
                                  medianprops=dict(color="black", linewidth=2),
                                  flierprops=dict(marker='o', 
                                                markerfacecolor=mouse_colors[mouse_id],
                                                markersize=8))
                for patch in bp['boxes']:
                    patch.set_facecolor(mouse_colors[mouse_id])
                    patch.set_alpha(0.4)
                    patch.set_edgecolor('black')
        
        # Add overlap labels
        for bin_idx, mice_list in overlaps.items():
            if len(mice_list) > 1:
                x = bin_centers[bin_idx]
                y = sum(hist_data[mouse_id][bin_idx] for mouse_id in mice_list)
                label = f"Mice {'+'.join(map(str, mice_list))}"
                ax_hist.text(x, y, label, ha='center', va='bottom', 
                           fontsize=8, rotation=90)
        
        # Customize histogram (top plot)
        feature_name = col.replace('_', ' ').title()
        description = descriptions.get(col, '')
        ax_hist.set_title(f"Distribution of {feature_name}\n{description}", 
                         fontsize=12, pad=20)
        ax_hist.set_ylabel("Number of Occurrences", fontsize=10)
        
        # Add legend with example of overlap
        ax_hist.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Customize boxplot (bottom plot)
        ax_box.set_xlabel(f"{feature_name} Value", fontsize=10)
        ax_box.set_yticks([])  # Hide y-axis ticks
        
        # Add mouse ID labels to boxplot
        for mouse_id in unique_mice:
            ax_box.text(-0.1, mouse_id, f'Mouse {mouse_id}', 
                       ha='right', va='center')
        
        # Add explanation text
        explanation = (
            "How to read this plot:\n\n"
            "TOP PLOT (Histogram):\n"
            "• Shows how often each value occurs\n"
            "• Taller bars = More frequent values\n"
            "• Different colors = Different mice\n"
            "• When bars overlap, colors blend together\n"
            "• Labels above bars show overlapping mice\n"
            "• Black edges help see individual bars\n\n"
            "BOTTOM PLOT (Box Plot):\n"
            "• Box = Middle 50% of values\n"
            "• Thick black line = Most common value\n"
            "• Whiskers = Range of typical values\n"
            "• Dots = Unusual values (outliers)"
        )
        
        # Add statistics for each mouse
        stats_text = []
        for mouse_id in unique_mice:
            mouse_data = ratios_df[ratios_df['mouse_id'] == mouse_id][col].dropna()
            if not mouse_data.empty:
                stats = f"Mouse {mouse_id} Summary:\n"
                stats += f"• Typical value: {mouse_data.median():.2f}\n"
                stats += f"• Normal range: {mouse_data.quantile(0.25):.2f} to {mouse_data.quantile(0.75):.2f}\n"
                stats += f"• Total measurements: {len(mouse_data)}"
                stats_text.append(stats)
        
        # Create a single text box with both explanation and statistics
        full_text = explanation + "\n\n" + "STATISTICS:\n" + "\n\n".join(stats_text)
        plt.figtext(1.02, 0.5, full_text,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=9, ha='left', va='center')
        
        # Adjust layout
        plt.subplots_adjust(right=0.7, hspace=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()

def check_ratio_validity(ratios_df):
    """Check ratios for invalid values (NaN, Inf) and extreme outliers."""
    print("\nRatio Validity Check:")
    for col in ratios_df.columns:
        if col in ['mouse_id', 'date', 'interval_start']:
            continue
            
        n_nan = ratios_df[col].isna().sum()
        n_inf = np.isinf(ratios_df[col]).sum()
        n_outliers = ((ratios_df[col] - ratios_df[col].mean()).abs() > 
                     3 * ratios_df[col].std()).sum()
        
        if n_nan > 0 or n_inf > 0 or n_outliers > 0:
            print(f"\n{col}:")
            print(f"  NaN values: {n_nan}")
            print(f"  Infinite values: {n_inf}")
            print(f"  Outliers (>3 std): {n_outliers}")
            
            if n_outliers > 0:
                outliers = ratios_df[abs(ratios_df[col] - ratios_df[col].mean()) > 
                                   3 * ratios_df[col].std()][col]
                print(f"  Outlier values: {outliers.values}")

def analyze_correlations(df_with_scores):
    """Analyze correlations between features and composite scores."""
    # Select relevant columns
    score_cols = [col for col in df_with_scores.columns if col.endswith('_score')]
    ratio_cols = [col for col in df_with_scores.columns 
                 if col.endswith('_ratio') and col in df_with_scores.columns]
    
    # Calculate correlations
    corr_matrix = df_with_scores[score_cols + ratio_cols].corr()
    
    # Create a more readable correlation heatmap
    plt.figure(figsize=(15, 12))
    
    # Use seaborn for better heatmap visualization
    sns.heatmap(corr_matrix, 
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                annot=True,  # Add correlation values
                fmt='.2f',   # Format correlation values
                square=True,
                cbar_kws={"shrink": .8},
                xticklabels=[col.replace('_', ' ').title() for col in corr_matrix.columns],
                yticklabels=[col.replace('_', ' ').title() for col in corr_matrix.columns])
    
    plt.title("Feature Correlations Heatmap", pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print strongest correlations
    print("\nStrongest Correlations:")
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlations.append((
                corr_matrix.index[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr in correlations[:10]:
        feat1_name = feat1.replace('_', ' ').title()
        feat2_name = feat2.replace('_', ' ').title()
        print(f"{feat1_name} vs {feat2_name}: {corr:.3f}")

def main():
    print("Loading data...")
    df = load_data()
    
    print("\nCalculating ratios...")
    ratios_df = calculate_behavioral_ratios(df)
    
    print("\nChecking ratio validity...")
    check_ratio_validity(ratios_df)
    
    print("\nPlotting ratio distributions...")
    plot_ratio_distributions(ratios_df)
    
    print("\nCalculating composite scores...")
    df_with_scores = calculate_composite_scores(ratios_df)
    
    print("\nAnalyzing feature correlations...")
    analyze_correlations(df_with_scores)
    
    print("\nPreparing PCA features...")
    df_normalized = prepare_pca_features(df)
    
    print("\nSummary of features:")
    print(f"Total features: {len(df_normalized.columns) - 3}")  # Excluding mouse_id, date, interval_start
    print("\nFeature descriptions:")
    for feature, description in get_feature_descriptions().items():
        feature_name = feature.replace('_', ' ').title()
        print(f"{feature_name}: {description}")

if __name__ == "__main__":
    main() 