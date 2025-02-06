"""
Analysis Pipeline Module
======================

Orchestrates the complete analysis workflow using modular components.
"""

import pandas as pd
from pathlib import Path
from .preprocessing import preprocess_data
from .sequential_analysis import analyze_identity_domain_sequential
from .parallel_pca import analyze_pca_parallel
from .parallel_lda import analyze_lda_parallel
from .statistics import perform_complete_statistical_analysis
from .reporting import generate_report

class AnalysisPipeline:
    """
    Orchestrates the complete behavioral analysis pipeline.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory for saving results (default: data/analyzed)
        """
        self.output_dir = output_dir or Path('data/analyzed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_analysis(self, data_path, analysis_type="both", correlation_threshold=0.95, variance_threshold=0.1, progress_callback=None):
        """
        Run analysis pipeline.
        
        Args:
            data_path: Path to input data CSV
            analysis_type: Type of analysis to run ("sequential", "parallel", or "both")
            correlation_threshold: Threshold for correlation filtering (default: 0.95)
            variance_threshold: Threshold for variance filtering (default: 0.1)
            progress_callback: Optional callback function to report progress
            
        Returns:
            dict: Complete analysis results
        """
        def update_progress(msg):
            print(msg)
            if progress_callback:
                progress_callback(f"Status: {msg}")

        update_progress(f"Starting analysis pipeline for: {Path(data_path).name}")
        
        # 1. Load and preprocess data
        update_progress("Loading and preprocessing data...")
        df = pd.read_csv(data_path)
        X, mouse_ids, cv_splits, feature_names = preprocess_data(
            df,
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold
        )
        
        results = {
            'original_dims': X.shape[1],
            'mouse_ids': mouse_ids,
            'feature_names': feature_names,
            'cv_splits': cv_splits
        }
        
        # 2. Run selected analyses
        if analysis_type in ["sequential", "both"]:
            update_progress("Running sequential analysis...")
            seq_stability, seq_space, seq_pca_var = analyze_identity_domain_sequential(X, mouse_ids)
            results.update({
                'sequential_stability': seq_stability,
                'sequential_space': seq_space,
                'sequential_pca_var': seq_pca_var
            })
            update_progress(f"Sequential analysis complete. Stability score: {seq_stability:.3f}")
        
        if analysis_type in ["parallel", "both"]:
            update_progress("Running parallel PCA...")
            pca_results = analyze_pca_parallel(X, mouse_ids)
            update_progress(f"PCA complete. Found {pca_results['n_components']} significant components")
            
            update_progress("Running parallel LDA...")
            lda_results = analyze_lda_parallel(X, mouse_ids)
            update_progress(f"LDA complete. Found {lda_results['n_components']} significant components")
            
            results.update({
                'pca_results': pca_results,
                'lda_results': lda_results,
                'pca_dims': pca_results['transformed_space'].shape[1]
            })
        
        # 3. Run statistical analysis based on available results
        update_progress("Performing statistical analysis...")
        if analysis_type == "sequential":
            stats_results = perform_complete_statistical_analysis(
                X, mouse_ids, seq_space, None, feature_names
            )
        elif analysis_type == "parallel":
            stats_results = perform_complete_statistical_analysis(
                X, mouse_ids, 
                pca_results['transformed_space'],
                lda_results['transformed_space'],
                feature_names
            )
        else:  # both
            stats_results = perform_complete_statistical_analysis(
                X, mouse_ids,
                pca_results['transformed_space'],
                lda_results['transformed_space'],
                feature_names
            )
        
        results['statistics'] = stats_results
        update_progress("Statistical analysis complete")
        
        # 4. Generate reports
        update_progress("Generating reports...")
        base_filename = Path(data_path).stem
        report_paths = generate_report(
            results,
            self.output_dir,
            base_filename,
            analysis_type
        )
        
        update_progress(f"Analysis complete! Results saved in: {self.output_dir}")
        update_progress(f"Generated report files: {report_paths[0]}, {report_paths[1]}")
        
        return results 