"""
LMT Analysis Package
===================

This package provides tools for behavioral identity analysis using PCA and LDA approaches.
It implements both sequential and parallel analysis methods following Forkosh et al.

Main Components:
- Sequential Analysis (PCA followed by LDA)
- Parallel Analysis (PCA and LDA separately)
- Statistical Analysis Tools
- Identity Domain Analysis Core
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt

# Import core analysis modules
from .sequential_analysis import analyze_identity_domain_sequential
from .parallel_pca import analyze_pca_parallel
from .parallel_lda import analyze_lda_parallel
from .identity_domain import IdentityDomainAnalyzer
from .statistics import (
    perform_complete_statistical_analysis,
    calculate_stability_scores,
    perform_permutation_test,
    perform_variance_ttest,
    perform_regression_analysis,
    perform_anova,
    perform_correlation_analysis
)

# Type definitions for better code completion and type checking
ArrayLike = Union[np.ndarray, List[List[float]], List[float]]
FeatureMatrix = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]

class Analysis:
    """
    Main interface for behavioral identity analysis.
    Provides a unified API for all analysis methods.
    """
    
    @staticmethod
    def sequential(X: ArrayLike, y: ArrayLike, variance_threshold: float = 0.80) -> Dict:
        """
        Perform sequential analysis (PCA followed by LDA)
        
        Args:
            X: Feature matrix
            y: Labels (mouse IDs)
            variance_threshold: PCA variance threshold
            
        Returns:
            Dict containing analysis results
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        
        stability, transformed, var_explained = analyze_identity_domain_sequential(
            X_arr, y_arr, variance_threshold
        )
        
        return {
            'stability_score': stability,
            'transformed_space': transformed,
            'variance_explained': var_explained
        }
    
    @staticmethod
    def parallel(X: ArrayLike, y: ArrayLike) -> Dict:
        """
        Perform parallel analysis (PCA and LDA separately)
        
        Args:
            X: Feature matrix
            y: Labels (mouse IDs)
            
        Returns:
            Dict containing both PCA and LDA results
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        
        pca_results = analyze_pca_parallel(X_arr, y_arr)
        lda_results = analyze_lda_parallel(X_arr, y_arr)
        
        return {
            'pca': pca_results,
            'lda': lda_results
        }
    
    @staticmethod
    def statistics(X: ArrayLike, y: ArrayLike, transformed_space: ArrayLike, 
                  feature_names: List[str]) -> Dict:
        """
        Perform complete statistical analysis
        
        Args:
            X: Original feature matrix
            y: Labels (mouse IDs)
            transformed_space: Transformed data (PCA or LDA space)
            feature_names: List of feature names
            
        Returns:
            Dict containing all statistical results
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        transformed_arr = np.asarray(transformed_space, dtype=np.float64)
        
        return perform_complete_statistical_analysis(
            X_arr, y_arr, transformed_arr, feature_names
        )

# Define package exports
__all__ = [
    # Main interface
    'Analysis',
    
    # Core components
    'IdentityDomainAnalyzer',
    
    # Individual analysis functions
    'analyze_identity_domain_sequential',
    'analyze_pca_parallel',
    'analyze_lda_parallel',
    
    # Statistical functions
    'perform_complete_statistical_analysis',
    'calculate_stability_scores',
    'perform_permutation_test',
    'perform_variance_ttest',
    'perform_regression_analysis',
    'perform_anova',
    'perform_correlation_analysis'
]

# Version info
__version__ = '0.1.0'
__author__ = 'Andrea Stivala'
__email__ = 'andreastivala.as@gmail.com' 