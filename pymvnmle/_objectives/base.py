"""
Base class for MLE objectives with shared preprocessing.

This module contains the data preprocessing logic that is common to all
objective implementations, regardless of backend or parameterization.
Implements R's mysort algorithm for pattern grouping.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PatternData:
    """
    Data structure for a missingness pattern group.
    
    This matches R's pattern grouping structure exactly.
    
    Attributes
    ----------
    pattern_id : int
        Unique identifier for this pattern
    observed_indices : np.ndarray
        Indices of observed variables for this pattern
    missing_indices : np.ndarray
        Indices of missing variables for this pattern
    n_obs : int
        Number of observations with this pattern
    data : np.ndarray
        Data subset for this pattern (n_obs × n_observed)
    pattern_start : int
        Start index in sorted data
    pattern_end : int
        End index in sorted data (exclusive)
    """
    pattern_id: int
    observed_indices: np.ndarray
    missing_indices: np.ndarray
    n_obs: int
    data: np.ndarray
    pattern_start: int
    pattern_end: int


class MLEObjectiveBase(ABC):
    """
    Base class for MLE objectives with shared preprocessing.
    
    This class handles:
    1. Data validation and preprocessing
    2. R's mysort algorithm for pattern grouping
    3. Initial parameter computation
    4. Pattern data extraction
    
    All operations here use NumPy since preprocessing is always
    done on CPU regardless of the computational backend.
    """
    
    def __init__(self, data: np.ndarray, validate: bool = True):
        """
        Initialize with data preprocessing.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data matrix with missing values as np.nan
        validate : bool
            Whether to validate input data
        """
        if validate:
            self._validate_input(data)
        
        # Store dimensions
        self.n_obs, self.n_vars = data.shape
        self.original_data = np.asarray(data, dtype=np.float64)
        
        # Check for complete data (no missing values)
        self.is_complete = not np.any(np.isnan(self.original_data))
        
        if self.is_complete:
            # Complete data - no pattern grouping needed
            self._handle_complete_data()
        else:
            # Apply R's mysort preprocessing
            self._apply_mysort()
        
        # Extract pattern information
        self.patterns = self._extract_patterns()
        self.n_patterns = len(self.patterns)
        
        # Compute initial parameters
        self._compute_initial_statistics()
    
    def _validate_input(self, data: np.ndarray) -> None:
        """
        Validate input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to validate
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        if data.shape[0] < 2:
            raise ValueError(f"Need at least 2 observations, got {data.shape[0]}")
        
        if data.shape[1] < 1:
            raise ValueError(f"Need at least 1 variable, got {data.shape[1]}")
        
        # Check for all missing patterns
        if np.all(np.isnan(data)):
            raise ValueError("All data values are missing")
        
        # Check for variables with no observed values
        for j in range(data.shape[1]):
            if np.all(np.isnan(data[:, j])):
                raise ValueError(f"Variable {j} has no observed values")
    
    def _handle_complete_data(self) -> None:
        """Handle special case of complete data (no missing values)."""
        self.sorted_data = self.original_data.copy()
        self.pattern_assignments = np.zeros(self.n_obs, dtype=np.int32)
        self.pattern_counts = np.array([self.n_obs], dtype=np.int32)
        self.unique_patterns = np.ones((1, self.n_vars), dtype=bool)
    
    def _apply_mysort(self) -> None:
        """
        Apply R's mysort algorithm to group observations by missingness pattern.
        
        This sorts the data so that observations with the same missingness
        pattern are contiguous, enabling efficient likelihood computation.
        """
        # Create binary missingness indicators (1 = observed, 0 = missing)
        missingness_matrix = ~np.isnan(self.original_data)
        
        # Convert each pattern to a unique identifier
        # Using base-2 representation for uniqueness
        pattern_ids = np.zeros(self.n_obs, dtype=np.int64)
        for i in range(self.n_obs):
            pattern_ids[i] = self._pattern_to_id(missingness_matrix[i])
        
        # Sort observations by pattern ID
        sort_indices = np.argsort(pattern_ids)
        self.sorted_data = self.original_data[sort_indices]
        sorted_patterns = pattern_ids[sort_indices]
        
        # Find unique patterns and their counts
        unique_patterns, pattern_starts, pattern_counts = np.unique(
            sorted_patterns, 
            return_index=True, 
            return_counts=True
        )
        
        # Store pattern information
        self.pattern_assignments = np.zeros(self.n_obs, dtype=np.int32)
        for i, (start, count) in enumerate(zip(pattern_starts, pattern_counts)):
            self.pattern_assignments[start:start+count] = i
        
        self.pattern_counts = pattern_counts
        
        # Convert pattern IDs back to binary matrix
        self.unique_patterns = np.zeros((len(unique_patterns), self.n_vars), dtype=bool)
        for i, pattern_id in enumerate(unique_patterns):
            self.unique_patterns[i] = self._id_to_pattern(pattern_id)
    
    def _pattern_to_id(self, pattern: np.ndarray) -> int:
        """
        Convert missingness pattern to unique integer ID.
        
        Parameters
        ----------
        pattern : np.ndarray, shape (n_vars,)
            Binary pattern (True = observed, False = missing)
            
        Returns
        -------
        int
            Unique pattern identifier
        """
        # Use base-2 representation
        result = 0
        for i, val in enumerate(pattern):
            if val:
                result += 2**i
        return result
    
    def _id_to_pattern(self, pattern_id: int) -> np.ndarray:
        """
        Convert pattern ID back to binary pattern.
        
        Parameters
        ----------
        pattern_id : int
            Pattern identifier
            
        Returns
        -------
        np.ndarray, shape (n_vars,)
            Binary pattern (True = observed, False = missing)
        """
        pattern = np.zeros(self.n_vars, dtype=bool)
        for i in range(self.n_vars):
            if pattern_id & (2**i):
                pattern[i] = True
        return pattern
    
    def _extract_patterns(self) -> List[PatternData]:
        """
        Extract pattern data structures from sorted data.
        
        Returns
        -------
        List[PatternData]
            Pattern data for each unique missingness pattern
        """
        patterns = []
        
        if self.is_complete:
            # Single pattern for complete data
            pattern = PatternData(
                pattern_id=0,
                observed_indices=np.arange(self.n_vars),
                missing_indices=np.array([], dtype=np.int32),
                n_obs=self.n_obs,
                data=self.sorted_data,
                pattern_start=0,
                pattern_end=self.n_obs
            )
            patterns.append(pattern)
        else:
            # Extract each pattern
            pattern_start = 0
            for i, (pattern_mask, count) in enumerate(zip(self.unique_patterns, self.pattern_counts)):
                pattern_end = pattern_start + count
                
                # Get observed/missing indices
                observed_idx = np.where(pattern_mask)[0]
                missing_idx = np.where(~pattern_mask)[0]
                
                # Extract data for this pattern (only observed columns)
                pattern_data = self.sorted_data[pattern_start:pattern_end][:, observed_idx]
                
                pattern = PatternData(
                    pattern_id=i,
                    observed_indices=observed_idx,
                    missing_indices=missing_idx,
                    n_obs=count,
                    data=pattern_data,
                    pattern_start=pattern_start,
                    pattern_end=pattern_end
                )
                patterns.append(pattern)
                
                pattern_start = pattern_end
        
        return patterns
    
    def _compute_initial_statistics(self) -> None:
        """
        Compute initial sample statistics for starting values.
        
        Uses pairwise complete observations for covariance estimation.
        """
        # Compute sample mean (using all available data per variable)
        self.sample_mean = np.zeros(self.n_vars)
        for j in range(self.n_vars):
            col_data = self.original_data[:, j]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                self.sample_mean[j] = np.mean(valid_data)
        
        # Compute sample covariance (pairwise complete observations)
        self.sample_cov = self._compute_pairwise_covariance()
        
        # Ensure positive definiteness
        self.sample_cov = self._regularize_covariance(self.sample_cov)
    
    def _compute_pairwise_covariance(self) -> np.ndarray:
        """
        Compute covariance using pairwise complete observations.
        
        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Sample covariance matrix
        """
        cov = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Find observations where both variables are observed
                valid_mask = ~np.isnan(self.original_data[:, i]) & ~np.isnan(self.original_data[:, j])
                
                if np.sum(valid_mask) > 1:
                    xi = self.original_data[valid_mask, i]
                    xj = self.original_data[valid_mask, j]
                    
                    # Compute covariance
                    cov_ij = np.cov(xi, xj)[0, 1] if i != j else np.var(xi)
                    cov[i, j] = cov_ij
                    cov[j, i] = cov_ij
                else:
                    # Not enough data - use identity
                    cov[i, j] = 1.0 if i == j else 0.0
                    cov[j, i] = cov[i, j]
        
        return cov
    
    def _regularize_covariance(self, cov: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
        """
        Regularize covariance matrix to ensure positive definiteness.
        
        Parameters
        ----------
        cov : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
        min_eigenval : float
            Minimum eigenvalue to enforce
            
        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Regularized positive definite covariance
        """
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Regularize if needed
        if np.min(eigenvals) < min_eigenval:
            eigenvals = np.maximum(eigenvals, min_eigenval)
            cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Ensure symmetry
            cov = 0.5 * (cov + cov.T)
        
        return cov
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of missingness patterns.
        
        Returns
        -------
        dict
            Pattern summary statistics
        """
        summary = {
            'n_patterns': self.n_patterns,
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'is_complete': self.is_complete,
            'patterns': []
        }
        
        for pattern in self.patterns:
            pattern_info = {
                'pattern_id': pattern.pattern_id,
                'n_obs': pattern.n_obs,
                'n_observed': len(pattern.observed_indices),
                'n_missing': len(pattern.missing_indices),
                'observed_vars': pattern.observed_indices.tolist(),
                'missing_vars': pattern.missing_indices.tolist(),
                'proportion': pattern.n_obs / self.n_obs
            }
            summary['patterns'].append(pattern_info)
        
        # Sort by frequency
        summary['patterns'].sort(key=lambda x: x['n_obs'], reverse=True)
        
        return summary
    
    @abstractmethod
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute objective function value.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        float
            Objective function value (-2 * log-likelihood)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter values.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        raise NotImplementedError