"""
Base class for MLE objectives with shared preprocessing.
All backend-specific implementations inherit from this.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PatternData:
    """
    R-compatible pattern data structure.
    
    Attributes
    ----------
    observed_indices : np.ndarray
        Indices of observed variables for this pattern
    n_k : int
        Number of observations with this pattern
    data_k : np.ndarray
        Data subset for this pattern (n_k × |observed_indices|)
    pattern_start : int
        Start index in sorted data
    pattern_end : int
        End index in sorted data
    """
    observed_indices: np.ndarray
    n_k: int
    data_k: np.ndarray
    pattern_start: int = 0
    pattern_end: int = 0


class MLEObjectiveBase:
    """
    Base class containing shared preprocessing for all objective implementations.
    
    This class handles:
    1. Data preprocessing with R's mysort algorithm
    2. Pattern extraction and organization
    3. Initial parameter computation
    
    All operations here use NumPy since preprocessing is always done on CPU
    regardless of the computational backend used for optimization.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize with data preprocessing.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data matrix with missing values as np.nan
        """
        # Store data dimensions
        self.n_obs, self.n_vars = data.shape
        self.original_data = np.asarray(data, dtype=np.float64)
        
        # Apply R's mysort preprocessing
        self.sorted_data, self.pattern_frequencies, self.presence_absence = self._mysort_preprocessing()
        
        # Extract pattern information
        self.patterns = self._extract_pattern_data()
        self.n_patterns = len(self.patterns)
        
        # Parameter structure: θ = [μ, log(diag(Δ)), upper(Δ)]
        self.n_params = self.n_vars + (self.n_vars * (self.n_vars + 1)) // 2
    
    def _mysort_preprocessing(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sort data by missingness patterns (R's mysort algorithm).
        
        This groups observations with identical missingness patterns together
        for efficient likelihood computation.
        
        Returns
        -------
        sorted_data : np.ndarray
            Data matrix with rows reordered by pattern
        frequencies : np.ndarray
            Number of observations in each pattern
        presence_absence : np.ndarray
            Binary matrix of observed variables per pattern
        """
        # Create binary representation (1=observed, 0=missing)
        is_observed = (~np.isnan(self.original_data)).astype(int)
        
        # Convert patterns to decimal for sorting
        # R uses powers of 2 to create unique codes for each pattern
        powers = 2 ** np.arange(self.n_vars - 1, -1, -1)
        pattern_codes = is_observed @ powers
        
        # Sort by pattern code
        sort_indices = np.argsort(pattern_codes)
        sorted_data = self.original_data[sort_indices]
        sorted_patterns = is_observed[sort_indices]
        sorted_codes = pattern_codes[sort_indices]
        
        # Extract unique patterns and frequencies
        unique_codes, frequencies = np.unique(sorted_codes, return_counts=True)
        
        # Build presence/absence matrix for unique patterns
        presence_absence = []
        current_code = -1
        
        for i, code in enumerate(sorted_codes):
            if code != current_code:
                presence_absence.append(sorted_patterns[i])
                current_code = code
        
        presence_absence = np.array(presence_absence)
        
        return sorted_data, frequencies, presence_absence
    
    def _extract_pattern_data(self) -> List[PatternData]:
        """
        Extract pattern-specific data structures.
        
        Creates PatternData objects for each unique missingness pattern,
        maintaining R's data organization for efficient computation.
        
        Returns
        -------
        patterns : List[PatternData]
            List of pattern data structures
        """
        patterns = []
        data_start = 0
        
        for pattern_idx in range(len(self.pattern_frequencies)):
            n_k = self.pattern_frequencies[pattern_idx]
            data_end = data_start + n_k
            
            # Get observed variable indices for this pattern
            pattern_vec = self.presence_absence[pattern_idx]
            observed_indices = np.where(pattern_vec == 1)[0]
            
            # Skip patterns with no observed variables
            if len(observed_indices) == 0:
                data_start = data_end
                continue
            
            # Extract data subset (only observed variables)
            data_k = self.sorted_data[data_start:data_end][:, observed_indices]
            
            patterns.append(PatternData(
                observed_indices=observed_indices,
                n_k=n_k,
                data_k=data_k,
                pattern_start=data_start,
                pattern_end=data_end
            ))
            
            data_start = data_end
        
        return patterns
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Compute initial parameter estimates using R-compatible method.
        
        Uses pairwise complete observations to compute starting covariance,
        exactly matching R's getstartvals() function.
        
        Returns
        -------
        theta : np.ndarray
            Initial parameter vector in R's format:
            [μ₁, ..., μₚ, log(δ₁₁), ..., log(δₚₚ), δ₁₂, δ₁₃, δ₂₃, ...]
        """
        # Step 1: Compute sample means
        mu_start = np.nanmean(self.original_data, axis=0)
        mu_start[np.isnan(mu_start)] = 0.0  # Handle completely missing variables
        
        # Step 2: Compute pairwise complete sample covariance
        cov_sample = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Find pairwise complete observations
                mask = ~(np.isnan(self.original_data[:, i]) | 
                        np.isnan(self.original_data[:, j]))
                n_complete = np.sum(mask)
                
                if n_complete > 1:
                    if i == j:
                        # Variance
                        var_i = np.var(self.original_data[mask, i], ddof=1)
                        cov_sample[i, i] = var_i if var_i > 1e-10 else 0.01
                    else:
                        # Covariance
                        data_pair = self.original_data[mask][:, [i, j]]
                        cov_ij = np.cov(data_pair.T, ddof=1)[0, 1]
                        cov_sample[i, j] = cov_sample[j, i] = cov_ij
                else:
                    # No pairwise complete observations
                    cov_sample[i, j] = cov_sample[j, i] = 0.0
                    if i == j:
                        cov_sample[i, i] = 1.0  # Default variance
        
        # Step 3: Regularize to ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(cov_sample)
        
        # Find smallest positive eigenvalue
        pos_eigenvals = eigenvals[eigenvals > 0]
        if len(pos_eigenvals) > 0:
            min_pos_eigenval = np.min(pos_eigenvals)
        else:
            min_pos_eigenval = 0.01
        
        # Regularize with small positive value
        eps = 1e-3 * min_pos_eigenval
        regularized_eigenvals = np.maximum(eigenvals, eps)
        
        # Reconstruct regularized covariance
        cov_regularized = eigenvecs @ np.diag(regularized_eigenvals) @ eigenvecs.T
        cov_regularized = (cov_regularized + cov_regularized.T) / 2  # Ensure symmetry
        
        # Step 4: Compute starting Delta via Cholesky
        try:
            # Cholesky decomposition: cov = L @ L.T
            L = np.linalg.cholesky(cov_regularized)
            # Delta is inverse of upper triangular Cholesky
            Delta_start = np.linalg.solve(L.T, np.eye(self.n_vars))
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            Delta_start = np.diag(1.0 / np.sqrt(np.diag(cov_regularized)))
        
        # Ensure positive diagonal
        for i in range(self.n_vars):
            if Delta_start[i, i] < 0:
                Delta_start[i, :] *= -1
        
        # Step 5: Pack into parameter vector
        theta_init = np.zeros(self.n_params)
        
        # Mean parameters
        theta_init[:self.n_vars] = mu_start
        
        # Log diagonal parameters
        diag_vals = np.diag(Delta_start)
        theta_init[self.n_vars:2*self.n_vars] = np.log(np.maximum(diag_vals, 1e-10))
        
        # Upper triangle parameters (column-major order)
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):
                theta_init[idx] = Delta_start[i, j]
                idx += 1
        
        return theta_init