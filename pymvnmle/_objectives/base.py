"""
Base objective function class for PyMVNMLE with R-compatible pattern ordering.

This implementation exactly replicates R's mvnmle behavior, including the
critical pattern ordering quirk where data is sorted by pattern codes but
patterns must be processed with complete cases first.

Author: Senior Biostatistician
Purpose: FDA-grade statistical software requiring exact R compatibility
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class PatternData:
    """Data structure for a single missingness pattern."""
    pattern_id: int                   # Unique identifier
    n_obs: int                        # Number of observations with this pattern
    data: np.ndarray                  # Data matrix (n_obs × n_observed_vars)
    observed_indices: np.ndarray      # Indices of observed variables
    missing_indices: np.ndarray       # Indices of missing variables  
    pattern_start: int                # Start index in sorted data
    pattern_end: int                  # End index in sorted data


class MLEObjectiveBase:
    """
    Base class for maximum likelihood estimation objective functions.
    
    This class handles:
    1. R-compatible data sorting (mysort algorithm)
    2. Pattern extraction with correct ordering (complete cases first)
    3. Initial parameter computation
    4. Subclass interface for objective/gradient computation
    
    Critical for regulatory compliance: Exact replication of R mvnmle behavior.
    """
    
    def __init__(self, data: np.ndarray, skip_validation: bool = False):
        """
        Initialize objective function with data.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data matrix with missing values as np.nan
        skip_validation : bool
            If True, skip data validation (assumes caller validated)
        """
        if not skip_validation:
            self._validate_data(data)
        
        # Store original data
        self.original_data = np.asarray(data, dtype=np.float64)
        self.n_obs, self.n_vars = self.original_data.shape
        
        # Parameter dimensions
        self.n_mean_params = self.n_vars
        self.n_cov_params = self.n_vars * (self.n_vars + 1) // 2
        self.n_params = self.n_mean_params + self.n_cov_params
        
        # Apply R's mysort algorithm
        self._apply_mysort()
        
        # Extract patterns WITHOUT reordering (keep R's sort order)
        self.patterns = self._extract_patterns()
        self.n_patterns = len(self.patterns)
        
        # Compute sample statistics for subclasses
        self.sample_mean = np.nanmean(self.original_data, axis=0)
        self.sample_cov = self._compute_sample_covariance()
    
    def _validate_data(self, data: np.ndarray) -> None:
        """
        Validate input data matrix.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to validate
            
        Raises
        ------
        ValueError
            If data is invalid (wrong shape, no variation, etc.)
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got shape {data.shape}")
        
        n_obs, n_vars = data.shape
        
        if n_obs < 2:
            raise ValueError(f"Need at least 2 observations, got {n_obs}")
        
        if n_vars < 1:
            raise ValueError(f"Need at least 1 variable, got {n_vars}")
        
        # Check for infinite values
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")
        
        # Check for complete missingness in any variable
        for j in range(n_vars):
            if np.all(np.isnan(data[:, j])):
                raise ValueError(f"Variable {j} is completely missing")
        
        # Check for complete missingness in any observation
        for i in range(n_obs):
            if np.all(np.isnan(data[i, :])):
                raise ValueError(f"Observation {i} is completely missing")
    
    def _apply_mysort(self) -> None:
        """
        Apply R's mysort algorithm to sort data by missingness patterns.
        
        This groups observations with identical missingness patterns together
        for efficient likelihood computation. Uses R's exact algorithm.
        """
        # Create binary missingness matrix (1 = observed, 0 = missing)
        self.is_observed = ~np.isnan(self.original_data)
        
        # R's exact power computation: 2^(p-1), 2^(p-2), ..., 2^0
        powers = 2 ** np.arange(self.n_vars - 1, -1, -1)
        
        # Convert each pattern to a decimal code
        self.pattern_codes = self.is_observed.astype(int) @ powers
        
        # Sort observations by pattern code
        self.sort_indices = np.argsort(self.pattern_codes)
        self.sorted_data = self.original_data[self.sort_indices]
        self.sorted_codes = self.pattern_codes[self.sort_indices]
        self.sorted_patterns = self.is_observed[self.sort_indices]
    
    def _extract_patterns(self) -> List[PatternData]:
        """
        Extract pattern data structures from sorted data.
        
        IMPORTANT: Patterns are kept in the order created by mysort.
        Do NOT reorder them - R's algorithm expects this exact order.
        
        Returns
        -------
        List[PatternData]
            Pattern data for each unique missingness pattern
        """
        # Find unique patterns in sorted data
        unique_codes, first_indices, counts = np.unique(
            self.sorted_codes, 
            return_index=True, 
            return_counts=True
        )
        
        patterns = []
        for i, (code, start_idx, count) in enumerate(zip(unique_codes, first_indices, counts)):
            end_idx = start_idx + count
            
            # Get binary pattern
            binary_pattern = self.sorted_patterns[start_idx]
            observed_indices = np.where(binary_pattern)[0]
            missing_indices = np.where(~binary_pattern)[0]
            
            # Extract data for observed columns only
            if len(observed_indices) > 0:
                pattern_data = self.sorted_data[start_idx:end_idx][:, observed_indices]
            else:
                pattern_data = np.empty((count, 0))
            
            patterns.append(PatternData(
                pattern_id=i,
                n_obs=count,
                data=pattern_data,
                observed_indices=observed_indices,
                missing_indices=missing_indices,
                pattern_start=start_idx,
                pattern_end=end_idx
            ))
        
        return patterns
    
    def _compute_sample_covariance(self) -> np.ndarray:
        """
        Compute sample covariance matrix using pairwise complete observations.
        
        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Positive definite covariance matrix
        """
        cov = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Find observations where both variables are observed
                mask = (~np.isnan(self.original_data[:, i]) & 
                       ~np.isnan(self.original_data[:, j]) &
                       ~np.isinf(self.original_data[:, i]) &
                       ~np.isinf(self.original_data[:, j]))
                
                n_complete = np.sum(mask)
                
                if n_complete > 1:
                    x_i = self.original_data[mask, i] - self.sample_mean[i]
                    x_j = self.original_data[mask, j] - self.sample_mean[j]
                    cov[i, j] = np.dot(x_i, x_j) / (n_complete - 1)
                    cov[j, i] = cov[i, j]
                elif i == j:
                    # Diagonal: use variance if available
                    var_i = np.nanvar(self.original_data[:, i], ddof=1)
                    cov[i, i] = var_i if var_i > 0 else 1.0
        
        # Ensure positive definiteness
        min_eigenval = np.min(np.linalg.eigvalsh(cov))
        if min_eigenval <= 0:
            ridge = abs(min_eigenval) + 1e-6
            cov += ridge * np.eye(self.n_vars)
        
        return cov
        """
        Compute initial parameter estimates.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
        """
        # Initialize parameter vector
        params = np.zeros(self.n_params)
        
        # Use pre-computed sample mean
        params[:self.n_vars] = self.sample_mean
        
        # Use pre-computed sample covariance
        initial_cov = self.sample_cov
        
        # Convert to inverse Cholesky parameterization
        try:
            # Cholesky decomposition: Σ = L L^T
            L = np.linalg.cholesky(initial_cov)
            
            # Inverse Cholesky factor: Δ = L^{-1}
            Delta = np.linalg.solve(L, np.eye(self.n_vars))
            
            # Extract log-diagonal elements
            params[self.n_vars:2*self.n_vars] = np.log(np.diag(Delta))
            
            # Extract off-diagonal elements (R's column-major order)
            idx = 2 * self.n_vars
            for j in range(1, self.n_vars):
                for i in range(j):
                    params[idx] = Delta[i, j]
                    idx += 1
                    
        except np.linalg.LinAlgError:
            # If Cholesky fails, use simple diagonal initialization
            # This is rare but can happen with extreme missingness
            params[self.n_vars:2*self.n_vars] = 0.0  # log(1) = 0
            # Off-diagonals remain zero
        
        return params
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter estimates for optimization.
        
        Public interface for subclasses and optimization routines.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        return self.initial_params.copy()
    
    def _compute_pairwise_covariance(self) -> np.ndarray:
        """
        Legacy method for compatibility. Just returns pre-computed covariance.
        
        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Sample covariance matrix
        """
        return self.sample_cov
    
    def reconstruct_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct μ and Δ from parameter vector.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        Delta : np.ndarray, shape (n_vars, n_vars)
            Inverse Cholesky factor (upper triangular)
        """
        # Extract mean
        mu = theta[:self.n_vars].copy()
        
        # Initialize Delta matrix
        Delta = np.zeros((self.n_vars, self.n_vars))
        
        # Set diagonal (with bounds to prevent overflow)
        log_diag = theta[self.n_vars:2*self.n_vars]
        log_diag = np.clip(log_diag, -10, 10)  # R's bounds
        np.fill_diagonal(Delta, np.exp(log_diag))
        
        # Fill upper triangle (R's column-major order)
        if self.n_vars > 1:
            idx = 2 * self.n_vars
            for j in range(1, self.n_vars):
                for i in range(j):
                    Delta[i, j] = np.clip(theta[idx], -100, 100)  # R's bounds
                    idx += 1
        
        return mu, Delta
    
    def compute_sigma_from_delta(self, Delta: np.ndarray) -> np.ndarray:
        """
        Compute Σ from Δ using the relationship Σ = (Δ^{-1})^T (Δ^{-1}).
        
        Parameters
        ----------
        Delta : np.ndarray, shape (n_vars, n_vars)
            Inverse Cholesky factor (upper triangular)
            
        Returns
        -------
        Sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix (positive definite)
        """
        # Compute X = Δ^{-1} via triangular solve
        X = np.linalg.solve(Delta, np.eye(self.n_vars))
        
        # Compute Σ = X^T X
        Sigma = X.T @ X
        
        # Ensure exact symmetry (numerical precision)
        Sigma = (Sigma + Sigma.T) / 2
        
        return Sigma
    
    def objective(self, theta: np.ndarray) -> float:
        """
        Compute negative log-likelihood objective.
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        float
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement objective()")
    
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective.
        
        May be implemented by subclasses for efficiency.
        Default uses finite differences.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Gradient vector
        """
        # Default: Use finite differences (like R)
        eps = 1e-8
        grad = np.zeros_like(theta)
        f0 = self.objective(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            f_plus = self.objective(theta_plus)
            grad[i] = (f_plus - f0) / eps
        
        return grad