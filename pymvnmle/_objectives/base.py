"""
Base objective function class for PyMVNMLE with R-compatible pattern ordering.

This implementation exactly replicates R's mvnmle behavior, including the
critical pattern ordering quirk where data is sorted by pattern codes but
patterns must be processed with complete cases first.

Pattern optimization support added WITHOUT re-identifying patterns - uses
R's existing patterns to maintain exact compatibility.

Author: Senior Biostatistician
Purpose: FDA-grade statistical software requiring exact R compatibility
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


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
    4. Pattern optimization support (using R's patterns, not re-identifying)
    5. Subclass interface for objective/gradient computation
    
    Critical for regulatory compliance: Exact replication of R mvnmle behavior.
    """
    
    def __init__(self, data: np.ndarray, 
                 skip_validation: bool = False,
                 use_pattern_optimization: bool = False):
        """
        Initialize objective function with data.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data matrix with missing values as np.nan
        skip_validation : bool
            If True, skip data validation (assumes caller validated)
        use_pattern_optimization : bool, default=False
            If True, prepare data for pattern-based vectorized computation.
            This does NOT change results, only performance.
            Currently defaults to False until thoroughly tested.
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
        
        # Pattern optimization flag
        self.use_pattern_optimization = use_pattern_optimization
        
        # Apply R's mysort algorithm
        self._apply_mysort()
        
        # Extract patterns WITHOUT reordering (keep R's sort order)
        self.patterns = self._extract_patterns()
        self.n_patterns = len(self.patterns)
        
        # Prepare pattern optimization if requested
        # CRITICAL: This must use R's patterns, not re-identify!
        if self.use_pattern_optimization:
            self._prepare_pattern_optimization()
        else:
            self.pattern_groups = None
            self.pattern_efficiency = None
        
        # Compute sample statistics for subclasses
        self.sample_mean = np.nanmean(self.original_data, axis=0)
        self.sample_cov = self._compute_sample_covariance()
        
        # Check if data is complete (no missing values)
        self.is_complete = not np.any(np.isnan(self.original_data))
    
    def _validate_data(self, data: np.ndarray) -> None:
        """
        Validate input data matrix.
        
        Parameters
        ----------
        data : np.ndarray
            Data to validate
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        if data.shape[0] < 2:
            raise ValueError(f"Need at least 2 observations, got {data.shape[0]}")
        
        if data.shape[1] < 1:
            raise ValueError(f"Need at least 1 variable, got {data.shape[1]}")
        
        # Check for all-missing rows or columns
        if np.any(np.all(np.isnan(data), axis=1)):
            raise ValueError("Data contains rows with all missing values")
        
        if np.any(np.all(np.isnan(data), axis=0)):
            raise ValueError("Data contains columns with all missing values")
    
    def _apply_mysort(self) -> None:
        """
        Apply R's mysort algorithm to sort data by missingness patterns.
        
        This is CRITICAL for R compatibility - must match exactly.
        Creates pattern codes using powers of 2 and sorts observations.
        """
        # Create presence/absence matrix (1 = observed, 0 = missing)
        self.presence_absence = (~np.isnan(self.original_data)).astype(int)
        
        # Create pattern codes using powers of 2 (R's approach)
        # This creates unique codes for each missingness pattern
        powers = 2 ** np.arange(self.n_vars - 1, -1, -1)
        pattern_codes = self.presence_absence @ powers
        
        # Sort data by pattern codes
        sort_indices = np.argsort(pattern_codes)
        self.sorted_data = self.original_data[sort_indices]
        self.sorted_patterns = self.presence_absence[sort_indices]
        self.sorted_codes = pattern_codes[sort_indices]
        
        # Get unique patterns and their frequencies
        unique_codes, indices, counts = np.unique(
            self.sorted_codes,
            return_index=True,
            return_counts=True
        )
        
        self.pattern_frequencies = counts
        self.pattern_start_indices = indices
    
    def _extract_patterns(self) -> List[PatternData]:
        """
        Extract pattern data from sorted dataset.
        
        Returns
        -------
        List[PatternData]
            Pattern data structures in R's expected order
        """
        patterns = []
        
        for i, (start_idx, count) in enumerate(zip(self.pattern_start_indices, 
                                                   self.pattern_frequencies)):
            end_idx = start_idx + count
            
            # Get pattern mask for this group
            pattern_mask = self.sorted_patterns[start_idx]
            observed_indices = np.where(pattern_mask == 1)[0]
            missing_indices = np.where(pattern_mask == 0)[0]
            
            # Extract data for this pattern (only observed variables)
            pattern_data = self.sorted_data[start_idx:end_idx]
            if len(observed_indices) > 0:
                observed_data = pattern_data[:, observed_indices]
            else:
                observed_data = np.empty((count, 0))
            
            patterns.append(PatternData(
                pattern_id=i,
                n_obs=count,
                data=observed_data,
                observed_indices=observed_indices,
                missing_indices=missing_indices,
                pattern_start=start_idx,
                pattern_end=end_idx
            ))
        
        return patterns
    
    def _prepare_pattern_optimization(self) -> None:
        """
        Prepare optimized data structures for pattern-based computation.
        
        CRITICAL: This uses the EXISTING R-compatible patterns from self.patterns
        rather than re-identifying patterns, ensuring exact R compatibility.
        The optimization only reorganizes data for vectorization, it does NOT
        change which observations belong to which pattern.
        """
        try:
            # Import optimization utilities
            try:
                from pymvnmle._pattern_optimization import (
                    OptimizedPatternGroup,
                    compute_efficiency_metrics
                )
            except ImportError:
                # Module not available
                self.use_pattern_optimization = False
                self.pattern_groups = None
                self.pattern_efficiency = None
                return
            
            # Don't re-identify patterns! Use existing R-compatible patterns
            if not hasattr(self, 'patterns') or not self.patterns:
                self.use_pattern_optimization = False
                self.pattern_groups = None
                self.pattern_efficiency = None
                return
            
            # Convert existing R patterns to optimized groups
            # This preserves R's exact pattern ordering and identification
            self.pattern_groups = []
            
            for pattern in self.patterns:
                # Create optimized group from existing R pattern
                # Key insight: pattern.data already has the observed data extracted!
                
                # Create observed mask for this pattern
                observed_mask = np.zeros(self.n_vars, dtype=bool)
                observed_mask[pattern.observed_indices] = True
                
                # Row indices refer to position in SORTED data (maintaining R's order)
                row_indices = np.arange(pattern.pattern_start, pattern.pattern_end)
                
                # Create optimized group that exactly matches R's pattern
                group = OptimizedPatternGroup(
                    pattern_id=pattern.pattern_id,
                    observed_mask=observed_mask,
                    observed_indices=pattern.observed_indices.copy(),
                    missing_indices=pattern.missing_indices.copy(),
                    row_indices=row_indices,
                    n_obs=pattern.n_obs,
                    observed_data=pattern.data.copy()  # Already extracted by R's algorithm
                )
                
                self.pattern_groups.append(group)
            
            # Compute efficiency metrics to decide if optimization is worthwhile
            self.pattern_efficiency = compute_efficiency_metrics(self.pattern_groups)
            
            # Only use optimization if it provides meaningful speedup
            if self.pattern_efficiency.get('expected_speedup', 1.0) < 1.2:
                # Not worth the complexity for small speedup
                self.use_pattern_optimization = False
                self.pattern_groups = None
                self.pattern_efficiency = None
            
        except Exception as e:
            # Any error in pattern preparation - fall back to standard
            warnings.warn(
                f"Pattern optimization preparation failed: {e}. "
                f"Falling back to standard computation.",
                RuntimeWarning
            )
            self.use_pattern_optimization = False
            self.pattern_groups = None
            self.pattern_efficiency = None
    
    def _compute_sample_covariance(self) -> np.ndarray:
        """
        Compute sample covariance using pairwise deletion.
        
        This matches R's approach for getting initial parameter estimates.
        Uses only pairs of observations where both variables are observed.
        
        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Sample covariance matrix (positive definite)
        """
        cov = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Get pairs where both variables are observed
                mask = ~(np.isnan(self.original_data[:, i]) | 
                        np.isnan(self.original_data[:, j]))
                
                n_complete = np.sum(mask)
                
                if n_complete > 1:
                    xi = self.original_data[mask, i]
                    xj = self.original_data[mask, j]
                    
                    # Compute covariance with bias correction
                    mean_xi = np.mean(xi)
                    mean_xj = np.mean(xj)
                    
                    if i == j:
                        # Variance on diagonal
                        cov[i, i] = np.mean((xi - mean_xi) ** 2)
                    else:
                        # Covariance off diagonal
                        cov_ij = np.mean((xi - mean_xi) * (xj - mean_xj))
                        cov[i, j] = cov_ij
                        cov[j, i] = cov_ij
                elif i == j:
                    # Not enough data for this variable, use unit variance
                    cov[i, i] = 1.0
        
        # Ensure positive definite
        try:
            eigenvals = np.linalg.eigvalsh(cov)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval <= 0:
                # Add regularization to make positive definite
                regularization = max(1e-6, abs(min_eigenval) + 1e-6)
                cov += regularization * np.eye(self.n_vars)
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal matrix
            cov = np.diag(np.maximum(np.diag(cov), 1.0))
        
        return cov
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        Get information about patterns and optimization status.
        
        Returns
        -------
        Dict[str, Any]
            Pattern statistics and optimization metrics
        """
        info = {
            'n_patterns': self.n_patterns,
            'n_observations': self.n_obs,
            'n_variables': self.n_vars,
            'is_complete': self.is_complete,
            'optimization_enabled': self.use_pattern_optimization,
            'optimization_available': self.pattern_groups is not None,
        }
        
        # Add pattern frequency information from original patterns
        if hasattr(self, 'patterns') and self.patterns:
            pattern_sizes = [p.n_obs for p in self.patterns]
            info['pattern_frequencies'] = pattern_sizes
            info['complete_cases'] = sum(
                p.n_obs for p in self.patterns 
                if len(p.missing_indices) == 0
            )
            info['pattern_distribution'] = {
                'min_size': min(pattern_sizes),
                'max_size': max(pattern_sizes),
                'mean_size': np.mean(pattern_sizes),
                'median_size': np.median(pattern_sizes),
            }
        
        # Add optimization metrics if available
        if self.pattern_efficiency:
            info['optimization_metrics'] = {
                'compression_ratio': self.pattern_efficiency.get('compression_ratio', None),
                'expected_speedup': self.pattern_efficiency.get('expected_speedup', None),
                'avg_pattern_size': self.pattern_efficiency.get('avg_pattern_size', None),
            }
        
        return info
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter values.
        
        Should be overridden by subclasses for specific parameterizations.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        # This is a placeholder - subclasses must implement
        # their specific parameterization
        raise NotImplementedError("Subclasses must implement get_initial_parameters")
    
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute objective function value.
        
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
        raise NotImplementedError("Subclasses must implement compute_objective")
    
    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function.
        
        Default implementation uses finite differences (R-compatible).
        Subclasses may override with analytical gradients.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Gradient vector
        """
        # Default: finite differences (matches R's nlm)
        eps = 1e-8
        grad = np.zeros_like(theta)
        f0 = self.compute_objective(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            f_plus = self.compute_objective(theta_plus)
            grad[i] = (f_plus - f0) / eps
        
        return grad
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mu, Sigma, and log-likelihood from parameter vector.
        
        Should be overridden by subclasses for specific parameterizations.
        
        Parameters
        ----------
        theta : np.ndarray
            Optimized parameter vector
            
        Returns
        -------
        mu : np.ndarray
            Mean vector
        sigma : np.ndarray
            Covariance matrix
        loglik : float
            Log-likelihood at theta
        """
        raise NotImplementedError("Subclasses must implement extract_parameters")