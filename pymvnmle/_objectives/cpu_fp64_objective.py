"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.

CRITICAL: This is an EXACT port of the working numpy_objective.py that passed
all regulatory tests, now with added pattern optimization for performance.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pymvnmle._objectives.base import MLEObjectiveBase, PatternData
from pymvnmle._objectives.parameterizations import InverseCholeskyParameterization


class CPUObjectiveFP64(MLEObjectiveBase):
    """
    R-compatible MLE objective using inverse Cholesky parameterization.
    
    This is the reference implementation that exactly matches R's mvnmle.
    Now includes optional pattern optimization for significant performance gains
    without changing the mathematical results.
    """
    
    def __init__(self, data: np.ndarray, 
                 validate: bool = True,
                 use_pattern_optimization: bool = True):
        """
        Initialize CPU objective with R-compatible settings.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with missing values as np.nan
        validate : bool
            Whether to validate input data
        use_pattern_optimization : bool
            Whether to use pattern-based vectorized computation.
            Improves performance without changing results.
        """
        super().__init__(data, 
                        skip_validation=not validate,
                        use_pattern_optimization=use_pattern_optimization)
        
        # Create parameterization
        self.parameterization = InverseCholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        
        # R compatibility settings
        self.use_inverse_cholesky = True
        self.objective_scale = -2.0  # R returns -2 * log-likelihood
        
        # Store optimization flag
        self.use_pattern_optimization_flag = use_pattern_optimization
        
        # Check if pattern optimization is available and worthwhile
        if self.use_pattern_optimization and hasattr(self, 'pattern_groups') and self.pattern_groups is not None:
            self.use_optimized_path = True
        else:
            self.use_optimized_path = False
        
        # Precompute constants for efficiency
        self._precompute_constants()
    
    def _precompute_constants(self) -> None:
        """Precompute constants used in likelihood calculation."""
        # Constant term in log-likelihood
        self.log_2pi = np.log(2 * np.pi)
        
        # Total number of observed values across all patterns
        self.total_observed = sum(
            pattern.n_obs * len(pattern.observed_indices) 
            for pattern in self.patterns
        )
    
    def get_initial_parameters(self) -> np.ndarray:
        """Get R-compatible initial parameters."""
        # For difficult datasets, may need conservative initialization
        sample_cov_regularized = self.sample_cov.copy()
        
        # Check condition number
        try:
            eigenvals = np.linalg.eigvalsh(sample_cov_regularized)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)
            
            # If poorly conditioned or non-PD, regularize
            if min_eig < 1e-6 or max_eig / min_eig > 1e10:
                reg_amount = max(1e-4, abs(min_eig) + 1e-4)
                sample_cov_regularized += reg_amount * np.eye(self.n_vars)
                
                # Shrink off-diagonals for stability
                for i in range(self.n_vars):
                    for j in range(i + 1, self.n_vars):
                        sample_cov_regularized[i, j] *= 0.95
                        sample_cov_regularized[j, i] *= 0.95
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal
            sample_cov_regularized = np.diag(np.diag(self.sample_cov))
            sample_cov_regularized += 0.1 * np.eye(self.n_vars)
        
        return self.parameterization.get_initial_parameters(
            self.sample_mean, 
            sample_cov_regularized
        )
    
    def _reconstruct_delta_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Reconstruct upper triangular Δ matrix from parameter vector.
        
        EXACT copy from working numpy_objective.py
        """
        Delta = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements: exp(log-parameters) to ensure positivity
        log_diag = theta[self.n_vars:2*self.n_vars]
        np.fill_diagonal(Delta, np.exp(log_diag))
        
        # Off-diagonal elements in R's column-major order
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):
                Delta[i, j] = theta[idx]
                idx += 1
        
        return Delta
    
    def _apply_givens_rotations(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply Givens rotations for numerical stability.
        
        EXACT copy from working numpy_objective.py - CRITICAL for R compatibility.
        Matches R's evallf.c implementation precisely.
        """
        result = matrix.copy()
        
        # R's exact algorithm: bottom-up, left-to-right
        for i in range(self.n_vars - 1, -1, -1):  # Bottom to top
            for j in range(i):  # Left to diagonal
                a = result[i, j]
                b = result[i, j+1] if j+1 < self.n_vars else 0.0
                
                # R's exact threshold
                if np.abs(a) < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                # Compute rotation parameters
                r = np.sqrt(a*a + b*b)
                if r < 0.000001:
                    continue
                
                c = a / r
                d = b / r
                
                # Apply rotation to entire matrix
                for k in range(self.n_vars):
                    old_kj = result[k, j]
                    old_kj1 = result[k, j+1] if j+1 < self.n_vars else 0.0
                    
                    result[k, j] = d * old_kj - c * old_kj1
                    if j+1 < self.n_vars:
                        result[k, j+1] = c * old_kj + d * old_kj1
                
                result[i, j] = 0.0
        
        # Ensure positive diagonal (R's sign adjustment)
        for i in range(self.n_vars):
            if result[i, i] < 0:
                for j in range(i+1):
                    result[j, i] *= -1
        
        return result
    
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute negative log-likelihood using standard or optimized path.
        
        Routes to the appropriate implementation based on whether pattern
        optimization is enabled and available.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [mu, log(diag(Delta)), off-diag(Delta)]
            
        Returns
        -------
        float
            -2 * log-likelihood (R convention)
        """
        if self.use_optimized_path:
            return self._compute_objective_optimized(theta)
        else:
            return self._compute_objective_standard(theta)
    
    def _compute_objective_standard(self, theta: np.ndarray) -> float:
        """
        Compute negative log-likelihood using R's exact algorithm.
        
        EXACT copy of compute_objective from working numpy_objective.py
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)
        
        # Apply Givens rotations for numerical stability (R's evallf.c)
        Delta_stabilized = self._apply_givens_rotations(Delta)
        
        # Compute negative log-likelihood using pattern-wise formula
        neg_loglik = 0.0
        
        for pattern in self.patterns:
            # Skip empty patterns - using n_obs instead of n_k
            if pattern.n_obs == 0 or len(pattern.observed_indices) == 0:
                continue
            
            # Extract relevant submatrices (R's approach)
            obs_indices = pattern.observed_indices
            n_obs_vars = len(obs_indices)
            mu_k = mu[obs_indices]
            
            # CRITICAL: Implement R's row shuffling algorithm exactly
            # Create reordered Delta with observed rows first, missing rows last
            subdel = np.zeros((self.n_vars, self.n_vars))
            
            # Put observed variable rows FIRST
            pcount = 0
            for i in range(self.n_vars):
                if i in obs_indices:
                    subdel[pcount, :] = Delta_stabilized[i, :]
                    pcount += 1
            
            # Put missing variable rows LAST
            acount = 0
            for i in range(self.n_vars):
                if i not in obs_indices:
                    subdel[self.n_vars - acount - 1, :] = Delta_stabilized[i, :]
                    acount += 1
            
            # Apply Givens rotations to shuffled matrix
            subdel_rotated = self._apply_givens_rotations(subdel)
            
            # Extract top-left submatrix for observed variables
            Delta_k = subdel_rotated[:n_obs_vars, :n_obs_vars]
            
            try:
                # Use R's exact computation approach
                # Log-determinant of Delta_k (more stable than computing Sigma_k first)
                diag_delta_k = np.diag(Delta_k)
                
                # Check for numerical issues
                if np.any(diag_delta_k <= 0):
                    # This shouldn't happen with proper Givens rotations
                    # Return large penalty value
                    return 1e20
                
                log_det_delta_k = np.sum(np.log(diag_delta_k))
                
                # Compute quadratic form efficiently
                # For each observation, compute (y - μ)'Σ^{-1}(y - μ)
                # Using Σ = (Δ^{-1})'Δ^{-1}, we get (Δ'(y - μ))'(Δ'(y - μ))
                obj_contribution = -2 * pattern.n_obs * log_det_delta_k
                
                for i in range(pattern.n_obs):
                    # pattern.data already has only observed columns
                    centered = pattern.data[i] - mu_k
                    # prod = Δ_k' @ centered
                    prod = Delta_k.T @ centered
                    obj_contribution += np.dot(prod, prod)
                
                neg_loglik += obj_contribution
                
            except (np.linalg.LinAlgError, RuntimeError):
                # Handle numerical issues
                return 1e20
        
        return neg_loglik
    
    def _compute_objective_optimized(self, theta: np.ndarray) -> float:
        """
        Pattern-optimized computation with vectorization.
        
        This method uses the EXACT SAME mathematical operations as the standard
        method, including row shuffling and Givens rotations, but groups
        computations by pattern for efficiency.
        
        Critical: Maintains exact R compatibility - only computation order changes.
        """
        # Extract mean parameters (same as standard)
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm (same as standard)
        Delta = self._reconstruct_delta_matrix(theta)
        
        # Apply Givens rotations for numerical stability (same as standard)
        Delta_stabilized = self._apply_givens_rotations(Delta)
        
        # Import pattern optimization utilities
        try:
            from pymvnmle._pattern_optimization import OptimizedPatternGroup
            
            # Use optimized pattern groups if available
            if not hasattr(self, 'pattern_groups') or self.pattern_groups is None:
                # Fall back to standard if patterns not available
                return self._compute_objective_standard(theta)
                
            pattern_groups = self.pattern_groups
        except ImportError:
            # Fall back to standard if module not available
            return self._compute_objective_standard(theta)
        
        # Compute negative log-likelihood using pattern groups
        neg_loglik = 0.0
        
        for group in pattern_groups:
            # Skip empty patterns
            if group.n_obs == 0 or group.n_observed == 0:
                continue
            
            # Extract relevant indices (same logic as standard)
            obs_indices = group.observed_indices
            n_obs_vars = len(obs_indices)
            mu_k = mu[obs_indices]
            
            # CRITICAL: Implement R's row shuffling algorithm exactly (SAME AS STANDARD)
            # Create reordered Delta with observed rows first, missing rows last
            subdel = np.zeros((self.n_vars, self.n_vars))
            
            # Put observed variable rows FIRST
            pcount = 0
            for i in range(self.n_vars):
                if i in obs_indices:
                    subdel[pcount, :] = Delta_stabilized[i, :]
                    pcount += 1
            
            # Put missing variable rows LAST
            acount = 0
            for i in range(self.n_vars):
                if i not in obs_indices:
                    subdel[self.n_vars - acount - 1, :] = Delta_stabilized[i, :]
                    acount += 1
            
            # Apply Givens rotations to shuffled matrix (SAME AS STANDARD)
            subdel_rotated = self._apply_givens_rotations(subdel)
            
            # Extract top-left submatrix for observed variables (SAME AS STANDARD)
            Delta_k = subdel_rotated[:n_obs_vars, :n_obs_vars]
            
            try:
                # Use R's exact computation approach
                diag_delta_k = np.diag(Delta_k)
                
                # Check for numerical issues (SAME AS STANDARD)
                if np.any(diag_delta_k <= 0):
                    return 1e20
                
                log_det_delta_k = np.sum(np.log(diag_delta_k))
                
                # VECTORIZED computation for this pattern group
                # This is the ONLY difference from standard method
                obj_contribution = -2 * group.n_obs * log_det_delta_k
                
                # Vectorized quadratic form computation
                # group.observed_data has shape (n_obs, n_observed_vars)
                centered = group.observed_data - mu_k[np.newaxis, :]  # Broadcasting
                
                # Compute Delta_k.T @ centered.T for all observations at once
                # Result shape: (n_observed_vars, n_obs)
                prod_all = Delta_k.T @ centered.T
                
                # Compute quadratic forms: sum of squared elements for each observation
                quadratic_forms = np.sum(prod_all * prod_all, axis=0)
                
                # Add all quadratic form contributions
                obj_contribution += np.sum(quadratic_forms)
                
                neg_loglik += obj_contribution
                
            except (np.linalg.LinAlgError, RuntimeError):
                # Handle numerical issues (SAME AS STANDARD)
                return 1e20
        
        return neg_loglik
    
    def compute_gradient(self, theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Compute gradient using finite differences (R-compatible).
        
        This matches R's nlm() behavior exactly.
        Pattern optimization speeds this up significantly since each
        objective evaluation is faster.
        """
        n_params = len(theta)
        grad = np.zeros(n_params)
        
        # Base objective value
        f_base = self.compute_objective(theta)
        
        # Compute gradient using forward differences
        for i in range(n_params):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            f_plus = self.compute_objective(theta_plus)
            grad[i] = (f_plus - f_base) / eps
        
        return grad
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mu, Sigma, and log-likelihood from parameter vector.
        
        EXACT copy from original implementation.
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)
        
        # Convert to covariance matrix: Σ = (Δ⁻¹)ᵀ Δ⁻¹
        try:
            # Use triangular solve for numerical stability (R's approach)
            I = np.eye(self.n_vars)
            Delta_inv = np.linalg.solve(Delta, I)  # More stable than inv(Delta)
            sigma = Delta_inv.T @ Delta_inv
            
            # Ensure exact symmetry (R does this)
            sigma = (sigma + sigma.T) / 2.0
            
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            sigma = np.eye(self.n_vars)
        
        # Compute log-likelihood
        loglik = -self.compute_objective(theta) / 2.0  # Objective is -2*log-likelihood
        
        return mu, sigma, loglik
    
    def validate_optimization(self, theta: np.ndarray, tol: float = 1e-12) -> Dict[str, Any]:
        """
        Validate that pattern optimization gives identical results to standard method.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector to test at
        tol : float
            Tolerance for comparison
            
        Returns
        -------
        Dict[str, Any]
            Validation results including objective values and difference
        """
        # Compute with standard method
        obj_standard = self._compute_objective_standard(theta)
        
        # Compute with optimized method if available
        if self.use_optimized_path:
            obj_optimized = self._compute_objective_optimized(theta)
            difference = abs(obj_standard - obj_optimized)
            relative_error = difference / (abs(obj_standard) + 1e-10)
            
            return {
                'standard_objective': obj_standard,
                'optimized_objective': obj_optimized,
                'absolute_difference': difference,
                'relative_error': relative_error,
                'passed': difference < tol,
                'tolerance_used': tol
            }
        else:
            return {
                'standard_objective': obj_standard,
                'optimized_objective': None,
                'message': 'Pattern optimization not available',
                'passed': True
            }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about pattern optimization effectiveness.
        
        Returns
        -------
        Dict[str, Any]
            Optimization statistics and expected performance gains
        """
        stats = {
            'optimization_enabled': self.use_optimized_path,
            'n_patterns': self.n_patterns,
            'n_observations': self.n_obs,
        }
        
        if hasattr(self, 'pattern_efficiency') and self.pattern_efficiency:
            stats.update({
                'compression_ratio': self.pattern_efficiency['compression_ratio'],
                'expected_speedup': self.pattern_efficiency['expected_speedup'],
                'avg_pattern_size': self.pattern_efficiency['avg_pattern_size'],
            })
        
        # Add pattern size distribution
        if hasattr(self, 'patterns'):
            pattern_sizes = [p.n_obs for p in self.patterns]
            stats['pattern_size_distribution'] = {
                'min': min(pattern_sizes),
                'max': max(pattern_sizes),
                'mean': np.mean(pattern_sizes),
                'median': np.median(pattern_sizes),
            }
        
        return stats
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about computational backend."""
        import platform
        
        info = {
            'backend': 'cpu',
            'device': platform.processor() or 'CPU',
            'precision': 'float64',
            'pattern_optimization': self.use_optimized_path,
        }
        
        if self.use_optimized_path and hasattr(self, 'pattern_efficiency'):
            info['pattern_speedup'] = self.pattern_efficiency.get('expected_speedup', 1.0)
        
        return info