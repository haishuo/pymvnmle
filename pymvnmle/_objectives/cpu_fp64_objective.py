"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.

CRITICAL FIX: Now includes row shuffling and Givens rotations that were
missing in the refactored version.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pymvnmle._objectives.base import MLEObjectiveBase, PatternData
from pymvnmle._objectives.parameterizations import InverseCholeskyParameterization


class CPUObjectiveFP64(MLEObjectiveBase):
    """
    R-compatible MLE objective using inverse Cholesky parameterization.
    
    This is the reference implementation that exactly matches R's mvnmle,
    including the critical row shuffling and Givens rotations algorithms.
    
    The objective returns -2 * log-likelihood, matching R's convention.
    """
    
    def __init__(self, data: np.ndarray, validate: bool = True):
        """
        Initialize CPU objective with R-compatible settings.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data with missing values as np.nan
        validate : bool
            Whether to validate input data
        """
        # Initialize base class (handles preprocessing)
        super().__init__(data, validate)
        
        # Create parameterization
        self.parameterization = InverseCholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        
        # R compatibility settings
        self.use_inverse_cholesky = True
        self.objective_scale = -2.0  # R returns -2 * log-likelihood
        
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
        """
        Get R-compatible initial parameters with improved robustness.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector using inverse Cholesky parameterization
        """
        # For very difficult datasets, may need more conservative initialization
        sample_cov_regularized = self.sample_cov.copy()
        
        # Check condition number
        try:
            eigenvals = np.linalg.eigvalsh(sample_cov_regularized)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)
            
            # If poorly conditioned or non-PD, regularize more aggressively
            if min_eig < 1e-6 or max_eig / min_eig > 1e10:
                # Add stronger regularization
                reg_amount = max(1e-4, abs(min_eig) + 1e-4)
                sample_cov_regularized += reg_amount * np.eye(self.n_vars)
                
                # Also shrink off-diagonals slightly for stability
                for i in range(self.n_vars):
                    for j in range(i + 1, self.n_vars):
                        sample_cov_regularized[i, j] *= 0.95
                        sample_cov_regularized[j, i] *= 0.95
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal covariance
            sample_cov_regularized = np.diag(np.diag(self.sample_cov))
            sample_cov_regularized += 0.1 * np.eye(self.n_vars)
        
        return self.parameterization.get_initial_parameters(
            self.sample_mean, 
            sample_cov_regularized
        )
    
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute -2 * log-likelihood (R convention) with exact R algorithms.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
            
        Returns
        -------
        float
            -2 * log-likelihood value
            
        Notes
        -----
        This implementation now includes the CRITICAL row shuffling and
        Givens rotations that are essential for R compatibility.
        """
        try:
            # Extract mu and reconstruct Delta matrix
            mu = theta[:self.n_vars].copy()
            delta = self._reconstruct_delta(theta)
            
            # Check for numerical issues in Delta
            diag_vals = np.diag(delta)
            if np.any(diag_vals <= 0) or np.any(~np.isfinite(delta)):
                return 1e10  # Return large value for invalid parameters
            
            # Initialize objective value
            obj_value = 0.0
            
            # Process each missingness pattern
            for pattern in self.patterns:
                if len(pattern.observed_indices) == 0:
                    continue  # Skip patterns with no observed variables
                
                # Apply R's row shuffling algorithm
                subdel = self._row_shuffle(delta, pattern.observed_indices)
                
                # Apply Givens rotations for numerical stability
                subdel = self._apply_givens_rotations(subdel)
                
                # Extract the relevant submatrix for observed variables
                n_obs_vars = len(pattern.observed_indices)
                delta_k = subdel[:n_obs_vars, :n_obs_vars]
                
                # Check numerical stability of delta_k
                diag_k = np.diag(delta_k)
                if np.any(diag_k <= 1e-10) or np.any(~np.isfinite(delta_k)):
                    return 1e10  # Return large value for singular matrix
                
                # Compute log-determinant contribution
                # In R: diagsum = sum(log(diag(subdel[1:pcount, 1:pcount])))
                # obj_value -= 2 * n_k * diagsum
                log_det_delta_k = np.sum(np.log(diag_k))
                if not np.isfinite(log_det_delta_k):
                    return 1e10
                
                obj_value -= 2.0 * pattern.n_obs * log_det_delta_k
                
                # Extract observed means
                mu_k = mu[pattern.observed_indices]
                
                # Compute quadratic form contribution
                # In R: for each observation, compute prod = subdel.T @ (y - mu)
                # then sum(prod^2)
                for i in range(pattern.n_obs):
                    obs_data = pattern.data[i, :]  # Already has only observed columns
                    centered = obs_data - mu_k
                    prod = delta_k.T @ centered
                    quad_term = np.dot(prod, prod)
                    if not np.isfinite(quad_term):
                        return 1e10
                    obj_value += quad_term
            
            # Final check
            if not np.isfinite(obj_value):
                return 1e10
                
            return obj_value
            
        except (np.linalg.LinAlgError, ValueError):
            # Return large value for any numerical errors
            return 1e10
    
    def _reconstruct_delta(self, theta: np.ndarray) -> np.ndarray:
        """
        Reconstruct Delta matrix from parameter vector.
        
        EXACTLY matches R's parameter ordering and structure.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Upper triangular Delta matrix
        """
        delta = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements (from log parameters)
        log_diag = theta[self.n_vars:2*self.n_vars]
        
        # Clip to prevent overflow/underflow
        log_diag = np.clip(log_diag, -10, 10)
        
        # Set diagonal with bounds checking
        diag_vals = np.exp(log_diag)
        diag_vals = np.maximum(diag_vals, 1e-10)  # Prevent zero diagonal
        np.fill_diagonal(delta, diag_vals)
        
        # Off-diagonal elements (R's column-major ordering)
        idx = 2 * self.n_vars
        for j in range(1, self.n_vars):  # Column
            for i in range(j):           # Row within column
                if idx < len(theta):
                    # Clip extreme values
                    val = theta[idx]
                    val = np.clip(val, -100, 100)
                    delta[i, j] = val
                    idx += 1
        
        return delta
    
    def _row_shuffle(self, delta: np.ndarray, observed_indices: np.ndarray) -> np.ndarray:
        """
        Apply R's row shuffling algorithm.
        
        This is CRITICAL for the inverse Cholesky parameterization to work
        correctly with missing data patterns.
        
        From evallf.c:
        - Put rows corresponding to observed variables FIRST
        - Put rows corresponding to missing variables LAST
        
        Parameters
        ----------
        delta : np.ndarray
            Full Delta matrix
        observed_indices : np.ndarray
            Indices of observed variables for this pattern
            
        Returns
        -------
        np.ndarray
            Row-shuffled matrix
        """
        subdel = np.zeros_like(delta)
        n_vars = self.n_vars
        
        # Put observed variable rows first
        pcount = 0
        for i in range(n_vars):
            if i in observed_indices:
                subdel[pcount, :] = delta[i, :]
                pcount += 1
        
        # Put missing variable rows last
        acount = 0
        for i in range(n_vars):
            if i not in observed_indices:
                subdel[n_vars - acount - 1, :] = delta[i, :]
                acount += 1
        
        return subdel
    
    def _apply_givens_rotations(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply Givens rotations to zero out elements below main diagonal.
        
        This is the EXACT algorithm from R's evallf.c, critical for
        numerical stability.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix to apply rotations to
            
        Returns
        -------
        np.ndarray
            Matrix with Givens rotations applied
        """
        result = matrix.copy()
        n_vars = self.n_vars
        
        # Process from bottom to top (R's exact order)
        for i in range(n_vars - 1, -1, -1):  # Bottom row moving up
            for j in range(i):                # Left to diagonal
                # Zero out result[i, j]
                a = result[i, j]
                
                # R's threshold: 0.000001
                if abs(a) < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                # Get next element
                b = result[i, j + 1] if j + 1 < n_vars else 0.0
                
                # Skip if both elements are too small
                if abs(a) < 0.000001 and abs(b) < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                # Compute rotation parameters
                r = np.sqrt(a * a + b * b)
                if r < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                c = a / r  # cos(theta)
                d = b / r  # sin(theta)
                
                # Apply rotation to entire matrix
                for k in range(n_vars):
                    old_kj = result[k, j]
                    old_kj1 = result[k, j + 1] if j + 1 < n_vars else 0.0
                    
                    result[k, j] = d * old_kj - c * old_kj1
                    if j + 1 < n_vars:
                        result[k, j + 1] = c * old_kj + d * old_kj1
                
                result[i, j] = 0.0
        
        # Flip column signs so diagonal elements are positive
        for i in range(n_vars):
            if i < n_vars and result[i, i] < 0:
                # Only flip up to and including row i
                result[:i + 1, i] *= -1
        
        return result
    
    def _compute_pattern_contribution(self, pattern: PatternData, 
                                     mu: np.ndarray, 
                                     sigma: np.ndarray) -> float:
        """
        Compute log-likelihood contribution from one missingness pattern.
        
        NOTE: This method is now DEPRECATED in favor of the direct
        computation in compute_objective() that uses row shuffling
        and Givens rotations. Kept for backward compatibility only.
        
        Parameters
        ----------
        pattern : PatternData
            Pattern data structure
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
            
        Returns
        -------
        float
            Pattern's contribution to -2 * log-likelihood
        """
        obs_idx = pattern.observed_indices
        n_obs_vars = len(obs_idx)
        
        # Extract observed submatrices
        mu_k = mu[obs_idx]
        sigma_k = sigma[np.ix_(obs_idx, obs_idx)]
        
        # Constant term
        const_term = n_obs_vars * self.log_2pi
        
        # Log determinant term
        try:
            # Use Cholesky for numerical stability
            L_k = np.linalg.cholesky(sigma_k)
            log_det_sigma_k = 2.0 * np.sum(np.log(np.diag(L_k)))
        except np.linalg.LinAlgError:
            # Fallback to eigenvalues if Cholesky fails
            eigenvals = np.linalg.eigvalsh(sigma_k)
            if np.min(eigenvals) <= 0:
                return 1e10  # Return large value for non-PD matrix
            log_det_sigma_k = np.sum(np.log(eigenvals))
        
        # Compute sample covariance for this pattern
        data_centered = pattern.data - mu_k
        S_k = (data_centered.T @ data_centered) / pattern.n_obs
        
        # Trace term: tr(Σ_k^-1 * S_k)
        try:
            # Solve Σ_k * X = S_k for X, then tr(X) = tr(Σ_k^-1 * S_k)
            sigma_k_inv_S_k = np.linalg.solve(sigma_k, S_k)
            trace_term = np.trace(sigma_k_inv_S_k)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            sigma_k_inv = np.linalg.pinv(sigma_k)
            trace_term = np.trace(sigma_k_inv @ S_k)
        
        # Combine terms (already scaled by -2)
        contribution = const_term + log_det_sigma_k + trace_term
        
        return contribution
    
    def compute_gradient(self, theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Compute gradient using finite differences (R-compatible).
        
        R uses finite differences, not analytical gradients.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
        eps : float
            Finite difference step size
            
        Returns
        -------
        np.ndarray
            Gradient vector
        """
        grad = np.zeros_like(theta)
        f0 = self.compute_objective(theta)
        
        # Check if objective is valid
        if not np.isfinite(f0) or f0 > 1e9:
            # Return zero gradient to avoid propagating NaN
            return grad
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            
            # Use adaptive step size for different parameter types
            if i < self.n_vars:
                # Mean parameters - use standard eps
                step = eps
            elif i < 2 * self.n_vars:
                # Log-diagonal parameters - use smaller step
                step = eps * 0.1
            else:
                # Off-diagonal parameters - use standard eps
                step = eps
            
            theta_plus[i] += step
            f_plus = self.compute_objective(theta_plus)
            
            # Check for numerical issues
            if np.isfinite(f_plus) and f_plus < 1e9:
                grad[i] = (f_plus - f0) / step
            else:
                # Try negative direction
                theta_minus = theta.copy()
                theta_minus[i] -= step
                f_minus = self.compute_objective(theta_minus)
                
                if np.isfinite(f_minus) and f_minus < 1e9:
                    grad[i] = (f0 - f_minus) / step
                else:
                    # Both directions failed, use zero
                    grad[i] = 0.0
        
        # Clip extreme gradients to prevent optimizer instability
        max_grad = 1000.0
        grad = np.clip(grad, -max_grad, max_grad)
        
        return grad
    
    def compute_hessian(self, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian using finite differences.
        
        Used for Newton-CG optimization if requested.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
        eps : float
            Finite difference step size
            
        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix
        """
        n = len(theta)
        hessian = np.zeros((n, n))
        
        # Get base gradient
        grad0 = self.compute_gradient(theta, eps)
        
        # Compute each column of Hessian
        for j in range(n):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            grad_plus = self.compute_gradient(theta_plus, eps)
            
            # Finite difference approximation
            hessian[:, j] = (grad_plus - grad0) / eps
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        return hessian
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mu, sigma, and log-likelihood from parameter vector.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        mu : np.ndarray
            Mean estimate
        sigma : np.ndarray
            Covariance estimate  
        loglik : float
            Log-likelihood value
        """
        # Unpack parameters
        mu, sigma = self.parameterization.unpack(theta)
        
        # Compute log-likelihood
        neg2_loglik = self.compute_objective(theta)
        loglik = -0.5 * neg2_loglik
        
        return mu, sigma, loglik
    
    def validate_parameters(self, theta: np.ndarray) -> Tuple[bool, str]:
        """
        Validate parameter vector.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        valid : bool
            True if parameters are valid
        message : str
            Error message if invalid
        """
        # Check length
        if len(theta) != self.n_params:
            return False, f"Expected {self.n_params} parameters, got {len(theta)}"
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(theta)):
            return False, "Parameters contain NaN or Inf"
        
        # Check bounds on log-diagonal elements
        log_diag = theta[self.n_vars:2*self.n_vars]
        if np.any(log_diag < -10) or np.any(log_diag > 10):
            return False, "Log-diagonal elements out of bounds [-10, 10]"
        
        # Check bounds on off-diagonal elements
        off_diag = theta[2*self.n_vars:]
        if np.any(np.abs(off_diag) > 100):
            return False, "Off-diagonal elements exceed bound of 100"
        
        return True, "Parameters valid"
    
    def check_convergence(self, theta: np.ndarray,
                         grad: Optional[np.ndarray] = None,
                         tol: float = 1e-6) -> bool:
        """
        Check convergence based on gradient norm.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        grad : np.ndarray or None
            Current gradient (computed if None)
        tol : float
            Convergence tolerance
            
        Returns
        -------
        bool
            True if converged
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        max_grad = np.max(np.abs(grad))
        return max_grad < tol
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of missingness patterns."""
        return self.summarize_patterns()