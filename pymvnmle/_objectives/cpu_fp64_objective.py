"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.

CRITICAL: This is an EXACT port of the working numpy_objective.py that passed
all regulatory tests, just adapted to the new class structure.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pymvnmle._objectives.base import MLEObjectiveBase, PatternData
from pymvnmle._objectives.parameterizations import InverseCholeskyParameterization


class CPUObjectiveFP64(MLEObjectiveBase):
    """
    R-compatible MLE objective using inverse Cholesky parameterization.
    
    This is the reference implementation that exactly matches R's mvnmle.
    EXACT copy of the working numpy_objective.py with new class names.
    """
    
    def __init__(self, data: np.ndarray, validate: bool = True):
        """Initialize CPU objective with R-compatible settings."""
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
        Compute negative log-likelihood using R's exact algorithm.
        
        EXACT copy of __call__ from working numpy_objective.py
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
                log_det_delta_k = np.sum(np.log(np.diag(Delta_k)))
                
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
    
    def compute_gradient(self, theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Compute gradient using finite differences (R-compatible).
        
        This matches R's nlm() behavior exactly.
        """
        n_params = len(theta)
        grad = np.zeros(n_params)
        
        # Base objective value
        f_base = self.compute_objective(theta)
        
        # Check if objective is valid
        if not np.isfinite(f_base) or f_base > 1e9:
            return grad
        
        # Compute gradient using forward differences
        for i in range(n_params):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            
            try:
                f_plus = self.compute_objective(theta_plus)
                if np.isfinite(f_plus) and f_plus < 1e9:
                    grad[i] = (f_plus - f_base) / eps
                else:
                    # Try backward difference
                    theta_minus = theta.copy()
                    theta_minus[i] -= eps
                    f_minus = self.compute_objective(theta_minus)
                    if np.isfinite(f_minus) and f_minus < 1e9:
                        grad[i] = (f_base - f_minus) / eps
                    else:
                        grad[i] = 0.0
            except:
                grad[i] = 0.0
        
        return grad
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract estimates from parameter vector.
        
        EXACT copy from working numpy_objective.py
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
    
    def compute_hessian(self, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute Hessian using finite differences."""
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
    
    def validate_parameters(self, theta: np.ndarray) -> Tuple[bool, str]:
        """Validate parameter vector."""
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
        """Check convergence based on gradient norm."""
        if grad is None:
            grad = self.compute_gradient(theta)
        
        max_grad = np.max(np.abs(grad))
        return max_grad < tol
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of missingness patterns."""
        return self.summarize_patterns()