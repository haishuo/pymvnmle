"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base import MLEObjectiveBase, PatternData
from .parameterizations import InverseCholeskyParameterization


class CPUObjectiveFP64(MLEObjectiveBase):
    """
    R-compatible MLE objective using inverse Cholesky parameterization.
    
    This is the reference implementation that exactly matches R's mvnmle.
    Uses FP64 precision throughout and inverse Cholesky parameterization.
    
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
        Get R-compatible initial parameters.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector using inverse Cholesky parameterization
        """
        return self.parameterization.get_initial_parameters(
            self.sample_mean, 
            self.sample_cov
        )
    
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute -2 * log-likelihood (R convention).
        
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
        This matches R's mvnmle exactly, computing:
        -2 * log L = Σ_k n_k * [p_k * log(2π) + log|Σ_k| + tr(Σ_k^-1 * S_k)]
        where k indexes missingness patterns.
        """
        # Unpack parameters
        mu, sigma = self.parameterization.unpack(theta)
        
        # Initialize objective value
        obj_value = 0.0
        
        # Process each missingness pattern
        for pattern in self.patterns:
            if len(pattern.observed_indices) == 0:
                continue  # Skip patterns with no observed variables
            
            # Compute contribution from this pattern
            pattern_contrib = self._compute_pattern_contribution(
                pattern, mu, sigma
            )
            
            # Weight by number of observations
            obj_value += pattern.n_obs * pattern_contrib
        
        # Scale by -2 for R compatibility
        return obj_value
    
    def _compute_pattern_contribution(self, pattern: PatternData, 
                                     mu: np.ndarray, 
                                     sigma: np.ndarray) -> float:
        """
        Compute log-likelihood contribution from one missingness pattern.
        
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
            
        Notes
        -----
        For pattern k with observed indices O_k:
        contribution = p_k * log(2π) + log|Σ_k| + tr(Σ_k^-1 * S_k)
        where:
        - p_k = number of observed variables
        - Σ_k = sigma[O_k, O_k] (observed submatrix)
        - S_k = sample covariance for pattern k
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
            
        Notes
        -----
        Uses central differences for better accuracy:
        ∂f/∂θ_i ≈ [f(θ + e_i*h) - f(θ - e_i*h)] / (2*h)
        """
        grad = np.zeros(len(theta))
        
        for i in range(len(theta)):
            # Create perturbation vectors
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            
            # Apply perturbation
            theta_plus[i] += eps
            theta_minus[i] -= eps
            
            # Compute finite difference
            f_plus = self.compute_objective(theta_plus)
            f_minus = self.compute_objective(theta_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def compute_hessian(self, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian using finite differences.
        
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
            
        Notes
        -----
        Uses central differences on the gradient:
        H_ij ≈ [∂f/∂θ_i(θ + e_j*h) - ∂f/∂θ_i(θ - e_j*h)] / (2*h)
        """
        n = len(theta)
        hess = np.zeros((n, n))
        
        # Base gradient
        grad_base = self.compute_gradient(theta, eps)
        
        for j in range(n):
            # Perturb parameter j
            theta_plus = theta.copy()
            theta_plus[j] += eps
            
            # Compute gradient at perturbed point
            grad_plus = self.compute_gradient(theta_plus, eps)
            
            # Finite difference for column j of Hessian
            hess[:, j] = (grad_plus - grad_base) / eps
        
        # Symmetrize (Hessian should be symmetric)
        hess = 0.5 * (hess + hess.T)
        
        return hess
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mean, covariance, and log-likelihood from parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean estimate
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance estimate
        loglik : float
            Log-likelihood value (not -2*log-lik)
        """
        # Unpack parameters
        mu, sigma = self.parameterization.unpack(theta)
        
        # Compute objective (-2 * log-likelihood)
        neg2_loglik = self.compute_objective(theta)
        
        # Convert to log-likelihood
        loglik = -0.5 * neg2_loglik
        
        return mu, sigma, loglik
    
    def check_convergence(self, theta: np.ndarray, 
                         grad: Optional[np.ndarray] = None,
                         tol: float = 1e-6) -> bool:
        """
        Check convergence using R's criterion.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameter vector
        grad : np.ndarray or None
            Current gradient (computed if None)
        tol : float
            Convergence tolerance
            
        Returns
        -------
        bool
            True if converged
            
        Notes
        -----
        Uses gradient norm criterion: ||grad||∞ < tol
        This matches R's nlm convergence check.
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Check maximum absolute gradient (infinity norm)
        max_grad = np.max(np.abs(grad))
        
        return max_grad < tol
    
    def get_diagnostics(self, theta: np.ndarray) -> Dict[str, Any]:
        """
        Get diagnostic information for current parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        dict
            Diagnostic information
        """
        mu, sigma, loglik = self.extract_parameters(theta)
        grad = self.compute_gradient(theta)
        
        # Compute condition number of covariance
        eigenvals = np.linalg.eigvalsh(sigma)
        condition_number = np.max(eigenvals) / np.min(eigenvals)
        
        diagnostics = {
            'loglik': loglik,
            'neg2_loglik': -2 * loglik,
            'gradient_norm': np.linalg.norm(grad),
            'gradient_max': np.max(np.abs(grad)),
            'sigma_min_eigenval': np.min(eigenvals),
            'sigma_max_eigenval': np.max(eigenvals),
            'sigma_condition_number': condition_number,
            'n_patterns': self.n_patterns,
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'n_params': self.n_params,
            'missing_rate': 1 - (self.total_observed / (self.n_obs * self.n_vars))
        }
        
        return diagnostics
    
    def validate_parameters(self, theta: np.ndarray) -> Tuple[bool, str]:
        """
        Validate parameter vector.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector to validate
            
        Returns
        -------
        valid : bool
            Whether parameters are valid
        message : str
            Error message if invalid
        """
        # Check length
        if len(theta) != self.n_params:
            return False, f"Expected {self.n_params} parameters, got {len(theta)}"
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(theta)):
            return False, "Parameters contain NaN or Inf"
        
        # Try to unpack and check covariance
        try:
            mu, sigma = self.parameterization.unpack(theta)
            
            # Check positive definiteness
            eigenvals = np.linalg.eigvalsh(sigma)
            if np.min(eigenvals) <= 0:
                return False, f"Covariance not positive definite (min eigenval = {np.min(eigenvals)})"
            
            # Check for reasonable condition number
            cond = np.max(eigenvals) / np.min(eigenvals)
            if cond > 1e10:
                return False, f"Covariance nearly singular (condition number = {cond:.2e})"
            
        except Exception as e:
            return False, f"Failed to unpack parameters: {e}"
        
        return True, "Parameters valid"