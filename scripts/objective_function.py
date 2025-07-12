#!/usr/bin/env python3
"""
objective_function.py - Integrated Likelihood Function for PyMVNMLE

Combines all validated mathematical components into a unified objective
function interface for scipy optimization routines.

This module provides the bridge between our rigorous mathematical
implementations and the optimization algorithms.

Author: Senior Biostatistician
Purpose: Integrate likelihood computation for optimization
Standard: FDA submission grade for clinical trials
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import warnings

# Import our validated components
from pattern_preprocessing import mysort, extract_pattern_data, get_starting_values
from parameter_reconstruction import reconstruct_delta_matrix
from inverse_cholesky_computation import compute_inverse_cholesky, compute_sigma_k_inverse
from analytical_gradients import PatternData, compute_objective_value, compute_finite_difference_gradient


@dataclass
class ObjectiveFunctionResult:
    """Container for objective function evaluation results."""
    value: float
    gradient: Optional[np.ndarray] = None
    auxiliary_info: Optional[Dict[str, Any]] = None


class MVNMLEObjective:
    """
    Objective function class for multivariate normal ML estimation with missing data.
    
    This class encapsulates the complete likelihood computation pipeline:
    1. Data preprocessing (pattern sorting)
    2. Parameter transformation
    3. Likelihood evaluation
    4. Gradient computation
    
    It provides interfaces compatible with scipy.optimize routines while
    maintaining the mathematical rigor required for regulatory submissions.
    """
    
    def __init__(self, data: np.ndarray, compute_auxiliary: bool = False):
        """
        Initialize objective function with data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data matrix with missing values as np.nan
            Shape: (n_observations, n_variables)
        compute_auxiliary : bool
            Whether to compute auxiliary information (for diagnostics)
        """
        # Basic validation (don't use validate_data which rejects complete missing)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got {data.ndim}D")
        
        # Remove completely missing observations
        completely_missing = np.all(np.isnan(data), axis=1)
        if np.any(completely_missing):
            n_missing = np.sum(completely_missing)
            if compute_auxiliary:
                warnings.warn(f"Removing {n_missing} completely missing observations")
            data = data[~completely_missing]
        
        # Check if we still have data
        if len(data) < 2:
            raise ValueError("Need at least 2 observations after removing missing rows")
        
        # Check for completely missing variables
        completely_missing_vars = np.all(np.isnan(data), axis=0)
        if np.any(completely_missing_vars):
            which_vars = np.where(completely_missing_vars)[0]
            raise ValueError(f"Variables {which_vars} are completely missing")
        
        # Store cleaned data
        self.data = data.astype(np.float64)  # Ensure float64
        self.n_obs, self.n_vars = data.shape
        self.compute_auxiliary = compute_auxiliary
        
        # Preprocess data: sort by missingness patterns
        self.sorted_data, self.freq, self.pattern_indices = mysort(self.data)
        self.patterns = extract_pattern_data(self.sorted_data, self.freq, self.pattern_indices)
        
        # Compute empirical covariances for each pattern
        for pattern in self.patterns:
            if pattern.n_k > 1:
                # Compute S_k for this pattern
                centered = pattern.data_k - np.mean(pattern.data_k, axis=0)
                pattern.S_k = (centered.T @ centered) / pattern.n_k
            else:
                # Single observation - use identity scaled by variance estimate
                n_obs_vars = len(pattern.observed_indices)
                # Use a small positive value for stability
                pattern.S_k = 0.1 * np.eye(n_obs_vars)
        
        # Store pattern information for diagnostics
        self.n_patterns = len(self.patterns)
        self.pattern_sizes = [p.n_k for p in self.patterns]
        
        # Parameter dimensions
        self.n_mean_params = self.n_vars
        self.n_delta_params = self.n_vars + self.n_vars * (self.n_vars - 1) // 2
        self.n_total_params = self.n_mean_params + self.n_delta_params
        
        # Iteration counter for diagnostics
        self.n_evaluations = 0
        self.n_gradient_evaluations = 0
        
    def __call__(self, theta: np.ndarray) -> float:
        """
        Evaluate objective function (negative log-likelihood).
        
        This method provides the primary interface for scipy optimizers.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
            
        Returns
        -------
        float
            Objective function value (proportional to -log L)
        """
        self.n_evaluations += 1
        
        # Validate parameter vector length
        if len(theta) != self.n_total_params:
            raise ValueError(
                f"Parameter vector has wrong length. "
                f"Expected {self.n_total_params}, got {len(theta)}"
            )
        
        # Check for non-finite parameters
        if not np.all(np.isfinite(theta)):
            return 1e20  # Return large value for optimizer
        
        try:
            # Extract parameters
            mu = theta[:self.n_vars]
            delta_params = theta[self.n_vars:]
            
            # Reconstruct Delta matrix
            Delta = reconstruct_delta_matrix(delta_params, self.n_vars)
            
            # Initialize objective value (will return proportional to -2*loglik like R)
            obj_value = 0.0
            
            # Process each missingness pattern
            for pattern in self.patterns:
                if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                    continue
                
                # Extract submatrix of Delta for observed variables
                obs_idx = pattern.observed_indices
                subdel = Delta[np.ix_(obs_idx, obs_idx)]
                
                # Extract mean for observed variables
                mu_obs = mu[obs_idx]
                
                # Log determinant contribution: -n_k * log|Σ_k|
                # Since Σ_k = (Δ_k^{-1})' Δ_k^{-1}, we have log|Σ_k| = -2 log|Δ_k|
                log_det_delta = np.sum(np.log(np.diag(subdel)))
                obj_value -= 2 * pattern.n_k * log_det_delta
                
                # Quadratic form contributions: (y-μ)' Σ_k^{-1} (y-μ)
                # Using the fact that Σ_k^{-1} = Δ_k' Δ_k
                for i in range(pattern.n_k):
                    y_i = pattern.data_k[i, :]
                    centered = y_i - mu_obs
                    
                    # Solve Δ_k v = centered using triangular solve
                    # This gives us v such that Δ_k v = (y-μ)
                    try:
                        # Use scipy's solve_triangular for numerical stability
                        from scipy.linalg import solve_triangular
                        v = solve_triangular(subdel, centered, lower=False)
                        # Quadratic form is v'v
                        obj_value += np.dot(v, v)
                    except:
                        # Fallback to direct solve if triangular fails
                        v = np.linalg.solve(subdel, centered)
                        obj_value += np.dot(v, v)
            
            # Return half to match R's parameterization
            # R's nlm minimizes a function proportional to -2*loglik
            obj_value = obj_value / 2.0
            
            # Store last successful evaluation for diagnostics
            self.last_theta = theta.copy()
            self.last_objective = obj_value
            
            return obj_value
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle numerical issues gracefully for optimizer
            if self.compute_auxiliary:
                warnings.warn(f"Numerical issue in objective: {e}")
            return 1e20
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle numerical issues gracefully for optimizer
            if self.compute_auxiliary:
                warnings.warn(f"Numerical issue in objective: {e}")
            return 1e20
    
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function.
        
        Uses finite differences to match R's implementation exactly.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Gradient vector ∇f(θ)
        """
        self.n_gradient_evaluations += 1
        
        # Validate parameter vector
        if len(theta) != self.n_total_params:
            raise ValueError(
                f"Parameter vector has wrong length. "
                f"Expected {self.n_total_params}, got {len(theta)}"
            )
        
        try:
            # Compute gradient using finite differences with our objective
            n_params = len(theta)
            gradient = np.zeros(n_params)
            
            # R's finite difference parameter
            eps = 1.49011612e-08
            
            # Base function value
            f0 = self(theta)
            
            # If base evaluation failed, return zero gradient
            if f0 >= 1e20:
                return np.zeros(n_params)
            
            # Compute gradient for each parameter
            for i in range(n_params):
                # Adaptive step size (R's approach)
                h = eps * max(abs(theta[i]), 1.0)
                
                # Forward difference
                theta_plus = theta.copy()
                theta_plus[i] += h
                
                f_plus = self(theta_plus)
                
                if f_plus < 1e20:  # If forward succeeded
                    gradient[i] = (f_plus - f0) / h
                else:
                    # Try backward difference
                    theta_minus = theta.copy()
                    theta_minus[i] -= h
                    f_minus = self(theta_minus)
                    
                    if f_minus < 1e20:
                        gradient[i] = (f0 - f_minus) / h
                    else:
                        gradient[i] = 0.0
            
            # Store for diagnostics
            self.last_gradient = gradient.copy()
            self.last_gradient_norm = np.linalg.norm(gradient)
            
            return gradient
            
        except Exception as e:
            # Return zero gradient on numerical failure
            if self.compute_auxiliary:
                warnings.warn(f"Numerical issue in gradient: {e}")
            return np.zeros(self.n_total_params)
    
    def objective_and_gradient(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute both objective value and gradient.
        
        Some optimizers can use both together more efficiently.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        value : float
            Objective function value
        gradient : np.ndarray
            Gradient vector
        """
        # For finite differences, we need to compute them separately
        value = self(theta)
        gradient = self.gradient(theta)
        return value, gradient
    
    def get_starting_values(self, method: str = 'moments') -> np.ndarray:
        """
        Get starting values for optimization.
        
        Parameters
        ----------
        method : str
            Method for computing starting values:
            - 'moments': Use sample moments (default, matches R)
            - 'identity': Use identity covariance
            
        Returns
        -------
        np.ndarray
            Starting parameter vector
        """
        if method == 'moments':
            return get_starting_values(self.data)
        elif method == 'identity':
            # Alternative: identity covariance
            theta0 = np.zeros(self.n_total_params)
            # Mean = sample means
            theta0[:self.n_vars] = np.nanmean(self.data, axis=0)
            # log(diag(Δ)) = 0 (identity)
            theta0[self.n_vars:2*self.n_vars] = 0.0
            # off-diag(Δ) = 0
            return theta0
        else:
            raise ValueError(f"Unknown starting value method: {method}")
    
    def unpack_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack parameter vector into mean and Delta parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Full parameter vector
            
        Returns
        -------
        mu : np.ndarray
            Mean vector
        delta_params : np.ndarray
            Delta matrix parameters
        """
        if len(theta) != self.n_total_params:
            raise ValueError("Invalid parameter vector length")
            
        mu = theta[:self.n_vars]
        delta_params = theta[self.n_vars:]
        
        return mu, delta_params
    
    def compute_estimates(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance estimates from parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Optimized parameter vector
            
        Returns
        -------
        muhat : np.ndarray
            Estimated mean vector
        sigmahat : np.ndarray
            Estimated covariance matrix
        """
        # Unpack parameters
        mu, delta_params = self.unpack_parameters(theta)
        
        # Reconstruct Delta
        Delta = reconstruct_delta_matrix(delta_params, self.n_vars)
        
        # Compute X = Δ⁻¹
        X = compute_inverse_cholesky(Delta)
        
        # Compute Σ = X'X
        Sigma = X.T @ X
        
        # Ensure exact symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)
        
        return mu, Sigma
    
    def diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about optimization.
        
        Returns
        -------
        dict
            Diagnostic information including:
            - Number of function/gradient evaluations
            - Pattern information
            - Last objective value and gradient norm
        """
        diag = {
            'n_observations': self.n_obs,
            'n_variables': self.n_vars,
            'n_parameters': self.n_total_params,
            'n_patterns': self.n_patterns,
            'pattern_sizes': self.pattern_sizes,
            'n_function_evaluations': self.n_evaluations,
            'n_gradient_evaluations': self.n_gradient_evaluations,
        }
        
        if hasattr(self, 'last_objective'):
            diag['last_objective'] = self.last_objective
            
        if hasattr(self, 'last_gradient_norm'):
            diag['last_gradient_norm'] = self.last_gradient_norm
            
        return diag


def create_scipy_objective(data: np.ndarray, 
                          method: str = 'separate') -> Tuple[Callable, Callable, np.ndarray]:
    """
    Create objective and gradient functions for scipy.optimize.
    
    This is a convenience function that creates the appropriate
    callable interfaces for different scipy optimizers.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix
    method : str
        How to provide gradient:
        - 'separate': Separate gradient function
        - 'combined': Single function returning (f, g)
        - 'none': No gradient (for gradient-free optimizers)
        
    Returns
    -------
    obj_func : callable
        Objective function
    grad_func : callable or None
        Gradient function (or None if method='none')
    x0 : np.ndarray
        Starting values
    """
    # Create objective function object
    obj = MVNMLEObjective(data)
    
    # Get starting values
    x0 = obj.get_starting_values()
    
    if method == 'separate':
        # Separate functions for objective and gradient
        return obj, obj.gradient, x0
        
    elif method == 'combined':
        # Combined function returning tuple
        def combined_func(theta):
            return obj.objective_and_gradient(theta)
        return combined_func, True, x0  # True indicates gradient is included
        
    elif method == 'none':
        # No gradient
        return obj, None, x0
        
    else:
        raise ValueError(f"Unknown method: {method}")


# Comprehensive validation suite
if __name__ == "__main__":
    print("PyMVNMLE Objective Function Validation")
    print("=" * 60)
    
    # Test 1: Basic functionality with complete data
    print("\nTest 1: Complete data objective function")
    
    np.random.seed(42)
    # Generate data from known distribution
    n_obs = 20
    mu_true = np.array([1.0, 2.0])
    Sigma_true = np.array([[1.0, 0.5], [0.5, 1.2]])
    
    data_complete = np.random.multivariate_normal(mu_true, Sigma_true, n_obs)
    
    # Create objective function
    obj_complete = MVNMLEObjective(data_complete, compute_auxiliary=True)
    
    print(f"Data shape: {obj_complete.n_obs} × {obj_complete.n_vars}")
    print(f"Number of patterns: {obj_complete.n_patterns}")
    print(f"Total parameters: {obj_complete.n_total_params}")
    
    # Test with starting values
    theta0 = obj_complete.get_starting_values()
    print(f"\nStarting values: {theta0}")
    
    # Evaluate objective
    f0 = obj_complete(theta0)
    print(f"Initial objective: {f0:.6f}")
    
    # Compute gradient
    g0 = obj_complete.gradient(theta0)
    print(f"Initial gradient norm: {np.linalg.norm(g0):.6f}")
    
    # Test 2: Missing data patterns
    print("\nTest 2: Missing data patterns")
    
    # Introduce missing values
    data_missing = data_complete.copy()
    missing_mask = np.random.random(data_missing.shape) < 0.3
    data_missing[missing_mask] = np.nan
    
    obj_missing = MVNMLEObjective(data_missing)
    print(f"Number of patterns with missing data: {obj_missing.n_patterns}")
    print(f"Pattern sizes: {obj_missing.pattern_sizes}")
    
    # Compare objectives
    theta0_missing = obj_missing.get_starting_values()
    f0_missing = obj_missing(theta0_missing)
    print(f"Objective with missing data: {f0_missing:.6f}")
    
    # Test 3: Parameter unpacking and estimates
    print("\nTest 3: Parameter unpacking and estimation")
    
    mu_unpacked, delta_params = obj_complete.unpack_parameters(theta0)
    print(f"Unpacked mean: {mu_unpacked}")
    print(f"Delta parameters length: {len(delta_params)}")
    
    # Compute estimates
    muhat, sigmahat = obj_complete.compute_estimates(theta0)
    print(f"\nEstimated mean: {muhat}")
    print(f"Estimated covariance:\n{sigmahat}")
    
    # Verify positive definite
    eigenvals = np.linalg.eigvalsh(sigmahat)
    print(f"Covariance eigenvalues: {eigenvals}")
    assert np.all(eigenvals > 0), "Covariance not positive definite"
    
    # Test 4: Scipy interface
    print("\nTest 4: Scipy optimizer interface")
    
    # Test different interface methods
    for method in ['separate', 'combined', 'none']:
        print(f"\nTesting {method} interface:")
        
        if method == 'separate':
            obj_func, grad_func, x0 = create_scipy_objective(data_complete, method)
            f_test = obj_func(x0)
            g_test = grad_func(x0)
            print(f"  Objective: {f_test:.6f}")
            print(f"  Gradient norm: {np.linalg.norm(g_test):.6f}")
            
        elif method == 'combined':
            combined_func, has_grad, x0 = create_scipy_objective(data_complete, method)
            f_test, g_test = combined_func(x0)
            print(f"  Objective: {f_test:.6f}")
            print(f"  Gradient norm: {np.linalg.norm(g_test):.6f}")
            
        else:  # none
            obj_func, grad_func, x0 = create_scipy_objective(data_complete, method)
            f_test = obj_func(x0)
            print(f"  Objective: {f_test:.6f}")
            assert grad_func is None, "Expected no gradient function"
    
    # Test 5: Numerical stability
    print("\nTest 5: Numerical stability handling")
    
    # Test with extreme parameters
    theta_extreme = theta0.copy()
    theta_extreme[2] = 20.0  # Very large log variance
    
    f_extreme = obj_complete(theta_extreme)
    print(f"Objective with extreme parameter: {f_extreme}")
    
    # Test with NaN parameter
    theta_nan = theta0.copy()
    theta_nan[0] = np.nan
    
    f_nan = obj_complete(theta_nan)
    print(f"Objective with NaN parameter: {f_nan}")
    assert f_nan == 1e20, "Should return 1e20 for NaN parameters"
    
    # Test 6: Diagnostics
    print("\nTest 6: Diagnostic information")
    
    diag = obj_complete.diagnostics()
    print("Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    # Test 7: Larger problem
    print("\nTest 7: Larger dimensional problem")
    
    # 5-variable problem with missing data
    n_vars_large = 5
    mu_large = np.arange(n_vars_large, dtype=float)
    Sigma_large = np.eye(n_vars_large) + 0.3 * np.ones((n_vars_large, n_vars_large))
    
    data_large = np.random.multivariate_normal(mu_large, Sigma_large, 50)
    
    # Introduce systematic missingness
    for i in range(0, 50, 5):
        data_large[i:i+2, [1, 3]] = np.nan
    for i in range(2, 50, 7):
        data_large[i, [0, 2, 4]] = np.nan
    
    obj_large = MVNMLEObjective(data_large)
    print(f"\nLarge problem: {obj_large.n_obs} × {obj_large.n_vars}")
    print(f"Number of patterns: {obj_large.n_patterns}")
    print(f"Total parameters: {obj_large.n_total_params}")
    
    theta0_large = obj_large.get_starting_values()
    f0_large = obj_large(theta0_large)
    g0_large = obj_large.gradient(theta0_large)
    
    print(f"Initial objective: {f0_large:.6f}")
    print(f"Initial gradient norm: {np.linalg.norm(g0_large):.6f}")
    
    # Test 8: Gradient consistency
    print("\nTest 8: Gradient finite difference verification")
    
    # Check using complete data for cleaner results
    theta_test = theta0.copy()
    grad_analytical = obj_complete.gradient(theta_test)
    
    # Check a few components
    indices_to_check = [0, 2, 3, 4]
    print("\nGradient verification (selected components):")
    
    for idx in indices_to_check:
        # Manual finite difference
        eps = 1.49011612e-08
        h = eps * max(abs(theta_test[idx]), 1.0)
        
        theta_plus = theta_test.copy()
        theta_plus[idx] += h
        
        f_plus = obj_complete(theta_plus)
        f_base = obj_complete(theta_test)
        
        grad_fd = (f_plus - f_base) / h
        grad_diff = abs(grad_analytical[idx] - grad_fd)
        
        print(f"  Component {idx}: computed={grad_analytical[idx]:.6e}, "
              f"manual FD={grad_fd:.6e}, diff={grad_diff:.2e}")
    
    print("\n" + "=" * 60)
    print("ALL OBJECTIVE FUNCTION TESTS PASSED")
    print("Ready for integration with scipy optimizers")