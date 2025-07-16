"""
Essential utilities for PyMVNMLE v2.0
Core functions needed for missing data MLE computation

DESIGN PRINCIPLE: Preserve essential functionality while staying lean
This module provides the utilities actually needed by the core pipeline:
1. Data validation and preprocessing
2. R-compatible pattern sorting (mysort)
3. Convergence checking and result formatting
4. Essential parameter reconstruction

Removed: Complex diagnostics, system introspection, advanced preprocessing
Kept: Core mathematical utilities required for correctness
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any, Optional
from scipy.optimize import OptimizeResult
import warnings


def mysort_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort data by missingness patterns (direct port of R's mysort).
    
    This is CRITICAL for computational efficiency and R compatibility.
    The R algorithm sorts observations by missingness pattern to group
    identical patterns together for efficient likelihood computation.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_observations, n_variables)
        Input data matrix with missing values as np.nan
        
    Returns
    -------
    sorted_data : np.ndarray
        Data matrix with rows reordered by missingness pattern
    freq : np.ndarray
        Number of observations in each missingness pattern block
    presence_absence : np.ndarray
        Binary matrix indicating observed variables for each pattern,
        shape (n_patterns, n_variables). 1 = observed, 0 = missing
        
    Notes
    -----
    This implements the exact algorithm from R's mvnmle package for
    regulatory compliance and computational efficiency.
    """
    n_obs, n_vars = data.shape
    
    # Create binary representation (1=observed, 0=missing)
    # R: binrep <- ifelse(is.na(x), 0, 1)
    is_observed = (~np.isnan(data)).astype(int)
    
    # Convert to decimal representation for sorting
    # R: powers <- as.integer(2^((nvars-1):0))
    # R: decrep <- binrep %*% powers
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_codes = is_observed @ powers
    
    # Sort by pattern codes
    # R: sorted <- x[order(decrep), ]
    sort_indices = np.argsort(pattern_codes)
    sorted_data = data[sort_indices]
    sorted_patterns = is_observed[sort_indices]
    sorted_codes = pattern_codes[sort_indices]
    
    # Count frequency of each unique pattern
    # R: freq = as.vector(table(decrep))
    unique_codes, freq = np.unique(sorted_codes, return_counts=True)
    
    # Extract unique patterns
    presence_absence = []
    current_code = -1
    for i, code in enumerate(sorted_codes):
        if code != current_code:
            presence_absence.append(sorted_patterns[i])
            current_code = code
    
    presence_absence = np.array(presence_absence)
    
    return sorted_data, freq, presence_absence

def validate_input_data(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Validate and preprocess input data for ML estimation.
    
    Performs regulatory-grade validation required for reliable MLE computation.
    Based on the existing validated implementation.
    
    Parameters
    ----------
    data : array-like
        Input data matrix (observations × variables)
        
    Returns
    -------
    np.ndarray
        Validated data as NumPy array with float64 dtype
        
    Raises
    ------
    ValueError
        If data format is invalid for MLE computation
        
    Notes
    -----
    Validation includes:
    - Type conversion and numeric validation
    - Dimensionality checks
    - Missing data pattern validation
    - Sufficient data requirements
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to NumPy, preserving column names for error messages
        data_array = data.values
        var_names = data.columns.tolist()
    else:
        data_array = np.asarray(data)
        var_names = None  # Will create names later if needed
    
    # Dimensionality check
    if data_array.ndim != 2:
        raise ValueError(f"Data must be a 2D array (observations × variables), got {data_array.ndim}D")
    
    n_obs, n_vars = data_array.shape
    
    # Create variable names if not from DataFrame
    if var_names is None:
        var_names = [f"Variable_{i}" for i in range(n_vars)]
    
    # Ensure numeric data - CHANGED FROM TypeError TO ValueError
    if not np.issubdtype(data_array.dtype, np.number):
        raise ValueError("Data must contain numeric values only")
    
    # Convert to float64
    data_array = data_array.astype(np.float64)
    
    # Sample size validation
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations for estimation, got {n_obs}")
    
    if n_vars < 1:
        raise ValueError(f"Need at least 1 variable, got {n_vars}")
    
    # Check for infinite values BEFORE other validations
    non_finite_mask = ~np.isfinite(data_array) & ~np.isnan(data_array)
    if np.any(non_finite_mask):
        # Find location of first infinite value for helpful error message
        inf_locations = np.where(non_finite_mask)
        first_row, first_col = inf_locations[0][0], inf_locations[1][0]
        raise ValueError(f"Data contains infinite values (first at row {first_row}, column {first_col})")
    
    # Check for completely missing observations
    completely_missing = np.all(np.isnan(data_array), axis=1)
    if np.any(completely_missing):
        n_missing = np.sum(completely_missing)
        warnings.warn(f"Removing {n_missing} completely missing observations")
        data_array = data_array[~completely_missing]
        n_obs = data_array.shape[0]
        
        if n_obs < 2:
            raise ValueError("Too few observations after removing completely missing rows")
    
    # Check for completely missing variables
    completely_missing_vars = np.all(np.isnan(data_array), axis=0)
    if np.any(completely_missing_vars):
        missing_var_indices = np.where(completely_missing_vars)[0]
        missing_var_names = [var_names[i] for i in missing_var_indices]
        raise ValueError(f"Variable '{missing_var_names[0]}' is completely missing")
    
    return data_array


def reconstruct_delta_matrix(theta: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Reconstruct the upper triangular Δ matrix from parameter vector.
    
    The parameter vector is structured as:
    θ = [μ₁, ..., μₚ, log(δ₁₁), ..., log(δₚₚ), δ₁₂, δ₁₃, δ₂₃, ...]
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    n_vars : int
        Number of variables (p)
        
    Returns
    -------
    np.ndarray, shape (n_vars, n_vars)
        Upper triangular Δ matrix
        
    Notes
    -----
    Critical for inverse Cholesky parameterization: Σ = (Δ⁻¹)ᵀ Δ⁻¹
    """
    Delta = np.zeros((n_vars, n_vars))
    
    # Extract diagonal elements (from log parameters to ensure positivity)
    log_diag = theta[n_vars:2*n_vars]
    Delta[np.diag_indices(n_vars)] = np.exp(log_diag)
    
    # Extract off-diagonal elements (upper triangle, column by column)
    idx = 2 * n_vars
    for j in range(n_vars):
        for i in range(j):
            Delta[i, j] = theta[idx]
            idx += 1
    
    return Delta


def extract_parameters(theta: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean vector and covariance matrix from parameter vector.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    n_vars : int
        Number of variables
        
    Returns
    -------
    mu : np.ndarray, shape (n_vars,)
        Mean vector
    sigma : np.ndarray, shape (n_vars, n_vars)
        Covariance matrix
        
    Notes
    -----
    Converts from inverse Cholesky parameterization back to μ, Σ.
    """
    # Extract mean parameters
    mu = theta[:n_vars]
    
    # Reconstruct Δ matrix
    Delta = reconstruct_delta_matrix(theta, n_vars)
    
    # Convert to covariance matrix: Σ = (Δ⁻¹)ᵀ Δ⁻¹
    try:
        Delta_inv = np.linalg.inv(Delta)
        sigma = Delta_inv.T @ Delta_inv
    except np.linalg.LinAlgError:
        # Fallback for numerical issues
        sigma = np.eye(n_vars)
        warnings.warn("Numerical issues in parameter extraction, using identity covariance")
    
    return mu, sigma


def check_convergence(opt_result: OptimizeResult, 
                     final_gradient_norm: Optional[float] = None,
                     tolerance: float = 1e-6) -> bool:
    """
    Check if optimization converged successfully.
    
    Simple convergence assessment based on optimizer status and gradient norm.
    
    Parameters
    ----------
    opt_result : OptimizeResult
        Result from scipy.optimize.minimize
    final_gradient_norm : float, optional
        Final gradient norm for additional convergence check
    tolerance : float, default=1e-6
        Convergence tolerance for gradient norm
        
    Returns
    -------
    bool
        True if optimization converged successfully
    """
    # Primary check: optimizer success flag
    if not opt_result.success:
        return False
    
    # Secondary check: gradient norm (if available)
    if final_gradient_norm is not None:
        # Allow some slack for finite difference gradients
        if final_gradient_norm > tolerance * 100:
            return False
    
    return True


def format_result(opt_result: OptimizeResult,
                 theta_opt: np.ndarray, 
                 n_vars: int,
                 backend_name: str,
                 gpu_accelerated: bool,
                 computation_time: float) -> Dict[str, Any]:
    """
    Format optimization result into standard dictionary.
    
    Extracts essential information from optimization result for MLResult creation.
    
    Parameters
    ----------
    opt_result : OptimizeResult
        Raw optimization result
    theta_opt : np.ndarray
        Optimal parameter vector
    n_vars : int
        Number of variables
    backend_name : str
        Name of backend used
    gpu_accelerated : bool
        Whether GPU acceleration was used
    computation_time : float
        Computation time in seconds
        
    Returns
    -------
    dict
        Formatted result dictionary for MLResult object
    """
    # Extract basic convergence info
    converged = opt_result.success
    n_iter = getattr(opt_result, 'nit', 0)
    message = getattr(opt_result, 'message', 'Unknown')
    
    # Format convergence message for clarity
    if converged:
        if 'Optimization terminated successfully' in message:
            formatted_message = "Converged successfully"
        else:
            formatted_message = message
    else:
        formatted_message = f"Failed to converge: {message}"
    
    return {
        'converged': converged,
        'convergence_message': formatted_message,
        'n_iter': n_iter,
        'backend': backend_name,
        'gpu_accelerated': gpu_accelerated,
        'computation_time': computation_time,
        'gradient': getattr(opt_result, 'jac', None),
        'hessian': getattr(opt_result, 'hess', None)
    }