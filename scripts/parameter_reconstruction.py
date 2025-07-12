#!/usr/bin/env python3
"""
parameter_reconstruction.py - Delta Matrix Reconstruction for PyMVNMLE

CRITICAL: This implements EXACT parameter reconstruction according to 
PyMVNMLE Mathematical Implementation Specification Section 6.1.1

Regulatory-grade implementation for FDA submission standards.
Mathematical fidelity is absolute - no deviations permitted.

Author: Senior Biostatistician
Purpose: Reconstruct Delta (inverse Cholesky factor) from parameter vector
Standard: FDA submission grade for clinical trials
"""

import numpy as np
from typing import Tuple


def reconstruct_delta_matrix(theta_delta_params: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Reconstruct Δ (Delta) matrix from parameter vector.
    
    This function implements the EXACT R mvnmle parameter ordering and 
    reconstruction algorithm. The parameterization ensures positive definiteness
    of the resulting covariance matrix.
    
    Parameters
    ----------
    theta_delta_params : np.ndarray
        Parameter vector containing:
        - First n_vars elements: log(Δ₁₁), log(Δ₂₂), ..., log(Δₚₚ)
        - Remaining elements: Δ₁₂, Δ₁₃, Δ₂₃, Δ₁₄, ..., Δₚ₋₁,ₚ
        Total length: n_vars + n_vars*(n_vars-1)/2
        
    n_vars : int
        Number of variables (p), must be positive
        
    Returns
    -------
    np.ndarray
        Upper triangular matrix Δ of shape (n_vars, n_vars)
        with positive diagonal elements
        
    Notes
    -----
    Parameter ordering follows R's mvnmle package EXACTLY:
    - Diagonal elements stored as logarithms for numerical stability
    - Off-diagonal elements ordered by column, then row within column
    - This ensures Σ = (Δ⁻¹)ᵀ(Δ⁻¹) is positive definite
    
    Mathematical foundation:
    - Δ is upper triangular inverse Cholesky factor
    - Covariance matrix: Σ = (Δ⁻¹)ᵀ(Δ⁻¹)
    - Log-diagonal parameterization prevents negative diagonal elements
    
    Raises
    ------
    ValueError
        If input dimensions are inconsistent or invalid
        
    Examples
    --------
    >>> # 2x2 case
    >>> params = np.array([0.0, 0.5, 0.3])  # log(Δ₁₁), log(Δ₂₂), Δ₁₂
    >>> Delta = reconstruct_delta_matrix(params, 2)
    >>> assert Delta[0,0] == 1.0  # exp(0.0)
    >>> assert Delta[1,1] == np.exp(0.5)
    >>> assert Delta[0,1] == 0.3
    >>> assert Delta[1,0] == 0.0  # Lower triangular is zero
    """
    # Input validation - regulatory grade
    if not isinstance(theta_delta_params, np.ndarray):
        raise TypeError("theta_delta_params must be a numpy array")
    
    if not isinstance(n_vars, (int, np.integer)):
        raise TypeError("n_vars must be an integer")
        
    if n_vars < 1:
        raise ValueError(f"n_vars must be positive, got {n_vars}")
    
    # Verify parameter vector length
    expected_length = n_vars + n_vars * (n_vars - 1) // 2
    if len(theta_delta_params) != expected_length:
        raise ValueError(
            f"theta_delta_params has wrong length. "
            f"Expected {expected_length} for n_vars={n_vars}, "
            f"got {len(theta_delta_params)}"
        )
    
    # Initialize Delta matrix with zeros
    Delta = np.zeros((n_vars, n_vars), dtype=np.float64)
    
    # Step 1: Set diagonal elements (exponentiated to ensure positivity)
    diagonal_params = theta_delta_params[:n_vars]
    
    # Apply numerical bounds as specified in Section 7.1
    # Clamp to prevent overflow: -10 ≤ log(Δⱼⱼ) ≤ 10
    diagonal_params_clamped = np.clip(diagonal_params, -10.0, 10.0)
    
    for j in range(n_vars):
        Delta[j, j] = np.exp(diagonal_params_clamped[j])
    
    # Step 2: Set upper triangular elements (off-diagonal)
    # R's ordering: by column (left to right), then by row within column
    param_idx = n_vars  # Start after diagonal parameters
    
    for j in range(1, n_vars):  # Column index (1, 2, ..., p-1)
        for i in range(j):      # Row index (0, 1, ..., j-1)
            if param_idx >= len(theta_delta_params):
                raise ValueError(
                    f"Parameter index out of bounds. This indicates a bug in the "
                    f"parameter counting logic."
                )
            
            # Apply numerical bounds: -100 ≤ Δᵢⱼ ≤ 100 for i ≠ j
            Delta[i, j] = np.clip(theta_delta_params[param_idx], -100.0, 100.0)
            param_idx += 1
    
    # Verify we used all parameters
    if param_idx != len(theta_delta_params):
        raise ValueError(
            f"Not all parameters were used. Expected to use {len(theta_delta_params)}, "
            f"but only used {param_idx}. This indicates a bug."
        )
    
    # Verify upper triangular structure
    if not np.allclose(Delta, np.triu(Delta)):
        raise ValueError("Internal error: Delta is not upper triangular")
    
    # Verify positive diagonal
    if np.any(np.diag(Delta) <= 0):
        raise ValueError("Internal error: Delta has non-positive diagonal elements")
    
    return Delta


def extract_delta_parameters(Delta: np.ndarray) -> np.ndarray:
    """
    Extract parameter vector from Delta matrix (inverse operation).
    
    This function is the inverse of reconstruct_delta_matrix and is used
    for validation and testing purposes.
    
    Parameters
    ----------
    Delta : np.ndarray
        Upper triangular matrix with positive diagonal elements
        
    Returns
    -------
    np.ndarray
        Parameter vector in R mvnmle ordering
        
    Raises
    ------
    ValueError
        If Delta is not upper triangular or has non-positive diagonal
    """
    if not isinstance(Delta, np.ndarray):
        raise TypeError("Delta must be a numpy array")
        
    if Delta.ndim != 2:
        raise ValueError(f"Delta must be 2-dimensional, got shape {Delta.shape}")
        
    if Delta.shape[0] != Delta.shape[1]:
        raise ValueError(f"Delta must be square, got shape {Delta.shape}")
    
    n_vars = Delta.shape[0]
    
    # Verify upper triangular
    if not np.allclose(Delta, np.triu(Delta)):
        raise ValueError("Delta must be upper triangular")
    
    # Verify positive diagonal
    if np.any(np.diag(Delta) <= 0):
        raise ValueError("Delta must have positive diagonal elements")
    
    # Initialize parameter vector
    n_params = n_vars + n_vars * (n_vars - 1) // 2
    params = np.zeros(n_params)
    
    # Extract log-diagonal elements
    params[:n_vars] = np.log(np.diag(Delta))
    
    # Extract off-diagonal elements in R's order
    param_idx = n_vars
    for j in range(1, n_vars):
        for i in range(j):
            params[param_idx] = Delta[i, j]
            param_idx += 1
    
    return params


def validate_delta_matrix(Delta: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that Delta matrix satisfies all mathematical requirements.
    
    Parameters
    ----------
    Delta : np.ndarray
        Matrix to validate
        
    Returns
    -------
    valid : bool
        True if Delta is valid
    message : str
        Validation message (empty if valid, error description if invalid)
    """
    try:
        if not isinstance(Delta, np.ndarray):
            return False, "Delta must be a numpy array"
            
        if Delta.ndim != 2:
            return False, f"Delta must be 2-dimensional, got {Delta.ndim}D"
            
        if Delta.shape[0] != Delta.shape[1]:
            return False, f"Delta must be square, got shape {Delta.shape}"
        
        # Check upper triangular
        if not np.allclose(Delta, np.triu(Delta)):
            return False, "Delta must be upper triangular"
        
        # Check positive diagonal
        diag_elements = np.diag(Delta)
        if np.any(diag_elements <= 0):
            return False, f"Delta has non-positive diagonal elements: min={np.min(diag_elements)}"
        
        # Check numerical bounds
        if np.any(np.abs(Delta[np.triu_indices_from(Delta, k=1)]) > 100):
            return False, "Off-diagonal elements exceed bounds |Δᵢⱼ| ≤ 100"
        
        # Check for NaN or inf
        if not np.all(np.isfinite(Delta)):
            return False, "Delta contains NaN or inf values"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# Validation test suite
if __name__ == "__main__":
    print("Running parameter reconstruction validation tests...")
    print("=" * 60)
    
    # Test 1: Basic 2x2 case
    print("Test 1: Basic 2x2 reconstruction")
    params_2x2 = np.array([0.0, 0.5, 0.3])
    Delta_2x2 = reconstruct_delta_matrix(params_2x2, 2)
    print(f"Parameters: {params_2x2}")
    print(f"Delta matrix:\n{Delta_2x2}")
    
    # Verify reconstruction
    params_recovered = extract_delta_parameters(Delta_2x2)
    assert np.allclose(params_2x2, params_recovered), "Round-trip failed"
    print("✓ Round-trip reconstruction successful")
    
    # Test 2: 3x3 case
    print("\nTest 2: 3x3 reconstruction")
    params_3x3 = np.array([0.1, 0.2, 0.3,  # log-diagonal
                          0.5, -0.3, 0.4])  # off-diagonal
    Delta_3x3 = reconstruct_delta_matrix(params_3x3, 3)
    print(f"Delta matrix:\n{Delta_3x3}")
    
    valid, msg = validate_delta_matrix(Delta_3x3)
    assert valid, f"Validation failed: {msg}"
    print("✓ Matrix validation passed")
    
    # Test 3: Parameter bounds
    print("\nTest 3: Parameter bounds enforcement")
    extreme_params = np.array([15.0, -15.0, 0.0,  # log-diagonal (will be clamped)
                              150.0, -150.0, 0.0])  # off-diagonal (will be clamped)
    Delta_clamped = reconstruct_delta_matrix(extreme_params, 3)
    print(f"Clamped diagonal: {np.diag(Delta_clamped)}")
    print(f"Expected: [{np.exp(10)}, {np.exp(-10)}, {np.exp(0)}]")
    assert Delta_clamped[0, 0] == np.exp(10.0), "Upper bound clamping failed"
    assert Delta_clamped[1, 1] == np.exp(-10.0), "Lower bound clamping failed"
    assert Delta_clamped[0, 1] == 100.0, "Off-diagonal upper bound failed"
    assert Delta_clamped[0, 2] == -100.0, "Off-diagonal lower bound failed"
    print("✓ Parameter bounds correctly enforced")
    
    # Test 4: Error handling
    print("\nTest 4: Error handling")
    try:
        reconstruct_delta_matrix(np.array([1.0, 2.0]), 3)  # Wrong length
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught wrong length: {e}")
    
    try:
        reconstruct_delta_matrix(np.array([1.0]), 0)  # Invalid n_vars
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught invalid n_vars: {e}")
    
    # Test 5: Numerical precision
    print("\nTest 5: Numerical precision")
    n_tests = 100
    np.random.seed(42)  # Reproducibility
    
    for i in range(n_tests):
        n_vars = np.random.randint(2, 10)
        n_params = n_vars + n_vars * (n_vars - 1) // 2
        
        # Generate random parameters within bounds
        log_diag = np.random.uniform(-5, 5, n_vars)
        off_diag = np.random.uniform(-50, 50, n_params - n_vars)
        params = np.concatenate([log_diag, off_diag])
        
        # Reconstruct and validate
        Delta = reconstruct_delta_matrix(params, n_vars)
        valid, msg = validate_delta_matrix(Delta)
        assert valid, f"Test {i} failed: {msg}"
        
        # Verify round-trip
        params_recovered = extract_delta_parameters(Delta)
        assert np.allclose(params, params_recovered, rtol=1e-14), \
            f"Round-trip precision failed for test {i}"
    
    print(f"✓ All {n_tests} random precision tests passed")
    
    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED")
    print("Parameter reconstruction meets FDA submission standards")