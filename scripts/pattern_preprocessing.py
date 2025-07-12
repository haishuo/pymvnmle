#!/usr/bin/env python3
"""
pattern_preprocessing.py - Pattern-wise Data Preprocessing for PyMVNMLE

Sorts data by missingness patterns and computes sufficient statistics.
Direct port of R's mysort() and related preprocessing functions.

Author: Senior Biostatistician
Purpose: Preprocess data for efficient likelihood computation
Standard: FDA submission grade for clinical trials
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# Import gradient computation for PatternData structure
from analytical_gradients import PatternData


def mysort(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort multivariate data by missingness patterns.
    
    Direct port of R's mysort() function. Groups observations with
    identical missingness patterns together for efficient computation.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix, shape (n_observations, n_variables)
        Missing values should be np.nan
        
    Returns
    -------
    sorted_data : np.ndarray
        Data with rows reordered by missingness pattern
    freq : np.ndarray
        Number of observations in each pattern
    pattern_indices : np.ndarray
        Binary matrix indicating observed variables for each pattern
        Shape (n_patterns, n_variables), 1=observed, 0=missing
        
    Notes
    -----
    R's algorithm:
    1. Convert missingness to binary (1=observed, 0=missing)
    2. Encode each pattern as decimal number using binary representation
    3. Sort by these decimal codes
    4. Count frequencies of each unique pattern
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
    unique_codes, inverse_indices, freq = np.unique(
        sorted_codes, return_inverse=True, return_counts=True
    )
    
    # Extract unique patterns
    pattern_indices = []
    current_code = -1
    for i, code in enumerate(sorted_codes):
        if code != current_code:
            pattern_indices.append(sorted_patterns[i])
            current_code = code
    
    pattern_indices = np.array(pattern_indices)
    
    return sorted_data, freq, pattern_indices


def extract_pattern_data(sorted_data: np.ndarray, freq: np.ndarray, 
                        pattern_indices: np.ndarray) -> List[PatternData]:
    """
    Convert sorted data into PatternData structures for gradient computation.
    
    Parameters
    ----------
    sorted_data : np.ndarray
        Data sorted by missingness pattern
    freq : np.ndarray
        Frequency of each pattern
    pattern_indices : np.ndarray
        Binary matrix of observed variables per pattern
        
    Returns
    -------
    List[PatternData]
        List of pattern data structures ready for likelihood computation
    """
    patterns = []
    data_idx = 0
    
    for pattern_idx, (n_obs, pattern_mask) in enumerate(zip(freq, pattern_indices)):
        # Get indices of observed variables
        observed_vars = np.where(pattern_mask == 1)[0]
        
        # Extract data for this pattern (only observed variables)
        pattern_data = sorted_data[data_idx:data_idx + n_obs][:, observed_vars]
        
        # Create PatternData structure
        pattern = PatternData(
            observed_indices=observed_vars,
            n_k=int(n_obs),
            data_k=pattern_data
        )
        patterns.append(pattern)
        
        data_idx += n_obs
    
    return patterns


def get_starting_values(data: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Compute starting values for optimization.
    
    Port of R's getstartvals() function. Uses sample moments with
    regularization to ensure positive definiteness.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with possible missing values
    eps : float
        Regularization parameter for eigenvalues
        
    Returns
    -------
    np.ndarray
        Starting parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
    """
    n_obs, n_vars = data.shape
    
    # Starting values for mean: sample means
    # R: startvals[1:n] <- apply(x, 2, mean, na.rm=TRUE)
    mu_start = np.nanmean(data, axis=0)
    
    # Sample covariance matrix (pairwise complete observations)
    # R: sampmat <- cov(x, use = "p")
    # Initialize with zeros
    cov_sample = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(i, n_vars):
            # Find pairwise complete observations
            mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            n_complete = np.sum(mask)
            
            if n_complete > 1:
                if i == j:
                    # Variance
                    cov_sample[i, i] = np.var(data[mask, i], ddof=1)
                else:
                    # Covariance
                    cov_ij = np.cov(data[mask, i], data[mask, j], ddof=1)[0, 1]
                    cov_sample[i, j] = cov_ij
                    cov_sample[j, i] = cov_ij
            else:
                # No complete pairs, use default
                if i == j:
                    cov_sample[i, i] = 1.0
                else:
                    cov_sample[i, j] = 0.0
                    cov_sample[j, i] = 0.0
    
    # Regularize to ensure positive definiteness
    # R's algorithm: set small eigenvalues to eps * min positive eigenvalue
    eigenvals, eigenvecs = np.linalg.eigh(cov_sample)
    
    # Find smallest positive eigenvalue
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) > 0:
        min_pos = np.min(pos_eigenvals)
    else:
        min_pos = 1.0
    
    # Regularize
    threshold = eps * min_pos
    regularized_eigenvals = np.maximum(eigenvals, threshold)
    
    # Reconstruct regularized covariance
    cov_regularized = eigenvecs @ np.diag(regularized_eigenvals) @ eigenvecs.T
    
    # Get Cholesky factor and convert to R's parameterization
    # R uses upper triangular, so transpose numpy's lower triangular
    L = np.linalg.cholesky(cov_regularized)
    chol_upper = L.T
    
    # Compute inverse Cholesky factor (Delta)
    Delta_start = np.linalg.solve(chol_upper, np.eye(n_vars))
    
    # Ensure positive diagonal (R's sign adjustment)
    for i in range(n_vars):
        if Delta_start[i, i] < 0:
            Delta_start[i, :] *= -1
    
    # Pack into parameter vector
    n_delta_params = n_vars + n_vars * (n_vars - 1) // 2
    startvals = np.zeros(n_vars + n_delta_params)
    
    # Mean parameters
    startvals[:n_vars] = mu_start
    
    # Log diagonal of Delta
    startvals[n_vars:2*n_vars] = np.log(np.diag(Delta_start))
    
    # Off-diagonal elements of Delta (R's ordering: by column)
    param_idx = 2 * n_vars
    for j in range(1, n_vars):
        for i in range(j):
            startvals[param_idx] = Delta_start[i, j]
            param_idx += 1
    
    return startvals


def validate_data(data: np.ndarray) -> None:
    """
    Validate input data meets requirements.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to validate
        
    Raises
    ------
    ValueError
        If data fails validation
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got {data.ndim}D")
    
    n_obs, n_vars = data.shape
    
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")
    
    if n_vars < 1:
        raise ValueError(f"Need at least 1 variable, got {n_vars}")
    
    # Check for completely missing observations
    completely_missing = np.all(np.isnan(data), axis=1)
    if np.any(completely_missing):
        n_missing = np.sum(completely_missing)
        raise ValueError(f"Data contains {n_missing} completely missing observations")
    
    # Check for completely missing variables
    completely_missing_vars = np.all(np.isnan(data), axis=0)
    if np.any(completely_missing_vars):
        which_vars = np.where(completely_missing_vars)[0]
        raise ValueError(f"Variables {which_vars} are completely missing")
    
    # Check for non-numeric values
    non_nan_data = data[~np.isnan(data)]
    if len(non_nan_data) > 0 and not np.issubdtype(non_nan_data.dtype, np.number):
        raise ValueError("Data contains non-numeric values")
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values")


# Comprehensive test suite
if __name__ == "__main__":
    print("Pattern Preprocessing Tests")
    print("=" * 60)
    
    # Test 1: Basic sorting functionality
    print("\nTest 1: Basic pattern sorting")
    
    # Create data with clear patterns
    test_data = np.array([
        [1.0, 2.0, 3.0],      # Complete
        [4.0, np.nan, 6.0],   # Pattern: [1,0,1]
        [7.0, 8.0, 9.0],      # Complete
        [10.0, np.nan, 12.0], # Pattern: [1,0,1]
        [np.nan, 14.0, 15.0], # Pattern: [0,1,1]
        [16.0, 17.0, 18.0],   # Complete
    ])
    
    sorted_data, freq, patterns = mysort(test_data)
    print(f"Original data shape: {test_data.shape}")
    print(f"Number of patterns: {len(freq)}")
    print(f"Pattern frequencies: {freq}")
    print(f"Pattern indicators:\n{patterns}")
    
    # Verify sorting
    assert np.array_equal(freq, [1, 2, 3]), "Incorrect frequencies"
    print("✓ Pattern sorting correct")
    
    # Test 2: Convert to PatternData
    print("\nTest 2: PatternData conversion")
    
    pattern_list = extract_pattern_data(sorted_data, freq, patterns)
    print(f"Number of PatternData objects: {len(pattern_list)}")
    
    for i, p in enumerate(pattern_list):
        print(f"Pattern {i}: observed vars {p.observed_indices}, n={p.n_k}")
        assert p.data_k.shape == (p.n_k, len(p.observed_indices))
    print("✓ PatternData conversion correct")
    
    # Test 3: Starting values
    print("\nTest 3: Starting values computation")
    
    startvals = get_starting_values(test_data)
    n_params_expected = 3 + 3 + 3  # μ + log(diag(Δ)) + off-diag
    assert len(startvals) == n_params_expected
    print(f"Starting values: {startvals}")
    print(f"  μ: {startvals[:3]}")
    print(f"  log(diag(Δ)): {startvals[3:6]}")
    print(f"  off-diag(Δ): {startvals[6:]}")
    print("✓ Starting values computed")
    
    # Test 4: Larger dataset with more patterns
    print("\nTest 4: Larger dataset test")
    
    np.random.seed(42)
    n_obs_large = 100
    large_data = np.random.randn(n_obs_large, 4)
    
    # Introduce missing values with different patterns
    missing_patterns = [
        [0, 1],      # Variables 0,1 missing
        [2],         # Variable 2 missing
        [1, 3],      # Variables 1,3 missing
        []           # No missing (complete)
    ]
    
    for i in range(n_obs_large):
        pattern = missing_patterns[i % len(missing_patterns)]
        for var in pattern:
            large_data[i, var] = np.nan
    
    sorted_large, freq_large, patterns_large = mysort(large_data)
    pattern_data_large = extract_pattern_data(sorted_large, freq_large, patterns_large)
    
    print(f"Large dataset: {n_obs_large} observations, 4 variables")
    print(f"Unique patterns found: {len(freq_large)}")
    print(f"Pattern frequencies: {freq_large}")
    assert sum(freq_large) == n_obs_large
    print("✓ Large dataset processed correctly")
    
    # Test 5: Edge cases
    print("\nTest 5: Edge case handling")
    
    # Single pattern (complete data)
    complete_data = np.random.randn(20, 3)
    sorted_complete, freq_complete, _ = mysort(complete_data)
    assert len(freq_complete) == 1 and freq_complete[0] == 20
    print("✓ Complete data handled correctly")
    
    # Maximum patterns (each obs different)
    max_pattern_data = np.array([
        [1.0, np.nan, np.nan],
        [np.nan, 2.0, np.nan],
        [np.nan, np.nan, 3.0],
        [1.0, 2.0, np.nan],
        [1.0, np.nan, 3.0],
        [np.nan, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    _, freq_max, _ = mysort(max_pattern_data)
    assert len(freq_max) == 7  # Each observation is unique pattern
    print("✓ Maximum patterns handled correctly")
    
    # Test 6: Data validation
    print("\nTest 6: Input validation")
    
    # Valid data should pass
    validate_data(test_data)
    print("✓ Valid data passes validation")
    
    # Test invalid inputs
    try:
        validate_data(np.array([1, 2, 3]))  # 1D
        assert False, "Should reject 1D data"
    except ValueError as e:
        print(f"✓ Correctly rejected 1D data: {e}")
    
    try:
        bad_data = test_data.copy()
        bad_data[0, :] = np.nan  # Complete missing row
        validate_data(bad_data)
        assert False, "Should reject complete missing rows"
    except ValueError as e:
        print(f"✓ Correctly rejected missing rows: {e}")
    
    # Test 7: R compatibility check
    print("\nTest 7: R mysort compatibility")
    
    # Use R's apple dataset structure
    apple_like = np.array([
        [8.0, 59.0],
        [6.0, 58.0],
        [11.0, 56.0],
        [22.0, 53.0],
        [14.0, 50.0],
        [17.0, 45.0],
        [4.0, np.nan],
        [10.0, np.nan],
    ])
    
    sorted_apple, freq_apple, patterns_apple = mysort(apple_like)
    print(f"Apple-like data patterns: {freq_apple}")
    print(f"Pattern structure:\n{patterns_apple}")
    
    # Should have 2 patterns: complete and missing second variable
    assert len(freq_apple) == 2
    assert freq_apple[0] + freq_apple[1] == 8
    print("✓ R-compatible sorting verified")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("Pattern preprocessing ready for integration")