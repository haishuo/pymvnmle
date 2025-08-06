#!/usr/bin/env python3
"""
Deep debugging of Apple dataset to find remaining discrepancy.
Compare every step with expected R behavior.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import datasets, mlest
from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
from pymvnmle._validation import load_r_reference


def deep_debug_apple():
    """Comprehensive debugging of Apple dataset processing."""
    
    print("=" * 70)
    print("DEEP DEBUG: APPLE DATASET ANALYSIS")
    print("=" * 70)
    
    # Load data
    data = datasets.apple
    print(f"\n1. RAW DATA")
    print("-" * 40)
    print(f"Shape: {data.shape}")
    print("\nFirst 10 rows:")
    for i in range(min(10, len(data))):
        print(f"  Row {i:2}: {data[i]}")
    
    # Create objective
    obj = CPUObjectiveFP64(data)
    
    print(f"\n2. PATTERN ANALYSIS")
    print("-" * 40)
    print(f"Number of patterns: {obj.n_patterns}")
    
    for i, pattern in enumerate(obj.patterns):
        print(f"\nPattern {i}:")
        print(f"  N observations: {pattern.n_obs}")
        print(f"  Observed variables: {pattern.observed_indices}")
        print(f"  Missing variables: {pattern.missing_indices}")
        print(f"  Data shape: {pattern.data.shape}")
        
        # Show sample statistics for this pattern
        if pattern.data.size > 0:
            pattern_mean = np.mean(pattern.data, axis=0)
            print(f"  Pattern mean: {pattern_mean}")
            if pattern.data.shape[0] > 1:
                pattern_cov = np.cov(pattern.data.T, ddof=1)
                print(f"  Pattern covariance shape: {pattern_cov.shape}")
                if pattern_cov.ndim == 0:
                    print(f"  Pattern variance: {pattern_cov}")
                else:
                    print(f"  Pattern covariance:\n{pattern_cov}")
    
    print(f"\n3. INITIAL PARAMETERS")
    print("-" * 40)
    print(f"Sample mean: {obj.sample_mean}")
    print(f"Sample covariance:\n{obj.sample_cov}")
    
    initial_params = obj.get_initial_parameters()
    print(f"\nInitial parameter vector length: {len(initial_params)}")
    print(f"Expected length: {obj.n_params}")
    
    # Decode initial parameters
    n_vars = obj.n_vars
    mu_init = initial_params[:n_vars]
    log_diag_init = initial_params[n_vars:2*n_vars]
    off_diag_init = initial_params[2*n_vars:]
    
    print(f"\nInitial μ: {mu_init}")
    print(f"Initial log(diag(Δ)): {log_diag_init}")
    print(f"Initial off-diag(Δ): {off_diag_init}")
    
    # Compute initial objective
    print(f"\n4. OBJECTIVE FUNCTION")
    print("-" * 40)
    
    try:
        obj_init = obj.compute_objective(initial_params)
        print(f"Initial objective value: {obj_init}")
        print(f"Initial log-likelihood: {-obj_init/2}")
    except Exception as e:
        print(f"ERROR computing objective: {e}")
        import traceback
        traceback.print_exc()
    
    # Try optimization
    print(f"\n5. OPTIMIZATION ATTEMPT")
    print("-" * 40)
    
    try:
        from scipy.optimize import minimize
        
        result = minimize(
            fun=obj.compute_objective,
            x0=initial_params,
            method='BFGS',
            options={'maxiter': 100, 'disp': True}
        )
        
        print(f"\nOptimization success: {result.success}")
        print(f"Final objective: {result.fun}")
        print(f"Final log-likelihood: {-result.fun/2}")
        print(f"Number of iterations: {result.nit}")
        
        # Decode final parameters
        mu_final = result.x[:n_vars]
        print(f"\nFinal μ: {mu_final}")
        
    except Exception as e:
        print(f"ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare with R
    print(f"\n6. R COMPARISON")
    print("-" * 40)
    
    try:
        r_ref = load_r_reference('apple')
        print(f"R log-likelihood: {r_ref['loglik']}")
        print(f"R mean: {r_ref['muhat']}")
        print(f"R covariance:\n{r_ref['sigmahat']}")
    except Exception as e:
        print(f"Could not load R reference: {e}")
    
    # Manual verification
    print(f"\n7. MANUAL PATTERN ORDER CHECK")
    print("-" * 40)
    
    # Check if patterns are in the right order
    if obj.n_patterns >= 2:
        first_n_obs = len(obj.patterns[0].observed_indices)
        second_n_obs = len(obj.patterns[1].observed_indices)
        
        print(f"First pattern has {first_n_obs} observed variables")
        print(f"Second pattern has {second_n_obs} observed variables")
        
        if first_n_obs >= second_n_obs:
            print("✓ Patterns are correctly ordered (complete cases first)")
        else:
            print("✗ PATTERNS ARE STILL IN WRONG ORDER!")
            print("  This is why we don't match R!")
    
    return obj


def check_pattern_data_details(obj):
    """Check detailed pattern data."""
    print(f"\n8. DETAILED PATTERN DATA")
    print("-" * 40)
    
    for i, pattern in enumerate(obj.patterns):
        print(f"\nPattern {i} detailed data:")
        print(f"  Pattern ID: {pattern.pattern_id}")
        print(f"  Start index: {pattern.pattern_start}")
        print(f"  End index: {pattern.pattern_end}")
        
        # Show first few rows of actual data
        print(f"  First 3 rows of pattern data:")
        for j in range(min(3, pattern.n_obs)):
            print(f"    {pattern.data[j]}")
        
        # Check data extraction
        if i == 0 and obj.n_patterns > 1:
            # For first pattern, check if it's really complete cases
            print(f"\n  Checking if first pattern is complete cases:")
            original_rows = obj.sorted_data[pattern.pattern_start:pattern.pattern_end]
            n_missing = np.sum(np.isnan(original_rows))
            print(f"    Missing values in first pattern: {n_missing}")
            if n_missing == 0:
                print("    ✓ First pattern is complete cases")
            else:
                print("    ✗ First pattern has missing values!")


if __name__ == "__main__":
    obj = deep_debug_apple()
    check_pattern_data_details(obj)
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    # Final diagnosis
    if obj.n_patterns >= 2:
        first_complete = len(obj.patterns[0].missing_indices) == 0
        expected_loglik = -74.217476
        
        if first_complete:
            print("✓ Pattern order looks correct")
            print("⚠ But log-likelihood still doesn't match!")
            print("  This suggests an issue in the objective computation")
        else:
            print("✗ Pattern order is STILL WRONG")
            print("  The reordering fix didn't work properly")
        
        print(f"\nExpected log-likelihood: {expected_loglik}")
        print("Current difference: 0.216")
        print("\nPossible remaining issues:")
        print("1. Pattern reordering not applied correctly")
        print("2. Objective computation has a bug")
        print("3. Initial parameters are different from R")
        print("4. Optimization settings need adjustment")