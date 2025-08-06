#!/usr/bin/env python3
"""
Comprehensive debugging to find why we get -76.144 instead of -74.217.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.optimize import minimize

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import datasets, mlest
from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64


def debug_patterns():
    """Debug pattern extraction in detail."""
    print("=" * 70)
    print("PATTERN EXTRACTION DEBUG")
    print("=" * 70)
    
    data = datasets.apple
    obj = CPUObjectiveFP64(data)
    
    print(f"\n1. SORTED DATA ORDER")
    print("-" * 40)
    print("First 10 rows of sorted data:")
    for i in range(min(10, len(obj.sorted_data))):
        print(f"  {i:2}: {obj.sorted_data[i]}")
    
    print(f"\n2. PATTERN DETAILS")
    print("-" * 40)
    for i, pattern in enumerate(obj.patterns):
        print(f"\nPattern {i}:")
        print(f"  Pattern ID: {pattern.pattern_id}")
        print(f"  N obs: {pattern.n_obs}")
        print(f"  Observed vars: {pattern.observed_indices}")
        print(f"  Missing vars: {pattern.missing_indices}")
        print(f"  Start index: {pattern.pattern_start}")
        print(f"  End index: {pattern.pattern_end}")
        print(f"  Data shape: {pattern.data.shape}")
        
        # Verify data extraction
        orig_slice = obj.sorted_data[pattern.pattern_start:pattern.pattern_end]
        print(f"  Original slice shape: {orig_slice.shape}")
        print(f"  First row from sorted_data: {orig_slice[0] if len(orig_slice) > 0 else 'empty'}")
        print(f"  First row from pattern.data: {pattern.data[0] if len(pattern.data) > 0 else 'empty'}")
    
    return obj


def test_different_optimizers(obj):
    """Try different optimization approaches."""
    print(f"\n3. OPTIMIZATION COMPARISON")
    print("-" * 40)
    
    initial = obj.get_initial_parameters()
    print(f"Initial parameters shape: {initial.shape}")
    print(f"Initial objective: {obj.compute_objective(initial)}")
    print(f"Initial log-likelihood: {-obj.compute_objective(initial)/2}")
    
    # Try different optimizers
    methods = ['BFGS', 'L-BFGS-B', 'CG', 'Nelder-Mead']
    results = {}
    
    for method in methods:
        print(f"\n{method}:")
        try:
            result = minimize(
                fun=obj.compute_objective,
                x0=initial,
                method=method,
                options={'maxiter': 1000}
            )
            final_ll = -result.fun / 2
            print(f"  Converged: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Final log-likelihood: {final_ll:.9f}")
            results[method] = final_ll
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = None
    
    return results


def check_r_starting_values():
    """Check if R uses different starting values."""
    print(f"\n4. STARTING VALUES COMPARISON")
    print("-" * 40)
    
    data = datasets.apple
    obj = CPUObjectiveFP64(data)
    
    # Our starting values
    our_initial = obj.get_initial_parameters()
    n_vars = obj.n_vars
    
    our_mu = our_initial[:n_vars]
    our_log_diag = our_initial[n_vars:2*n_vars]
    our_off_diag = our_initial[2*n_vars:]
    
    print("Our starting values:")
    print(f"  μ: {our_mu}")
    print(f"  log(diag(Δ)): {our_log_diag}")
    print(f"  off-diag(Δ): {our_off_diag}")
    
    # Reconstruct initial Σ
    mu, Delta = obj.reconstruct_parameters(our_initial)
    Sigma = obj.compute_sigma_from_delta(Delta)
    
    print(f"\nInitial Σ:")
    print(Sigma)
    
    # Check condition number
    eigenvals = np.linalg.eigvalsh(Sigma)
    print(f"\nEigenvalues of initial Σ: {eigenvals}")
    print(f"Condition number: {np.max(eigenvals) / np.min(eigenvals):.2e}")
    
    return obj, our_initial


def test_manual_r_solution():
    """Test if R's known solution gives better likelihood."""
    print(f"\n5. TESTING R's SOLUTION")
    print("-" * 40)
    
    data = datasets.apple
    obj = CPUObjectiveFP64(data)
    
    # R's solution (from reference)
    r_mu = np.array([14.72222222, 45.0])
    r_sigma = np.array([
        [94.80065359, -85.09090909],
        [-85.09090909, 111.09090909]
    ])
    
    print("R's solution:")
    print(f"  μ: {r_mu}")
    print(f"  Σ:")
    print(r_sigma)
    
    # Convert to parameterization
    from pymvnmle._objectives.parameterizations import InverseCholeskyParameterization
    param = InverseCholeskyParameterization(obj.n_vars)
    r_theta = param.pack(r_mu, r_sigma)
    
    # Evaluate objective at R's solution
    obj_at_r = obj.compute_objective(r_theta)
    ll_at_r = -obj_at_r / 2
    
    print(f"\nObjective at R's solution: {obj_at_r}")
    print(f"Log-likelihood at R's solution: {ll_at_r}")
    print(f"Expected R log-likelihood: -74.217476")
    print(f"Difference: {ll_at_r - (-74.217476)}")
    
    return obj, r_theta


def trace_objective_computation(obj, theta):
    """Trace through objective computation step by step."""
    print(f"\n6. OBJECTIVE COMPUTATION TRACE")
    print("-" * 40)
    
    # Reconstruct parameters
    mu, Delta = obj.reconstruct_parameters(theta)
    Sigma = obj.compute_sigma_from_delta(Delta)
    
    print(f"Parameters:")
    print(f"  μ: {mu}")
    print(f"  Σ: {Sigma}")
    
    # Compute objective pattern by pattern
    total_obj = 0.0
    constant = obj.n_vars * np.log(2 * np.pi)
    
    for i, pattern in enumerate(obj.patterns):
        print(f"\nPattern {i}:")
        print(f"  N obs: {pattern.n_obs}")
        print(f"  Observed vars: {pattern.observed_indices}")
        
        # Extract submatrices
        mu_k = mu[pattern.observed_indices]
        Sigma_k = Sigma[np.ix_(pattern.observed_indices, pattern.observed_indices)]
        
        print(f"  μ_k: {mu_k}")
        print(f"  Σ_k: {Sigma_k}")
        
        # Check if Sigma_k is positive definite
        try:
            L_k = np.linalg.cholesky(Sigma_k)
            log_det = 2 * np.sum(np.log(np.diag(L_k)))
            print(f"  log|Σ_k|: {log_det}")
        except:
            print(f"  WARNING: Σ_k not positive definite!")
            continue
        
        # Compute contribution
        pattern_obj = pattern.n_obs * (constant + log_det)
        
        # Add quadratic form
        for j in range(pattern.n_obs):
            diff = pattern.data[j] - mu_k
            quad = diff @ np.linalg.solve(Sigma_k, diff)
            pattern_obj += quad
        
        print(f"  Pattern contribution: {pattern_obj}")
        total_obj += pattern_obj
    
    print(f"\nTotal objective: {total_obj}")
    print(f"Log-likelihood: {-total_obj/2}")
    
    return total_obj


if __name__ == "__main__":
    # Debug patterns
    obj = debug_patterns()
    
    # Try different optimizers
    results = test_different_optimizers(obj)
    
    # Check starting values
    obj, initial = check_r_starting_values()
    
    # Test R's solution
    obj, r_theta = test_manual_r_solution()
    
    # Trace computation
    trace_objective_computation(obj, r_theta)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"We're getting: -76.144")
    print(f"R gets: -74.217")
    print(f"Difference: {76.144 - 74.217:.3f}")
    print("\nPossible causes:")
    print("1. Different objective computation (check trace above)")
    print("2. Different parameterization handling")
    print("3. Bug in pattern data extraction")
    print("4. Different treatment of missing data in likelihood")