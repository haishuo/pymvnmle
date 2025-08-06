#!/usr/bin/env python3
"""
Debug test to figure out why we differ from R on the Apple dataset.
This will print detailed information to help identify the discrepancy.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import datasets, mlest
from pymvnmle._objectives import CPUObjectiveFP64

def debug_apple():
    """Debug the Apple dataset to find the R discrepancy."""
    
    print("=" * 70)
    print("DEBUG: Apple Dataset Analysis")
    print("=" * 70)
    
    # Load Apple dataset
    data = datasets.apple
    print(f"\n1. RAW DATA:")
    print(f"   Shape: {data.shape}")
    print(f"   Missing values: {np.sum(np.isnan(data))}")
    print(f"   First 3 rows:\n{data[:3]}")
    print(f"   Last 3 rows:\n{data[-3:]}")
    
    # Create objective to inspect preprocessing
    obj = CPUObjectiveFP64(data)
    
    print(f"\n2. PREPROCESSING RESULTS:")
    print(f"   Number of patterns: {obj.n_patterns}")
    print(f"   Number of parameters: {obj.n_params}")
    
    # Inspect patterns
    print(f"\n3. PATTERN DETAILS:")
    for i, pattern in enumerate(obj.patterns):
        print(f"   Pattern {i+1}:")
        print(f"     - Observed indices: {pattern.observed_indices}")
        print(f"     - Number of observations: {pattern.n_obs}")
        print(f"     - Data shape: {pattern.data.shape}")
        if pattern.data.shape[0] > 0:
            print(f"     - First observation: {pattern.data[0]}")
    
    # Get initial parameters
    theta_init = obj.get_initial_parameters()
    print(f"\n4. INITIAL PARAMETERS:")
    print(f"   Length: {len(theta_init)}")
    print(f"   Mean parameters (μ): {theta_init[:obj.n_vars]}")
    print(f"   Log-diagonal (log(Δ)): {theta_init[obj.n_vars:2*obj.n_vars]}")
    print(f"   Off-diagonal (first 5): {theta_init[2*obj.n_vars:2*obj.n_vars+5]}")
    
    # Compute initial objective
    obj_init = obj.compute_objective(theta_init)
    print(f"\n5. INITIAL OBJECTIVE:")
    print(f"   f(θ₀) = {obj_init:.12f}")
    print(f"   Initial log-likelihood = {-obj_init/2:.12f}")
    
    # Extract initial mu and sigma
    mu_init, sigma_init, loglik_init = obj.extract_parameters(theta_init)
    print(f"\n6. INITIAL ESTIMATES:")
    print(f"   μ = {mu_init}")
    print(f"   Σ = \n{sigma_init}")
    print(f"   Log-likelihood = {loglik_init:.12f}")
    
    # Check sample statistics
    print(f"\n7. SAMPLE STATISTICS:")
    print(f"   Sample mean: {obj.sample_mean}")
    print(f"   Sample covariance diagonal: {np.diag(obj.sample_cov)}")
    
    # Run optimization to get final result
    print(f"\n8. RUNNING OPTIMIZATION...")
    result = mlest(data, backend='cpu', max_iter=100, verbose=False)
    
    print(f"\n9. FINAL RESULTS:")
    print(f"   Converged: {result.converged}")
    print(f"   Iterations: {result.n_iter}")
    print(f"   Final μ: {result.muhat}")
    print(f"   Final Σ diagonal: {np.diag(result.sigmahat)}")
    print(f"   Final log-likelihood: {result.loglik:.12f}")
    
    # R reference values
    print(f"\n10. R REFERENCE (mvnmle):")
    print(f"   μ = [14.722266, 49.333248]")
    print(f"   Σ[1,1] = 89.534150")
    print(f"   Σ[2,2] = 114.694700")
    print(f"   Σ[1,2] = -90.696532")
    print(f"   Log-likelihood = -74.217476")
    
    # Compute differences
    r_loglik = -74.217476
    our_loglik = result.loglik
    diff = our_loglik - r_loglik
    
    print(f"\n11. COMPARISON:")
    print(f"   Our log-likelihood: {our_loglik:.12f}")
    print(f"   R log-likelihood: {r_loglik:.12f}")
    print(f"   Difference: {diff:.12f}")
    print(f"   Relative difference: {100*abs(diff/r_loglik):.2f}%")
    
    # Check if patterns might be different
    print(f"\n12. PATTERN ANALYSIS:")
    print(f"   Complete cases pattern size: {obj.patterns[0].n_obs if obj.patterns else 'N/A'}")
    print(f"   Missing pattern size: {obj.patterns[1].n_obs if len(obj.patterns) > 1 else 'N/A'}")
    print(f"   Expected from R: Complete=12, Missing=6")
    
    # Manual check of pattern extraction
    is_complete = ~np.any(np.isnan(data), axis=1)
    n_complete = np.sum(is_complete)
    n_missing = len(data) - n_complete
    print(f"\n13. MANUAL PATTERN CHECK:")
    print(f"   Complete observations: {n_complete}")
    print(f"   Observations with missing: {n_missing}")
    
    print("\n" + "=" * 70)
    print("END DEBUG")
    print("=" * 70)

if __name__ == "__main__":
    debug_apple()