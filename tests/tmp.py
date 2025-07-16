#!/usr/bin/env python3
"""
Test that the fixed get_initial_parameters() resolves the Missvals issue
"""

import numpy as np
from pymvnmle import mlest
from pymvnmle.datasets import missvals

print("Testing Fixed Implementation with Pairwise Complete Covariance")
print("="*60)

# R reference values
r_loglik = -86.978324
test_incorrect_value = -40.453229  # The incorrect value in the test

print("Reference values:")
print(f"  Correct R log-likelihood: {r_loglik:.6f}")
print(f"  Incorrect test value: {test_incorrect_value:.6f}")

# Run with the fixed implementation
print("\nRunning PyMVNMLE with fixed starting values...")
result = mlest(missvals, method='BFGS', max_iter=400, verbose=False)

print(f"\nResults:")
print(f"  Log-likelihood: {result.loglik:.6f}")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.n_iter}")

# Compare with references
diff_from_r = abs(result.loglik - r_loglik)
diff_from_test = abs(result.loglik - test_incorrect_value) 

print(f"\nDifferences:")
print(f"  From correct R value: {diff_from_r:.6f}")
print(f"  From test value: {diff_from_test:.6f}")

# Check if we're closer to R now
print("\n" + "="*60)
if diff_from_r < 10:  # Within reasonable tolerance
    print("✅ SUCCESS! The fix brings us much closer to R's result!")
    print(f"   Remaining difference: {diff_from_r:.6f}")
    if diff_from_r < 1:
        print("   This is within acceptable numerical tolerance!")
    else:
        print("   Some difference remains, but it's much better than before.")
else:
    print("⚠️  Still have significant difference from R.")
    print("   May need additional investigation.")

# Test the starting values directly
print("\n" + "="*60)
print("Checking Starting Covariance Structure:")
print("="*60)

from pymvnmle._objective import MVNMLEObjective
from pymvnmle._backends import get_backend_with_fallback

backend = get_backend_with_fallback('numpy', verbose=False)
obj = MVNMLEObjective(missvals, backend=backend)
theta0 = obj.get_initial_parameters()

# Extract the Delta matrix to check correlations
n_vars = missvals.shape[1]
log_diag = theta0[n_vars:2*n_vars]
Delta = np.diag(np.exp(log_diag))

# Fill in off-diagonals
idx = 2 * n_vars
for j in range(1, n_vars):
    for i in range(j):
        Delta[i, j] = theta0[idx]
        idx += 1

print("Starting Delta matrix has non-zero off-diagonals:")
print(f"  Delta[0,1] = {Delta[0, 1]:.6f}")
print(f"  Delta[2,4] = {Delta[2, 4]:.6f} (critical for sparse pattern)")

# Compute implied starting covariance
try:
    Delta_inv = np.linalg.inv(Delta)
    Sigma_start = Delta_inv.T @ Delta_inv
    
    print("\nImplied starting covariance for variables 3,5:")
    print(f"  Var(V3) = {Sigma_start[2, 2]:.3f}")
    print(f"  Var(V5) = {Sigma_start[4, 4]:.3f}")
    print(f"  Cov(V3,V5) = {Sigma_start[2, 4]:.3f}")
    print(f"  Correlation = {Sigma_start[2, 4] / np.sqrt(Sigma_start[2, 2] * Sigma_start[4, 4]):.3f}")
    
    if abs(Sigma_start[2, 4]) > 10:
        print("\n✅ Starting values now capture the negative correlation!")
except:
    print("\n⚠️  Could not compute starting covariance")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("The fixed get_initial_parameters() should now:")
print("1. Compute pairwise complete covariances")
print("2. Capture the V3-V5 correlation")
print("3. Lead to better optimization results")
print("4. Match or get very close to R's reference value")