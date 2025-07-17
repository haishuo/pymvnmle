#!/usr/bin/env python3
"""
Simple verification that PyTorch backend now matches NumPy backend.

Tests:
1. Same starting (μ, Σ)
2. Same log-likelihood for given (μ, Σ)
3. Same objective function value
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import datasets
from pymvnmle._objectives import get_objective


def main():
    print("PYTORCH FIX VERIFICATION")
    print("=" * 70)
    
    # Use apple data
    data = datasets.apple
    print(f"Testing with apple data: {data.shape}")
    
    # Create objectives
    print("\nCreating objective functions...")
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Test 1: Initial parameters
    print("\n" + "-" * 50)
    print("TEST 1: Initial Parameters")
    print("-" * 50)
    
    numpy_theta = numpy_obj.get_initial_parameters()
    torch_theta = torch_obj.get_initial_parameters()
    
    # Extract mu and sigma
    mu_np, sigma_np, _ = numpy_obj.extract_parameters(numpy_theta)
    mu_torch, sigma_torch, _ = torch_obj.extract_parameters(torch_theta)
    
    print(f"NumPy μ:  {mu_np}")
    print(f"PyTorch μ: {mu_torch}")
    print(f"μ difference: {np.max(np.abs(mu_np - mu_torch)):.2e}")
    
    print(f"\nNumPy Σ[0,0]:  {sigma_np[0,0]:.6f}")
    print(f"PyTorch Σ[0,0]: {sigma_torch[0,0]:.6f}")
    print(f"Σ max difference: {np.max(np.abs(sigma_np - sigma_torch)):.2e}")
    
    # Test 2: Objective values
    print("\n" + "-" * 50)
    print("TEST 2: Objective Function Values")
    print("-" * 50)
    
    # Compute objective at their respective starting points
    obj_np = numpy_obj(numpy_theta)
    obj_torch = torch_obj(torch_theta)
    
    print(f"NumPy objective:  {obj_np:.6f}")
    print(f"PyTorch objective: {obj_torch:.6f}")
    print(f"Should be close if starting (μ,Σ) are the same")
    
    # Test 3: Log-likelihood values
    print("\n" + "-" * 50)
    print("TEST 3: Log-Likelihood Values")
    print("-" * 50)
    
    _, _, ll_np = numpy_obj.extract_parameters(numpy_theta)
    _, _, ll_torch = torch_obj.extract_parameters(torch_theta)
    
    print(f"NumPy log-likelihood:  {ll_np:.6f}")
    print(f"PyTorch log-likelihood: {ll_torch:.6f}")
    print(f"Difference: {abs(ll_np - ll_torch):.2e}")
    
    # Test 4: Cross-check - pack NumPy's (μ,Σ) into PyTorch format
    print("\n" + "-" * 50)
    print("TEST 4: Cross-Parameterization Check")
    print("-" * 50)
    
    # Pack NumPy's estimates into PyTorch parameterization
    torch_theta_from_numpy = torch_obj.pack_parameters(mu_np, sigma_np)
    
    # Compute objective and log-likelihood
    obj_torch_at_numpy = torch_obj(torch_theta_from_numpy)
    mu_check, sigma_check, ll_check = torch_obj.extract_parameters(torch_theta_from_numpy)
    
    print(f"PyTorch objective at NumPy's (μ,Σ): {obj_torch_at_numpy:.6f}")
    print(f"Should match NumPy objective: {obj_np:.6f}")
    print(f"Difference: {abs(obj_torch_at_numpy - obj_np):.2e}")
    
    print(f"\nLog-likelihood check: {ll_check:.6f}")
    print(f"Should match NumPy: {ll_np:.6f}")
    print(f"Difference: {abs(ll_check - ll_np):.2e}")
    
    # Test 5: Understanding the objective function
    print("\n" + "-" * 50)
    print("TEST 5: Understanding Objective Convention")
    print("-" * 50)
    print("The objective function returns -2 log L (not -log L)")
    print(f"NumPy objective = -2 log L = {obj_np:.6f}")
    print(f"NumPy log-likelihood = {ll_np:.6f}")
    print(f"Verify: -objective/2 = {-obj_np/2:.6f} (should match log-likelihood)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    mu_ok = np.max(np.abs(mu_np - mu_torch)) < 1e-10
    sigma_ok = np.max(np.abs(sigma_np - sigma_torch)) < 1e-10
    ll_ok = abs(ll_np - ll_torch) < 1e-6
    obj_ok = abs(obj_torch_at_numpy - obj_np) < 1e-6
    
    print(f"Starting μ match: {'✅' if mu_ok else '❌'}")
    print(f"Starting Σ match: {'✅' if sigma_ok else '❌'}")
    print(f"Log-likelihood match: {'✅' if ll_ok else '❌'}")
    print(f"Objective match: {'✅' if obj_ok else '❌'}")
    
    if all([mu_ok, sigma_ok, ll_ok, obj_ok]):
        print("\n✅ ALL TESTS PASSED! PyTorch backend is fixed.")
        print("Both backends now:")
        print("1. Start from same (μ, Σ)")
        print("2. Compute same log-likelihood")
        print("3. Should converge to same estimates")
    else:
        print("\n❌ TESTS FAILED! Issues remain.")


if __name__ == "__main__":
    main()