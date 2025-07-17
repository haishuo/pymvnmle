#!/usr/bin/env python3
"""
Diagnose numerical differences between CPU and GPU backends
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import mlest, datasets
from pymvnmle._objectives import get_objective

def test_starting_points():
    """Check if both backends use same starting points."""
    print("=" * 70)
    print("TEST 1: STARTING POINTS")
    print("=" * 70)
    
    # Use same data
    data = datasets.apple
    
    # Create objectives
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Get initial parameters
    numpy_start = numpy_obj.get_initial_parameters()
    torch_start = torch_obj.get_initial_parameters()
    
    print(f"\nNumPy starting params: {numpy_start}")
    print(f"PyTorch starting params: {torch_start}")
    print(f"Difference: {np.max(np.abs(numpy_start - torch_start)):.6e}")
    
    # Check if they unpack to same mu/sigma
    mu_np, sigma_np = numpy_obj.unpack_parameters(numpy_start)
    mu_torch, sigma_torch = torch_obj.unpack_parameters(torch_start)
    
    print(f"\nStarting μ difference: {np.max(np.abs(mu_np - mu_torch)):.6e}")
    print(f"Starting Σ difference: {np.max(np.abs(sigma_np - sigma_torch)):.6e}")
    
    return numpy_start, torch_start

def test_objective_values(numpy_start, torch_start):
    """Test if objectives give same values for same parameters."""
    print("\n" + "=" * 70)
    print("TEST 2: OBJECTIVE FUNCTION VALUES")
    print("=" * 70)
    
    data = datasets.apple
    
    # Create objectives
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Test 1: Each objective at its own starting point
    numpy_val_at_numpy = numpy_obj(numpy_start)
    torch_val_at_torch = torch_obj(torch_start)
    
    print(f"\nNumPy objective at NumPy start: {numpy_val_at_numpy:.6f}")
    print(f"PyTorch objective at PyTorch start: {torch_val_at_torch:.6f}")
    print(f"Difference: {abs(numpy_val_at_numpy - torch_val_at_torch):.6f}")
    
    # Test 2: Convert parameters between parameterizations
    # Get mu/sigma from numpy starting point
    mu_np, sigma_np = numpy_obj.unpack_parameters(numpy_start)
    
    # Pack into torch parameterization
    torch_equiv = torch_obj.pack_parameters(mu_np, sigma_np)
    
    # Evaluate
    torch_val_at_numpy_equiv = torch_obj(torch_equiv)
    
    print(f"\nPyTorch objective at NumPy equivalent: {torch_val_at_numpy_equiv:.6f}")
    print(f"Should match NumPy objective: {numpy_val_at_numpy:.6f}")
    print(f"Difference: {abs(torch_val_at_numpy_equiv - numpy_val_at_numpy):.6f}")
    
    # The log-likelihoods should match!
    _, _, ll_numpy = numpy_obj.extract_parameters(numpy_start)
    _, _, ll_torch = torch_obj.extract_parameters(torch_equiv)
    
    print(f"\nLog-likelihood comparison:")
    print(f"NumPy: {ll_numpy:.6f}")
    print(f"PyTorch: {ll_torch:.6f}")
    print(f"Difference: {abs(ll_numpy - ll_torch):.6f}")

def test_optimization_path():
    """Compare optimization paths step by step."""
    print("\n" + "=" * 70)
    print("TEST 3: OPTIMIZATION PATH COMPARISON")
    print("=" * 70)
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(50, 5)
    data[np.random.rand(50, 5) < 0.1] = np.nan
    
    print("Testing on synthetic data (50×5)")
    
    # Run both optimizations with same settings
    print("\n1. CPU/NumPy optimization:")
    result_cpu = mlest(data, backend='cpu', method='BFGS', max_iter=50, verbose=True)
    
    print("\n2. GPU/PyTorch optimization:")
    result_gpu = mlest(data, backend='gpu', max_iter=50, verbose=True)
    
    # Compare results
    print("\n" + "-" * 50)
    print("FINAL RESULTS COMPARISON:")
    print("-" * 50)
    
    mu_diff = np.max(np.abs(result_cpu.muhat - result_gpu.muhat))
    sigma_diff = np.max(np.abs(result_cpu.sigmahat - result_gpu.sigmahat))
    ll_diff = abs(result_cpu.loglik - result_gpu.loglik)
    
    print(f"μ difference: {mu_diff:.6e}")
    print(f"Σ difference: {sigma_diff:.6e}")
    print(f"Log-likelihood difference: {ll_diff:.6e}")
    
    # Show actual values
    print(f"\nCPU log-likelihood: {result_cpu.loglik:.6f}")
    print(f"GPU log-likelihood: {result_gpu.loglik:.6f}")
    
    print(f"\nCPU converged: {result_cpu.converged} in {result_cpu.n_iter} iterations")
    print(f"GPU converged: {result_gpu.converged} in {result_gpu.n_iter} iterations")

def test_gradient_comparison():
    """Compare gradients at same point."""
    print("\n" + "=" * 70)
    print("TEST 4: GRADIENT COMPARISON")
    print("=" * 70)
    
    data = datasets.apple
    
    # Create objectives
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Get same starting point in both parameterizations
    mu_start = np.array([10.0, 30.0])  # Reasonable values for apple data
    sigma_start = np.array([[20.0, 5.0], [5.0, 40.0]])
    
    numpy_theta = numpy_obj.pack_parameters(mu_start, sigma_start)
    torch_theta = torch_obj.pack_parameters(mu_start, sigma_start)
    
    print(f"Testing at μ = {mu_start}")
    print(f"Testing at Σ =\n{sigma_start}")
    
    # Compute gradients
    if hasattr(numpy_obj, 'gradient'):
        numpy_grad = numpy_obj.gradient(numpy_theta)
        print(f"\nNumPy gradient norm: {np.linalg.norm(numpy_grad):.6e}")
    
    torch_grad = torch_obj.gradient(torch_theta)
    print(f"PyTorch gradient norm: {np.linalg.norm(torch_grad):.6e}")
    
    # Check if gradients point in same direction (in parameter space)
    # This is tricky because different parameterizations!

def test_parameterization_roundtrip():
    """Test if pack/unpack is consistent."""
    print("\n" + "=" * 70)
    print("TEST 5: PARAMETERIZATION ROUND-TRIP")
    print("=" * 70)
    
    # Test values
    mu_test = np.array([1.0, 2.0, 3.0])
    sigma_test = np.array([[4.0, 0.5, 0.2],
                          [0.5, 5.0, 0.3],
                          [0.2, 0.3, 6.0]])
    
    # Create objectives
    data = np.random.randn(10, 3)  # Dummy data
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Test NumPy round-trip
    numpy_theta = numpy_obj.pack_parameters(mu_test, sigma_test)
    mu_np, sigma_np = numpy_obj.unpack_parameters(numpy_theta)
    
    print("NumPy round-trip:")
    print(f"  μ error: {np.max(np.abs(mu_test - mu_np)):.6e}")
    print(f"  Σ error: {np.max(np.abs(sigma_test - sigma_np)):.6e}")
    
    # Test PyTorch round-trip
    torch_theta = torch_obj.pack_parameters(mu_test, sigma_test)
    mu_torch, sigma_torch = torch_obj.unpack_parameters(torch_theta)
    
    print("\nPyTorch round-trip:")
    print(f"  μ error: {np.max(np.abs(mu_test - mu_torch)):.6e}")
    print(f"  Σ error: {np.max(np.abs(sigma_test - sigma_torch)):.6e}")
    
    # Test cross-parameterization
    print("\nCross-parameterization test:")
    print(f"  NumPy params shape: {numpy_theta.shape}")
    print(f"  PyTorch params shape: {torch_theta.shape}")
    print(f"  Same mu/sigma should give same likelihood!")

def main():
    """Run all diagnostic tests."""
    print("NUMERICAL DIFFERENCES DIAGNOSTIC")
    print("=" * 70)
    
    # Run tests
    numpy_start, torch_start = test_starting_points()
    test_objective_values(numpy_start, torch_start)
    test_parameterization_roundtrip()
    test_gradient_comparison()
    test_optimization_path()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    print("\nLikely causes of differences:")
    print("1. Different parameterizations (Cholesky vs Inverse Cholesky)")
    print("2. Different optimization methods (Newton-CG vs BFGS)")
    print("3. Different starting points in parameter space")
    print("4. Possible bug in parameter conversion")

if __name__ == "__main__":
    main()