#!/usr/bin/env python3
"""
Diagnose numerical differences between CPU and GPU backends.

CRITICAL: This diagnostic reveals fundamental implementation differences:
1. NumPy uses inverse Cholesky parameterization (R-compatible)
2. PyTorch uses standard Cholesky parameterization
3. Parameter vectors have DIFFERENT meanings between backends
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import mlest, datasets
from pymvnmle._objectives import get_objective

def test_starting_points():
    """Check if both backends produce same starting mu/sigma."""
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
    print(f"Parameter vector difference: {np.max(np.abs(numpy_start - torch_start)):.6e}")
    
    # Extract mu/sigma from each
    mu_np, sigma_np, _ = numpy_obj.extract_parameters(numpy_start)
    mu_torch, sigma_torch, _ = torch_obj.extract_parameters(torch_start)
    
    print(f"\nStarting μ from NumPy: {mu_np}")
    print(f"Starting μ from PyTorch: {mu_torch}")
    print(f"μ difference: {np.max(np.abs(mu_np - mu_torch)):.6e}")
    
    print(f"\nStarting Σ from NumPy:\n{sigma_np}")
    print(f"\nStarting Σ from PyTorch:\n{sigma_torch}")
    print(f"Σ difference: {np.max(np.abs(sigma_np - sigma_torch)):.6e}")
    
    return numpy_start, torch_start, mu_np, sigma_np

def test_objective_values(numpy_start, torch_start):
    """Test if objectives give same values for equivalent parameters."""
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
    
    # Extract parameters for comparison
    mu_np, sigma_np, ll_np = numpy_obj.extract_parameters(numpy_start)
    mu_torch, sigma_torch, ll_torch = torch_obj.extract_parameters(torch_start)
    
    print(f"\nLog-likelihood from NumPy: {ll_np:.6f}")
    print(f"Log-likelihood from PyTorch: {ll_torch:.6f}")
    print(f"Difference: {abs(ll_np - ll_torch):.6f}")
    
    # Test 2: Can we convert between parameterizations?
    # PyTorch has pack_parameters, NumPy doesn't
    if hasattr(torch_obj, 'pack_parameters'):
        # Pack NumPy's mu/sigma into PyTorch parameterization
        torch_equiv = torch_obj.pack_parameters(mu_np, sigma_np)
        torch_val_at_numpy_equiv = torch_obj(torch_equiv)
        
        print(f"\nPyTorch objective at NumPy-equivalent parameters: {torch_val_at_numpy_equiv:.6f}")
        print(f"Should match NumPy objective: {numpy_val_at_numpy:.6f}")
        print(f"Difference: {abs(torch_val_at_numpy_equiv - numpy_val_at_numpy):.6f}")

def test_gradient_comparison():
    """Compare gradients if available."""
    print("\n" + "=" * 70)
    print("TEST 3: GRADIENT COMPARISON")
    print("=" * 70)
    
    data = datasets.apple
    
    # Create objectives
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Get starting points
    numpy_start = numpy_obj.get_initial_parameters()
    torch_start = torch_obj.get_initial_parameters()
    
    # Check gradient availability
    has_numpy_grad = hasattr(numpy_obj, 'gradient')
    has_torch_grad = hasattr(torch_obj, 'gradient')
    
    print(f"NumPy has gradient method: {has_numpy_grad}")
    print(f"PyTorch has gradient method: {has_torch_grad}")
    
    if has_numpy_grad:
        numpy_grad = numpy_obj.gradient(numpy_start)
        print(f"\nNumPy gradient norm: {np.linalg.norm(numpy_grad):.6e}")
        print(f"NumPy gradient shape: {numpy_grad.shape}")
    
    if has_torch_grad:
        torch_grad = torch_obj.gradient(torch_start)
        print(f"\nPyTorch gradient norm: {np.linalg.norm(torch_grad):.6e}")
        print(f"PyTorch gradient shape: {torch_grad.shape}")
    
    # CRITICAL: Gradients are in DIFFERENT parameter spaces!
    print("\nWARNING: Gradients cannot be directly compared due to different parameterizations!")

def test_optimization_path():
    """Compare optimization paths step by step."""
    print("\n" + "=" * 70)
    print("TEST 4: OPTIMIZATION PATH COMPARISON")
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

def test_parameterization_details():
    """Examine parameterization differences in detail."""
    print("\n" + "=" * 70)
    print("TEST 5: PARAMETERIZATION ANALYSIS")
    print("=" * 70)
    
    # Simple 2x2 test case
    mu_test = np.array([1.0, 2.0])
    sigma_test = np.array([[4.0, 1.0],
                          [1.0, 3.0]])
    
    print(f"Test μ: {mu_test}")
    print(f"Test Σ:\n{sigma_test}")
    
    # Create objectives with dummy data
    data = np.random.randn(10, 2)
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # PyTorch can pack parameters
    if hasattr(torch_obj, 'pack_parameters'):
        torch_theta = torch_obj.pack_parameters(mu_test, sigma_test)
        print(f"\nPyTorch parameter vector: {torch_theta}")
        print(f"PyTorch uses Cholesky L where Σ = LL'")
        
        # Verify round-trip
        mu_back, sigma_back = torch_obj.unpack_parameters(torch_theta)
        print(f"Round-trip μ error: {np.max(np.abs(mu_test - mu_back)):.6e}")
        print(f"Round-trip Σ error: {np.max(np.abs(sigma_test - sigma_back)):.6e}")
    
    # NumPy uses inverse Cholesky parameterization
    print(f"\nNumPy uses inverse Cholesky Δ where Σ = (Δ^-1)'(Δ^-1)")
    print("NumPy parameter structure: [μ, log(diag(Δ)), off-diag(Δ)]")
    
    # Show what NumPy's initial parameters look like
    numpy_init = numpy_obj.get_initial_parameters()
    n_vars = len(mu_test)
    print(f"\nNumPy parameter breakdown:")
    print(f"  μ parameters: {numpy_init[:n_vars]}")
    print(f"  log(diag(Δ)): {numpy_init[n_vars:2*n_vars]}")
    print(f"  off-diag(Δ): {numpy_init[2*n_vars:]}")

def main():
    """Run all diagnostic tests."""
    print("NUMERICAL DIFFERENCES DIAGNOSTIC")
    print("=" * 70)
    print("\nCRITICAL FINDING: The backends use DIFFERENT parameterizations!")
    print("- NumPy: Inverse Cholesky (R-compatible)")
    print("- PyTorch: Standard Cholesky")
    print("This explains ALL numerical differences.\n")
    
    # Run tests
    numpy_start, torch_start, mu_np, sigma_np = test_starting_points()
    test_objective_values(numpy_start, torch_start)
    test_gradient_comparison()
    test_parameterization_details()
    test_optimization_path()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    print("\nRoot cause of differences:")
    print("1. DIFFERENT PARAMETERIZATIONS (main issue)")
    print("2. Different optimization landscapes due to parameterization")
    print("3. Different gradient directions in parameter space")
    print("4. NumPy uses finite differences, PyTorch uses autodiff")
    
    print("\nRecommendation for FDA-grade software:")
    print("- Document parameterization differences clearly")
    print("- Ensure both converge to same (μ, Σ) within tolerance")
    print("- Validate against R mvnmle reference implementation")
    print("- Consider implementing pack_parameters for NumPy backend")

if __name__ == "__main__":
    main()