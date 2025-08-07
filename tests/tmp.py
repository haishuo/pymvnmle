#!/usr/bin/env python3
"""
Compare mlest() vs direct minimize() to understand timing differences.
"""

import numpy as np
import time
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymvnmle import mlest
from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
from pymvnmle._objectives.gpu_fp32_objective import GPUObjectiveFP32


def create_test_data(n_obs=2000, n_vars=15, n_patterns=3, seed=42):
    """Create data EXACTLY matching gpu_showcase test."""
    np.random.seed(seed)
    
    # True parameters
    A = np.random.randn(n_vars, n_vars)
    sigma_true = A @ A.T + np.eye(n_vars)
    mu_true = np.random.randn(n_vars) * 3
    
    # Generate complete data
    data = np.random.multivariate_normal(mu_true, sigma_true, n_obs)
    
    # Create patterns EXACTLY like gpu_showcase
    if n_patterns == 3:
        # Three patterns - matching gpu_showcase
        n_per_pattern = n_obs // 10
        indices1 = np.arange(n_per_pattern)
        indices2 = np.arange(n_per_pattern, 2 * n_per_pattern)
        data[indices1, -1] = np.nan  # Last var missing
        data[indices2, 0] = np.nan   # First var missing
        # Rest stays complete
    elif n_patterns == 2:
        # Two patterns
        n_missing = n_obs // 10
        indices = np.random.choice(n_obs, n_missing, replace=False)
        data[indices, -1] = np.nan
    # No additional random missing!
    
    return data, mu_true, sigma_true


def compare_cpu_methods(data):
    """Compare mlest() vs direct minimize() for CPU."""
    print(f"\n{'='*70}")
    print("CPU: mlest() vs minimize() comparison")
    print(f"{'='*70}")
    
    # Method 1: Using mlest (what gpu_showcase does)
    print("\n1. Using mlest() - matching gpu_showcase:")
    start = time.perf_counter()
    result_mlest = mlest(data, backend='cpu', max_iter=100, verbose=False)
    mlest_time = time.perf_counter() - start
    print(f"   Time: {mlest_time:.2f}s")
    print(f"   Iterations: {result_mlest.n_iter}")
    print(f"   Converged: {result_mlest.converged}")
    print(f"   Gradient norm: {result_mlest.gradient_norm:.2e}")
    
    # Method 2: Direct minimize with same settings
    print("\n2. Direct minimize() with gtol=1e-5:")
    cpu_obj = CPUObjectiveFP64(data)
    theta_cpu = cpu_obj.get_initial_parameters()
    
    start = time.perf_counter()
    result_direct = minimize(
        cpu_obj.compute_objective,
        theta_cpu,
        method='BFGS',
        jac=cpu_obj.compute_gradient,
        options={'maxiter': 100, 'gtol': 1e-5}
    )
    direct_time = time.perf_counter() - start
    print(f"   Time: {direct_time:.2f}s")
    print(f"   Iterations: {result_direct.nit}")
    print(f"   Converged: {result_direct.success}")
    print(f"   Gradient norm: {np.linalg.norm(result_direct.jac):.2e}")
    
    # Method 3: Direct minimize with looser tolerance
    print("\n3. Direct minimize() with gtol=1e-4:")
    theta_cpu = cpu_obj.get_initial_parameters()
    
    start = time.perf_counter()
    result_loose = minimize(
        cpu_obj.compute_objective,
        theta_cpu,
        method='BFGS',
        jac=cpu_obj.compute_gradient,
        options={'maxiter': 100, 'gtol': 1e-4}
    )
    loose_time = time.perf_counter() - start
    print(f"   Time: {loose_time:.2f}s")
    print(f"   Iterations: {result_loose.nit}")
    print(f"   Converged: {result_loose.success}")
    print(f"   Gradient norm: {np.linalg.norm(result_loose.jac):.2e}")
    
    return result_mlest


def test_gpu_with_reference(data, cpu_result):
    """Test GPU with different tolerances against CPU reference."""
    print(f"\n{'='*70}")
    print("GPU optimization with different tolerances")
    print(f"{'='*70}")
    
    mu_ref = cpu_result.muhat
    sigma_ref = cpu_result.sigmahat
    
    gpu_obj = GPUObjectiveFP32(data)
    
    tolerances = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    
    print(f"\n{'gtol':<10} {'iter':<6} {'time(s)':<8} {'μ_err':<10} {'Σ_err':<10} {'converged':<10}")
    print("-" * 60)
    
    for tol in tolerances:
        theta_gpu = gpu_obj.get_initial_parameters()
        
        start = time.perf_counter()
        result = minimize(
            gpu_obj.compute_objective,
            theta_gpu,
            method='BFGS',
            jac=gpu_obj.compute_gradient,
            options={'maxiter': 100, 'gtol': tol}
        )
        gpu_time = time.perf_counter() - start
        
        mu_gpu, sigma_gpu, _ = gpu_obj.extract_parameters(result.x)
        
        mu_err = np.max(np.abs(mu_gpu - mu_ref))
        sigma_err = np.max(np.abs(sigma_gpu - sigma_ref))
        
        print(f"{tol:<10.0e} {result.nit:<6d} {gpu_time:<8.3f} {mu_err:<10.2e} {sigma_err:<10.2e} {str(result.success):<10}")


def test_gpu_via_mlest(data, cpu_result):
    """Test GPU using mlest wrapper."""
    print(f"\n{'='*70}")
    print("GPU via mlest() wrapper")
    print(f"{'='*70}")
    
    mu_ref = cpu_result.muhat
    sigma_ref = cpu_result.sigmahat
    
    # Default mlest
    print("\n1. Default mlest (tol=1e-5):")
    start = time.perf_counter()
    result1 = mlest(data, backend='gpu', max_iter=100, verbose=False)
    print(f"   Time: {time.perf_counter() - start:.2f}s")
    print(f"   Iterations: {result1.n_iter}")
    print(f"   μ error: {np.max(np.abs(result1.muhat - mu_ref)):.2e}")
    print(f"   Σ error: {np.max(np.abs(result1.sigmahat - sigma_ref)):.2e}")
    
    # With looser tolerance
    print("\n2. mlest with tol=1e-4:")
    start = time.perf_counter()
    result2 = mlest(data, backend='gpu', max_iter=100, tol=1e-4, verbose=False)
    print(f"   Time: {time.perf_counter() - start:.2f}s")
    print(f"   Iterations: {result2.n_iter}")
    print(f"   μ error: {np.max(np.abs(result2.muhat - mu_ref)):.2e}")
    print(f"   Σ error: {np.max(np.abs(result2.sigmahat - sigma_ref)):.2e}")


def main():
    """Run comparison tests."""
    
    # Create data matching gpu_showcase
    print("Creating test data (2000×15, 3 patterns)...")
    data, _, _ = create_test_data(2000, 15, 3)
    print(f"Data shape: {data.shape}")
    print(f"Missing: {np.sum(np.isnan(data))}/{data.size} ({100*np.sum(np.isnan(data))/data.size:.1f}%)")
    
    # Compare CPU methods
    cpu_result = compare_cpu_methods(data)
    
    # Test GPU
    test_gpu_with_reference(data, cpu_result)
    test_gpu_via_mlest(data, cpu_result)
    
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    print("""
1. Check if mlest() uses different default tolerance than 1e-5
2. GPU needs gtol=1e-4 or 5e-5 for good accuracy with FP32
3. The 'max_iter' in gpu_showcase might be less than 100
""")


if __name__ == "__main__":
    main()