#!/usr/bin/env python3
"""
Simple direct comparison of CPU vs GPU optimization.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import mlest
from scipy.optimize import minimize


def simple_convergence_test():
    """Simple test to check convergence behavior."""
    
    print("=" * 70)
    print("SIMPLE CONVERGENCE TEST")
    print("=" * 70)
    
    # Generate simple test data
    np.random.seed(42)
    n_obs, n_vars = 500, 5
    
    # True parameters
    true_mu = np.arange(1, n_vars + 1, dtype=float)
    true_sigma = np.eye(n_vars) + 0.3  # Simple covariance
    
    # Generate complete data first (no missing)
    from scipy.stats import multivariate_normal
    data = multivariate_normal.rvs(mean=true_mu, cov=true_sigma, size=n_obs)
    
    print(f"\nTest data: {n_obs}×{n_vars} (COMPLETE - no missing)")
    print(f"True μ: {true_mu}")
    print(f"True Σ diagonal: {np.diag(true_sigma)}")
    
    # Test 1: Complete data (should give very similar results)
    print("\n" + "=" * 70)
    print("TEST 1: COMPLETE DATA")
    print("=" * 70)
    
    print("\nCPU optimization...")
    cpu_result = mlest(data, method='BFGS', backend='cpu', verbose=False)
    
    print(f"CPU converged: {cpu_result.converged} in {cpu_result.n_iter} iterations")
    print(f"CPU μ[0:3]: {cpu_result.muhat[:3]}")
    print(f"CPU Σ[0,0]: {cpu_result.sigmahat[0,0]:.6f}")
    print(f"CPU log-lik: {cpu_result.log_likelihood:.6f}")
    
    print("\nGPU optimization...")
    gpu_result = mlest(data, method='BFGS', backend='gpu', verbose=False)
    
    print(f"GPU converged: {gpu_result.converged} in {gpu_result.n_iter} iterations")
    print(f"GPU μ[0:3]: {gpu_result.muhat[:3]}")
    print(f"GPU Σ[0,0]: {gpu_result.sigmahat[0,0]:.6f}")
    print(f"GPU log-lik: {gpu_result.log_likelihood:.6f}")
    
    print(f"\nDifferences (complete data):")
    print(f"  Max μ diff: {np.max(np.abs(cpu_result.muhat - gpu_result.muhat)):.2e}")
    print(f"  Max Σ diff: {np.max(np.abs(cpu_result.sigmahat - gpu_result.sigmahat)):.2e}")
    
    # Test 2: Add 5% missing
    print("\n" + "=" * 70)
    print("TEST 2: WITH 5% MISSING DATA")
    print("=" * 70)
    
    data_missing = data.copy()
    mask = np.random.random(data.shape) < 0.05
    data_missing[mask] = np.nan
    print(f"\nMissing: {np.sum(mask)}/{data.size} ({100*np.sum(mask)/data.size:.1f}%)")
    
    print("\nCPU optimization...")
    cpu_result2 = mlest(data_missing, method='BFGS', backend='cpu', verbose=False)
    
    print(f"CPU converged: {cpu_result2.converged} in {cpu_result2.n_iter} iterations")
    print(f"CPU μ[0:3]: {cpu_result2.muhat[:3]}")
    print(f"CPU Σ[0,0]: {cpu_result2.sigmahat[0,0]:.6f}")
    print(f"CPU log-lik: {cpu_result2.log_likelihood:.6f}")
    
    print("\nGPU optimization...")
    gpu_result2 = mlest(data_missing, method='BFGS', backend='gpu', verbose=False)
    
    print(f"GPU converged: {gpu_result2.converged} in {gpu_result2.n_iter} iterations")
    print(f"GPU μ[0:3]: {gpu_result2.muhat[:3]}")
    print(f"GPU Σ[0,0]: {gpu_result2.sigmahat[0,0]:.6f}")
    print(f"GPU log-lik: {gpu_result2.log_likelihood:.6f}")
    
    print(f"\nDifferences (5% missing):")
    print(f"  Max μ diff: {np.max(np.abs(cpu_result2.muhat - gpu_result2.muhat)):.2e}")
    print(f"  Max Σ diff: {np.max(np.abs(cpu_result2.sigmahat - gpu_result2.sigmahat)):.2e}")
    
    # Test 3: Try with different tolerances
    print("\n" + "=" * 70)
    print("TEST 3: TOLERANCE SENSITIVITY (5% missing)")
    print("=" * 70)
    
    for tol in [1e-4, 1e-6, 1e-8]:
        print(f"\n--- Tolerance = {tol:.0e} ---")
        
        cpu_tol = mlest(data_missing, method='BFGS', backend='cpu', 
                       tol=tol, max_iter=200, verbose=False)
        gpu_tol = mlest(data_missing, method='BFGS', backend='gpu',
                       tol=tol, max_iter=200, verbose=False)
        
        print(f"CPU: {cpu_tol.n_iter} iterations")
        print(f"GPU: {gpu_tol.n_iter} iterations")
        print(f"Max Σ diff: {np.max(np.abs(cpu_tol.sigmahat - gpu_tol.sigmahat)):.2e}")
    
    # Test 4: Direct scipy minimize comparison
    print("\n" + "=" * 70)
    print("TEST 4: DIRECT SCIPY MINIMIZE")
    print("=" * 70)
    
    from pymvnmle._objectives import get_objective
    from pymvnmle._objectives.parameterizations import (
        InverseCholeskyParameterization,
        CholeskyParameterization
    )
    
    cpu_obj = get_objective(data_missing, backend='cpu')
    gpu_obj = get_objective(data_missing, backend='gpu', precision='fp32')
    
    x0_cpu = cpu_obj.get_initial_parameters()
    x0_gpu = gpu_obj.get_initial_parameters()
    
    print("\nDirect scipy.optimize.minimize with BFGS:")
    
    # CPU optimization
    print("\nCPU (scipy direct):")
    res_cpu = minimize(
        cpu_obj.compute_objective,
        x0_cpu,
        method='BFGS',
        jac=cpu_obj.compute_gradient,
        options={'gtol': 1e-6, 'maxiter': 200}
    )
    print(f"  Success: {res_cpu.success}")
    print(f"  Iterations: {res_cpu.nit}")
    print(f"  Final objective: {res_cpu.fun:.6f}")
    
    # GPU optimization  
    print("\nGPU (scipy direct):")
    res_gpu = minimize(
        gpu_obj.compute_objective,
        x0_gpu,
        method='BFGS',
        jac=gpu_obj.compute_gradient,
        options={'gtol': 1e-6, 'maxiter': 200}
    )
    print(f"  Success: {res_gpu.success}")
    print(f"  Iterations: {res_gpu.nit}")
    print(f"  Final objective: {res_gpu.fun:.6f}")
    
    # Extract and compare parameters
    cpu_param = InverseCholeskyParameterization(n_vars)
    gpu_param = CholeskyParameterization(n_vars)
    
    mu_cpu, sigma_cpu, _ = cpu_obj.extract_parameters(res_cpu.x)
    mu_gpu, sigma_gpu, _ = gpu_obj.extract_parameters(res_gpu.x)
    
    print(f"\nFinal parameter differences (scipy direct):")
    print(f"  Max μ diff: {np.max(np.abs(mu_cpu - mu_gpu)):.2e}")
    print(f"  Max Σ diff: {np.max(np.abs(sigma_cpu - sigma_gpu)):.2e}")
    
    # Show actual Sigma values
    print("\n" + "=" * 70)
    print("ACTUAL SIGMA VALUES")
    print("=" * 70)
    
    print("\nTrue Σ:")
    print(true_sigma)
    
    print("\nCPU Σ:")
    print(sigma_cpu)
    
    print("\nGPU Σ:")
    print(sigma_gpu)
    
    print("\nDifference (CPU - GPU):")
    print(sigma_cpu - sigma_gpu)
    
    # Check condition numbers
    print("\n" + "=" * 70)
    print("CONDITION NUMBERS")
    print("=" * 70)
    
    print(f"True Σ condition number: {np.linalg.cond(true_sigma):.2e}")
    print(f"CPU Σ condition number: {np.linalg.cond(sigma_cpu):.2e}")
    print(f"GPU Σ condition number: {np.linalg.cond(sigma_gpu):.2e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    simple_convergence_test()