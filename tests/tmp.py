#!/usr/bin/env python3
"""
Diagnose the Sigma accuracy problem between CPU and GPU objectives.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymvnmle._objectives import get_objective
from pymvnmle._objectives.parameterizations import (
    InverseCholeskyParameterization,
    CholeskyParameterization
)


def diagnose_sigma_issue():
    """Diagnose why Sigma estimates differ between CPU and GPU."""
    
    print("=" * 70)
    print("DIAGNOSING SIGMA ACCURACY ISSUE")
    print("=" * 70)
    
    # Create simple test data
    np.random.seed(42)
    n_obs, n_vars = 100, 5
    
    # Known true parameters
    true_mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_sigma = np.array([
        [2.0, 0.5, 0.3, 0.2, 0.1],
        [0.5, 3.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 2.5, 0.5, 0.3],
        [0.2, 0.3, 0.5, 2.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.5]
    ])
    
    # Generate complete data (no missing values for simplicity)
    data = np.random.multivariate_normal(true_mu, true_sigma, size=n_obs)
    
    print(f"\nTest setup:")
    print(f"  Data: {n_obs}×{n_vars} (complete, no missing)")
    print(f"  True μ: {true_mu}")
    print(f"  True Σ diagonal: {np.diag(true_sigma)}")
    
    # Create objectives
    cpu_obj = get_objective(data, backend='cpu')
    gpu_obj = get_objective(data, backend='gpu', precision='fp32')
    
    print(f"\nObjective types:")
    print(f"  CPU: {type(cpu_obj).__name__}")
    print(f"  GPU: {type(gpu_obj).__name__}")
    
    # Create parameterizations
    cpu_param = InverseCholeskyParameterization(n_vars)
    gpu_param = CholeskyParameterization(n_vars)
    
    # Test 1: Check if we can round-trip the true parameters
    print("\n" + "-" * 70)
    print("TEST 1: Parameter round-trip with true values")
    print("-" * 70)
    
    # CPU parameterization
    theta_cpu = cpu_param.pack(true_mu, true_sigma)
    mu_cpu_unpack, sigma_cpu_unpack = cpu_param.unpack(theta_cpu)
    
    print(f"\nCPU (Inverse Cholesky):")
    print(f"  θ length: {len(theta_cpu)}")
    print(f"  μ error: {np.max(np.abs(mu_cpu_unpack - true_mu)):.2e}")
    print(f"  Σ error: {np.max(np.abs(sigma_cpu_unpack - true_sigma)):.2e}")
    
    # GPU parameterization
    theta_gpu = gpu_param.pack(true_mu, true_sigma)
    mu_gpu_unpack, sigma_gpu_unpack = gpu_param.unpack(theta_gpu)
    
    print(f"\nGPU (Standard Cholesky):")
    print(f"  θ length: {len(theta_gpu)}")
    print(f"  μ error: {np.max(np.abs(mu_gpu_unpack - true_mu)):.2e}")
    print(f"  Σ error: {np.max(np.abs(sigma_gpu_unpack - true_sigma)):.2e}")
    
    # Test 2: Check extract_parameters method
    print("\n" + "-" * 70)
    print("TEST 2: extract_parameters with true theta")
    print("-" * 70)
    
    # CPU extract
    mu_cpu_ext, sigma_cpu_ext, loglik_cpu = cpu_obj.extract_parameters(theta_cpu)
    print(f"\nCPU extract_parameters:")
    print(f"  μ error: {np.max(np.abs(mu_cpu_ext - true_mu)):.2e}")
    print(f"  Σ error: {np.max(np.abs(sigma_cpu_ext - true_sigma)):.2e}")
    print(f"  Log-likelihood: {loglik_cpu:.3f}")
    
    # GPU extract
    mu_gpu_ext, sigma_gpu_ext, loglik_gpu = gpu_obj.extract_parameters(theta_gpu)
    print(f"\nGPU extract_parameters:")
    print(f"  μ error: {np.max(np.abs(mu_gpu_ext - true_mu)):.2e}")
    print(f"  Σ error: {np.max(np.abs(sigma_gpu_ext - true_sigma)):.2e}")
    print(f"  Log-likelihood: {loglik_gpu:.3f}")
    
    # Test 3: Check objectives at true parameters
    print("\n" + "-" * 70)
    print("TEST 3: Objective values at true parameters")
    print("-" * 70)
    
    obj_cpu = cpu_obj.compute_objective(theta_cpu)
    obj_gpu = gpu_obj.compute_objective(theta_gpu)
    
    print(f"\nObjective values (-2 * log-likelihood):")
    print(f"  CPU: {obj_cpu:.6f}")
    print(f"  GPU: {obj_gpu:.6f}")
    print(f"  Difference: {abs(obj_cpu - obj_gpu):.6f}")
    
    # Test 4: Check gradients at true parameters
    print("\n" + "-" * 70)
    print("TEST 4: Gradients at true parameters")
    print("-" * 70)
    
    grad_cpu = cpu_obj.compute_gradient(theta_cpu)
    grad_gpu = gpu_obj.compute_gradient(theta_gpu)
    
    print(f"\nGradient norms:")
    print(f"  CPU: {np.linalg.norm(grad_cpu):.6f}")
    print(f"  GPU: {np.linalg.norm(grad_gpu):.6f}")
    print(f"  Max gradient:")
    print(f"    CPU: {np.max(np.abs(grad_cpu)):.6f}")
    print(f"    GPU: {np.max(np.abs(grad_gpu)):.6f}")
    
    # Test 5: Run a few optimization steps
    print("\n" + "-" * 70)
    print("TEST 5: Run 5 optimization steps from initial parameters")
    print("-" * 70)
    
    # Get initial parameters
    theta0_cpu = cpu_obj.get_initial_parameters()
    theta0_gpu = gpu_obj.get_initial_parameters()
    
    # Initial extract
    mu0_cpu, sigma0_cpu, _ = cpu_obj.extract_parameters(theta0_cpu)
    mu0_gpu, sigma0_gpu, _ = gpu_obj.extract_parameters(theta0_gpu)
    
    print(f"\nInitial parameters:")
    print(f"  CPU μ[0]: {mu0_cpu[0]:.6f}")
    print(f"  GPU μ[0]: {mu0_gpu[0]:.6f}")
    print(f"  CPU Σ[0,0]: {sigma0_cpu[0,0]:.6f}")
    print(f"  GPU Σ[0,0]: {sigma0_gpu[0,0]:.6f}")
    
    # Simple gradient descent
    theta_cpu = theta0_cpu.copy()
    theta_gpu = theta0_gpu.copy()
    lr = 0.001
    
    for i in range(5):
        # CPU step
        grad_cpu = cpu_obj.compute_gradient(theta_cpu)
        theta_cpu -= lr * grad_cpu
        
        # GPU step
        grad_gpu = gpu_obj.compute_gradient(theta_gpu)
        theta_gpu -= lr * grad_gpu
        
        # Extract and compare
        mu_cpu, sigma_cpu, _ = cpu_obj.extract_parameters(theta_cpu)
        mu_gpu, sigma_gpu, _ = gpu_obj.extract_parameters(theta_gpu)
        
        print(f"\nStep {i+1}:")
        print(f"  CPU μ[0]: {mu_cpu[0]:.6f}, Σ[0,0]: {sigma_cpu[0,0]:.6f}")
        print(f"  GPU μ[0]: {mu_gpu[0]:.6f}, Σ[0,0]: {sigma_gpu[0,0]:.6f}")
        print(f"  μ diff: {np.max(np.abs(mu_cpu - mu_gpu)):.2e}")
        print(f"  Σ diff: {np.max(np.abs(sigma_cpu - sigma_gpu)):.2e}")
    
    # Test 6: Check if parameterizations are equivalent
    print("\n" + "-" * 70)
    print("TEST 6: Cross-parameterization test")
    print("-" * 70)
    
    # Take CPU's optimal mu and sigma, pack with GPU parameterization
    mu_cpu_final, sigma_cpu_final, _ = cpu_obj.extract_parameters(theta_cpu)
    theta_gpu_from_cpu = gpu_param.pack(mu_cpu_final, sigma_cpu_final)
    mu_check, sigma_check = gpu_param.unpack(theta_gpu_from_cpu)
    
    print(f"\nCPU solution -> GPU parameterization -> unpack:")
    print(f"  μ preserved: {np.max(np.abs(mu_check - mu_cpu_final)):.2e}")
    print(f"  Σ preserved: {np.max(np.abs(sigma_check - sigma_cpu_final)):.2e}")
    
    # Compute objective with GPU using CPU's solution
    obj_gpu_at_cpu_sol = gpu_obj.compute_objective(theta_gpu_from_cpu)
    obj_cpu_at_cpu_sol = cpu_obj.compute_objective(theta_cpu)
    
    print(f"\nObjective at CPU solution:")
    print(f"  CPU objective: {obj_cpu_at_cpu_sol:.6f}")
    print(f"  GPU objective (CPU's solution): {obj_gpu_at_cpu_sol:.6f}")
    print(f"  Difference: {abs(obj_cpu_at_cpu_sol - obj_gpu_at_cpu_sol):.6f}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    
    print("\n🔍 KEY FINDINGS:")
    if abs(obj_cpu_at_cpu_sol - obj_gpu_at_cpu_sol) > 0.01:
        print("❌ GPU and CPU compute different objective values for same (μ,Σ)!")
        print("   This indicates a bug in the objective computation.")
    else:
        print("✅ GPU and CPU compute same objective for same (μ,Σ)")
    
    if np.max(np.abs(sigma_cpu - sigma_gpu)) > 0.01:
        print("❌ GPU and CPU converge to different Σ values!")
        print("   This could be due to different convergence criteria or numerical issues.")
    
    if np.max(np.abs(sigma_check - sigma_cpu_final)) > 1e-10:
        print("❌ Parameterization round-trip fails!")
        print("   This indicates a bug in pack/unpack methods.")


if __name__ == "__main__":
    diagnose_sigma_issue()