#!/usr/bin/env python3
"""
Diagnostic to identify the gradient computation issue in GPU implementation.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle._objectives import get_objective


def diagnose_gradient_issue():
    """Diagnose the gradient scaling/computation issue."""
    
    print("=" * 70)
    print("GRADIENT COMPUTATION DIAGNOSTIC")
    print("=" * 70)
    
    # Very simple test case
    np.random.seed(42)
    n_obs, n_vars = 100, 3  # Very small for detailed analysis
    
    # Simple data - identity covariance
    data = np.random.randn(n_obs, n_vars)
    
    print(f"\nTest data: {n_obs}×{n_vars} (COMPLETE - no missing)")
    print(f"Parameters: {n_vars + n_vars*(n_vars+1)//2} = {3 + 6}")
    
    # Create objectives
    cpu_obj = get_objective(data, backend='cpu')
    gpu_obj = get_objective(data, backend='gpu', precision='fp32')
    
    # Get initial parameters
    x0_cpu = cpu_obj.get_initial_parameters()
    x0_gpu = gpu_obj.get_initial_parameters()
    
    print(f"\nInitial parameter norm difference: {np.linalg.norm(x0_cpu - x0_gpu):.2e}")
    
    # Test 1: Compare objectives at same point
    print("\n" + "=" * 70)
    print("TEST 1: OBJECTIVE VALUES AT INITIAL POINT")
    print("=" * 70)
    
    obj_cpu_at_cpu = cpu_obj.compute_objective(x0_cpu)
    obj_gpu_at_gpu = gpu_obj.compute_objective(x0_gpu)
    
    print(f"CPU objective at CPU's x0: {obj_cpu_at_cpu:.6f}")
    print(f"GPU objective at GPU's x0: {obj_gpu_at_gpu:.6f}")
    print(f"Difference: {abs(obj_cpu_at_cpu - obj_gpu_at_gpu):.2e}")
    
    # Test 2: Compare gradients at initial point
    print("\n" + "=" * 70)
    print("TEST 2: GRADIENTS AT INITIAL POINT")
    print("=" * 70)
    
    grad_cpu = cpu_obj.compute_gradient(x0_cpu)
    grad_gpu = gpu_obj.compute_gradient(x0_gpu)
    
    print(f"CPU gradient shape: {grad_cpu.shape}")
    print(f"GPU gradient shape: {grad_gpu.shape}")
    print(f"CPU gradient norm: {np.linalg.norm(grad_cpu):.6f}")
    print(f"GPU gradient norm: {np.linalg.norm(grad_gpu):.6f}")
    print(f"Ratio (GPU/CPU): {np.linalg.norm(grad_gpu)/np.linalg.norm(grad_cpu):.2f}")
    
    # Show first few components
    print(f"\nFirst 5 gradient components:")
    print(f"CPU: {grad_cpu[:5]}")
    print(f"GPU: {grad_gpu[:5]}")
    print(f"Ratio: {grad_gpu[:5]/grad_cpu[:5]}")
    
    # Test 3: Finite difference check for GPU
    print("\n" + "=" * 70)
    print("TEST 3: FINITE DIFFERENCE VALIDATION")
    print("=" * 70)
    
    # Check GPU gradient with finite differences
    eps = 1e-6
    fd_grad = np.zeros_like(grad_gpu)
    
    for i in range(min(5, len(x0_gpu))):  # Check first 5 components
        x_plus = x0_gpu.copy()
        x_plus[i] += eps
        
        x_minus = x0_gpu.copy()
        x_minus[i] -= eps
        
        f_plus = gpu_obj.compute_objective(x_plus)
        f_minus = gpu_obj.compute_objective(x_minus)
        
        fd_grad[i] = (f_plus - f_minus) / (2 * eps)
    
    print(f"\nGPU gradient vs finite differences (first 5 components):")
    print(f"Autodiff:  {grad_gpu[:5]}")
    print(f"Finite diff: {fd_grad[:5]}")
    print(f"Relative error: {np.abs(grad_gpu[:5] - fd_grad[:5])/np.abs(grad_gpu[:5] + 1e-10)}")
    
    # Test 4: Check if gradients point in same direction
    print("\n" + "=" * 70)
    print("TEST 4: GRADIENT DIRECTION ANALYSIS")
    print("=" * 70)
    
    # Normalize gradients
    grad_cpu_normalized = grad_cpu / (np.linalg.norm(grad_cpu) + 1e-10)
    grad_gpu_normalized = grad_gpu / (np.linalg.norm(grad_gpu) + 1e-10)
    
    # Cosine similarity
    cosine_sim = np.dot(grad_cpu_normalized, grad_gpu_normalized)
    print(f"Cosine similarity between gradients: {cosine_sim:.4f}")
    
    if cosine_sim < 0.9:
        print("⚠️ WARNING: Gradients pointing in different directions!")
    elif cosine_sim > 0.99:
        print("✓ Gradients point in same direction (likely just scaling issue)")
    
    # Test 5: Pattern-by-pattern analysis
    print("\n" + "=" * 70)
    print("TEST 5: PARAMETER STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Split parameters into mean and covariance parts
    print(f"\nParameter structure (total {len(x0_cpu)} params):")
    print(f"  Mean parameters: 0 to {n_vars-1} (n={n_vars})")
    print(f"  Cov parameters: {n_vars} to {len(x0_cpu)-1} (n={len(x0_cpu)-n_vars})")
    
    grad_cpu_mean = grad_cpu[:n_vars]
    grad_cpu_cov = grad_cpu[n_vars:]
    
    grad_gpu_mean = grad_gpu[:n_vars]
    grad_gpu_cov = grad_gpu[n_vars:]
    
    print(f"\nMean gradient norms:")
    print(f"  CPU: {np.linalg.norm(grad_cpu_mean):.6f}")
    print(f"  GPU: {np.linalg.norm(grad_gpu_mean):.6f}")
    print(f"  Ratio: {np.linalg.norm(grad_gpu_mean)/np.linalg.norm(grad_cpu_mean):.2f}")
    
    print(f"\nCovariance gradient norms:")
    print(f"  CPU: {np.linalg.norm(grad_cpu_cov):.6f}")
    print(f"  GPU: {np.linalg.norm(grad_gpu_cov):.6f}")
    print(f"  Ratio: {np.linalg.norm(grad_gpu_cov)/np.linalg.norm(grad_cpu_cov):.2f}")
    
    # Test 6: Check for NaN or Inf
    print("\n" + "=" * 70)
    print("TEST 6: NUMERICAL STABILITY CHECK")
    print("=" * 70)
    
    print(f"CPU gradient has NaN: {np.any(np.isnan(grad_cpu))}")
    print(f"CPU gradient has Inf: {np.any(np.isinf(grad_cpu))}")
    print(f"GPU gradient has NaN: {np.any(np.isnan(grad_gpu))}")
    print(f"GPU gradient has Inf: {np.any(np.isinf(grad_gpu))}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if np.linalg.norm(grad_gpu) > np.linalg.norm(grad_cpu) * 10:
        print("\n🔴 CRITICAL: GPU gradient is an order of magnitude larger!")
        print("   Likely causes:")
        print("   1. Missing normalization factor (e.g., 1/n)")
        print("   2. Different parameterization Jacobian not accounted for")
        print("   3. Accumulation without averaging in batched computation")
        
        # Check if it's a simple scaling
        ratio = np.linalg.norm(grad_gpu) / np.linalg.norm(grad_cpu)
        if abs(ratio - n_obs) < 1:
            print(f"\n   💡 Ratio ≈ n_obs ({n_obs}) - missing 1/n normalization?")
        elif abs(ratio - np.sqrt(n_obs)) < 1:
            print(f"\n   💡 Ratio ≈ sqrt(n_obs) ({np.sqrt(n_obs):.1f}) - missing 1/sqrt(n) normalization?")


if __name__ == "__main__":
    diagnose_gradient_issue()