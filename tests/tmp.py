#!/usr/bin/env python3
"""
Debug pattern-by-pattern contributions to find the issue.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def debug_patterns():
    """Debug each pattern's contribution."""
    print("=" * 70)
    print("PATTERN-BY-PATTERN DEBUG")
    print("=" * 70)
    
    # Create data with missing values
    np.random.seed(42)
    data = np.array([[1.0, 2.0, np.nan],
                     [2.0, np.nan, 3.0],
                     [3.0, 4.0, 5.0],
                     [np.nan, 5.0, 6.0]])
    
    print(f"Data:\n{data}")
    
    # Import objectives
    from pymvnmle._objectives import get_objective
    from pymvnmle._objectives.parameterizations import (
        InverseCholeskyParameterization,
        CholeskyParameterization
    )
    
    # Create objectives
    cpu_obj = get_objective(data, backend='cpu')
    gpu_obj = get_objective(data, backend='gpu', precision='fp32')
    
    # Use same mu and sigma
    mu = np.array([2.0, 3.0, 4.0])
    sigma = np.array([[1.0, 0.3, 0.2],
                      [0.3, 1.0, 0.4],
                      [0.2, 0.4, 1.0]])
    
    print(f"\nμ = {mu}")
    print(f"Σ = \n{sigma}")
    
    # Pack parameters
    cpu_param = InverseCholeskyParameterization(3)
    gpu_param = CholeskyParameterization(3)
    
    theta_cpu = cpu_param.pack(mu, sigma)
    theta_gpu = gpu_param.pack(mu, sigma)
    
    # Manually compute each pattern's contribution
    print("\n" + "=" * 70)
    print("CPU PATTERNS")
    print("=" * 70)
    
    total_cpu = 0
    for i, pattern in enumerate(cpu_obj.patterns):
        print(f"\nPattern {i}:")
        print(f"  N obs: {pattern.n_obs}")
        print(f"  Observed indices: {pattern.observed_indices}")
        
        # Extract submatrices
        obs_idx = pattern.observed_indices
        mu_k = mu[obs_idx]
        sigma_k = sigma[np.ix_(obs_idx, obs_idx)]
        
        print(f"  μ_k: {mu_k}")
        print(f"  Σ_k shape: {sigma_k.shape}")
        
        # Compute contribution (R-style)
        n_k = pattern.n_obs
        p_k = len(obs_idx)
        
        const = p_k * np.log(2 * np.pi)
        log_det = np.linalg.slogdet(sigma_k)[1]
        
        # Get pattern data
        data_k = pattern.data
        data_centered = data_k - mu_k
        S_k = (data_centered.T @ data_centered) / n_k
        
        sigma_k_inv = np.linalg.inv(sigma_k)
        trace_term = np.trace(sigma_k_inv @ S_k)
        
        contrib = n_k * (const + log_det + trace_term)
        total_cpu += contrib
        
        print(f"  Contribution: {contrib:.6f}")
        print(f"    Const: {n_k * const:.6f}")
        print(f"    Log-det: {n_k * log_det:.6f}")
        print(f"    Trace: {n_k * trace_term:.6f}")
    
    print(f"\nTotal CPU objective: {total_cpu:.6f}")
    print(f"CPU compute_objective: {cpu_obj.compute_objective(theta_cpu):.6f}")
    
    # Now check GPU patterns
    print("\n" + "=" * 70)
    print("GPU PATTERNS")
    print("=" * 70)
    
    # Convert to GPU
    mu_gpu = torch.tensor(mu, dtype=torch.float32, device='cuda')
    sigma_gpu = torch.tensor(sigma, dtype=torch.float32, device='cuda')
    
    total_gpu = 0
    for i, pattern in enumerate(gpu_obj.gpu_patterns):
        print(f"\nPattern {i}:")
        print(f"  N obs: {pattern['n_obs']}")
        print(f"  N observed vars: {pattern['n_observed']}")
        
        obs_idx = pattern['observed_indices']
        mu_k = mu_gpu[obs_idx]
        sigma_k = sigma_gpu[obs_idx][:, obs_idx]
        
        # Add small diagonal for stability
        sigma_k = sigma_k + 1e-6 * torch.eye(pattern['n_observed'], device='cuda', dtype=torch.float32)
        
        n_k = pattern['n_obs']
        p_k = pattern['n_observed']
        
        # Constants
        const = p_k * np.log(2 * np.pi)
        
        # Log determinant
        L_k = torch.linalg.cholesky(sigma_k)
        log_det = 2 * torch.sum(torch.log(torch.diag(L_k)))
        
        # Sample covariance
        data_centered = pattern['data'] - mu_k
        S_k = (data_centered.T @ data_centered) / n_k
        
        # Trace term
        X = torch.linalg.solve(sigma_k, S_k)
        trace_term = torch.trace(X)
        
        # Total contribution
        contrib = n_k * (const + log_det.item() + trace_term.item())
        total_gpu += contrib
        
        print(f"  Contribution: {contrib:.6f}")
        print(f"    Const: {n_k * const:.6f}")
        print(f"    Log-det: {n_k * log_det.item():.6f}")
        print(f"    Trace: {n_k * trace_term.item():.6f}")
        
        # Check if this matches the method
        contrib_method = gpu_obj._compute_pattern_contribution_gpu(
            pattern, mu_k, sigma_k - 1e-6 * torch.eye(pattern['n_observed'], device='cuda', dtype=torch.float32)
        )
        print(f"  Method contribution (per obs): {contrib_method.item():.6f}")
        print(f"  Method * n_obs: {contrib_method.item() * n_k:.6f}")
    
    print(f"\nTotal GPU objective (manual): {total_gpu:.6f}")
    print(f"Total GPU * 2 (R convention): {total_gpu * 2:.6f}")
    print(f"GPU compute_objective: {gpu_obj.compute_objective(theta_gpu):.6f}")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"CPU total: {total_cpu:.6f}")
    print(f"GPU total (manual): {total_gpu:.6f}")
    print(f"GPU total * 2: {total_gpu * 2:.6f}")
    print(f"Ratio GPU*2/CPU: {(total_gpu * 2) / total_cpu:.2f}")


if __name__ == "__main__":
    debug_patterns()