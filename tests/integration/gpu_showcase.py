#!/usr/bin/env python3
"""
GPU Showcase Test - Demonstrates massive GPU speedup potential.

This test creates data specifically designed to showcase GPU strengths:
- Large number of observations
- Few patterns (ideally 1-3)
- High-dimensional data
- Large matrix operations that parallelize well
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymvnmle import mlest
from pymvnmle._objectives import get_objective


def create_gpu_optimal_data(n_obs: int = 5000, 
                           n_vars: int = 20,
                           n_patterns: int = 3,
                           seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create data optimized for GPU performance.
    
    Parameters
    ----------
    n_obs : int
        Number of observations (5000 = lots of parallel work)
    n_vars : int  
        Number of variables (20 = substantial matrix operations)
    n_patterns : int
        Number of missing patterns (3 = few kernel launches)
    seed : int
        Random seed
        
    Returns
    -------
    data : np.ndarray
        Data with missing values
    info : dict
        Information about the data
    """
    np.random.seed(seed)
    
    # Create true parameters
    A = np.random.randn(n_vars, n_vars)
    true_sigma = A @ A.T + np.eye(n_vars)
    true_mu = np.random.randn(n_vars) * 3
    
    # Generate complete data
    data = np.random.multivariate_normal(true_mu, true_sigma, size=n_obs)
    
    # Create structured missing patterns
    # Pattern 1: First 10% missing last variable
    # Pattern 2: Next 10% missing first variable  
    # Pattern 3: Rest complete (80% of data)
    
    if n_patterns == 1:
        # Best case: all complete
        pass
    elif n_patterns == 2:
        # Two patterns: mostly complete, some missing one var
        n_missing = n_obs // 10
        indices = np.random.choice(n_obs, n_missing, replace=False)
        data[indices, -1] = np.nan
    elif n_patterns == 3:
        # Three patterns
        n_per_pattern = n_obs // 10
        indices1 = np.arange(n_per_pattern)
        indices2 = np.arange(n_per_pattern, 2 * n_per_pattern)
        data[indices1, -1] = np.nan
        data[indices2, 0] = np.nan
    else:
        # More complex pattern
        for i in range(min(n_patterns - 1, n_vars)):
            start = i * (n_obs // n_patterns)
            end = (i + 1) * (n_obs // n_patterns)
            if i < n_vars:
                data[start:end, i] = np.nan
    
    # Count actual patterns
    patterns = []
    for row in data:
        pattern = tuple(np.isnan(row))
        if pattern not in patterns:
            patterns.append(pattern)
    
    n_missing = np.sum(np.isnan(data))
    
    info = {
        'n_obs': n_obs,
        'n_vars': n_vars,
        'n_params': n_vars + n_vars * (n_vars + 1) // 2,
        'n_patterns_requested': n_patterns,
        'n_patterns_actual': len(patterns),
        'n_missing': n_missing,
        'missing_pct': 100 * n_missing / (n_obs * n_vars),
        'true_mu': true_mu,
        'true_sigma': true_sigma
    }
    
    return data, info


def profile_gpu_vs_cpu(data: np.ndarray, 
                       max_iter: int = 30,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Profile GPU vs CPU performance on the data.
    
    Parameters
    ----------
    data : np.ndarray
        Test data
    max_iter : int
        Maximum iterations for optimization
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Performance metrics
    """
    results = {}
    
    # Check GPU availability
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        gpu_available = has_cuda or has_mps
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_type = "CUDA"
        elif has_mps:
            gpu_name = "Apple Silicon"
            gpu_type = "MPS"
        else:
            gpu_name = "None"
            gpu_type = "None"
    except ImportError:
        gpu_available = False
        gpu_name = "None"
        gpu_type = "None"
    
    results['gpu_available'] = gpu_available
    results['gpu_name'] = gpu_name
    results['gpu_type'] = gpu_type
    
    if verbose:
        print("=" * 70)
        print("GPU vs CPU PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"GPU: {gpu_name} ({gpu_type})")
        print(f"Data shape: {data.shape}")
        print(f"Missing values: {np.sum(np.isnan(data))}/{data.size} "
              f"({100*np.sum(np.isnan(data))/data.size:.1f}%)")
        print("-" * 70)
    
    # Create objectives
    if verbose:
        print("\nInitializing objectives...")
    
    start = time.perf_counter()
    cpu_obj = get_objective(data, backend='cpu')
    cpu_init_time = time.perf_counter() - start
    
    if gpu_available:
        start = time.perf_counter()
        gpu_obj = get_objective(data, backend='gpu', precision='fp32')
        gpu_init_time = time.perf_counter() - start
    
    if verbose:
        print(f"CPU init: {cpu_init_time:.3f}s")
        if gpu_available:
            print(f"GPU init: {gpu_init_time:.3f}s")
        print(f"Number of patterns: {cpu_obj.n_patterns}")
        
        # Show pattern sizes
        pattern_sizes = [p.n_obs for p in cpu_obj.patterns]
        print(f"Pattern sizes: min={min(pattern_sizes)}, "
              f"max={max(pattern_sizes)}, "
              f"mean={np.mean(pattern_sizes):.0f}")
    
    # Get initial parameters
    theta = cpu_obj.get_initial_parameters()
    
    # Time objective computation
    if verbose:
        print("\nTiming objective computation (10 iterations)...")
    
    # CPU objective
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = cpu_obj.compute_objective(theta)
        times.append(time.perf_counter() - start)
    cpu_obj_time = np.mean(times[2:])  # Skip warmup
    
    results['cpu_objective_time'] = cpu_obj_time
    
    if gpu_available:
        # GPU objective (with warmup)
        _ = gpu_obj.compute_objective(theta)  # Warmup
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = gpu_obj.compute_objective(theta)
            times.append(time.perf_counter() - start)
        gpu_obj_time = np.mean(times)
        results['gpu_objective_time'] = gpu_obj_time
        results['objective_speedup'] = cpu_obj_time / gpu_obj_time
    
    # Time gradient computation
    if verbose:
        print("Timing gradient computation (10 iterations)...")
    
    # CPU gradient
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = cpu_obj.compute_gradient(theta)
        times.append(time.perf_counter() - start)
    cpu_grad_time = np.mean(times[2:])  # Skip warmup
    
    results['cpu_gradient_time'] = cpu_grad_time
    
    if gpu_available:
        # GPU gradient (with warmup)
        _ = gpu_obj.compute_gradient(theta)  # Warmup
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = gpu_obj.compute_gradient(theta)
            times.append(time.perf_counter() - start)
        gpu_grad_time = np.mean(times)
        results['gpu_gradient_time'] = gpu_grad_time
        results['gradient_speedup'] = cpu_grad_time / gpu_grad_time
    
    # Full optimization
    if verbose:
        print(f"\nRunning full optimization (max {max_iter} iterations)...")
        print("CPU optimization...")
    
    start = time.perf_counter()
    cpu_result = mlest(data, method='BFGS', max_iter=max_iter, 
                      backend='cpu', verbose=False)
    cpu_total_time = time.perf_counter() - start
    
    results['cpu_total_time'] = cpu_total_time
    results['cpu_iterations'] = cpu_result.n_iter
    results['cpu_converged'] = cpu_result.converged
    
    if gpu_available:
        if verbose:
            print("GPU optimization...")
        
        start = time.perf_counter()
        gpu_result = mlest(data, method='BFGS', max_iter=max_iter,
                          backend='gpu', verbose=False)
        gpu_total_time = time.perf_counter() - start
        
        results['gpu_total_time'] = gpu_total_time
        results['gpu_iterations'] = gpu_result.n_iter
        results['gpu_converged'] = gpu_result.converged
        results['total_speedup'] = cpu_total_time / gpu_total_time
        
        # Check accuracy
        mu_diff = np.max(np.abs(cpu_result.muhat - gpu_result.muhat))
        sigma_diff = np.max(np.abs(cpu_result.sigmahat - gpu_result.sigmahat))
        results['mu_diff'] = mu_diff
        results['sigma_diff'] = sigma_diff
    
    return results


def print_results(results: Dict[str, Any], data_info: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nData Configuration:")
    print(f"  Observations: {data_info['n_obs']:,}")
    print(f"  Variables: {data_info['n_vars']}")
    print(f"  Parameters: {data_info['n_params']}")
    print(f"  Patterns: {data_info['n_patterns_actual']} "
          f"(requested {data_info['n_patterns_requested']})")
    print(f"  Missing: {data_info['missing_pct']:.1f}%")
    
    if not results['gpu_available']:
        print("\n⚠️  No GPU available for comparison!")
        print(f"\nCPU Performance:")
        print(f"  Objective: {results['cpu_objective_time']*1000:.2f} ms")
        print(f"  Gradient: {results['cpu_gradient_time']*1000:.2f} ms")
        print(f"  Total optimization: {results['cpu_total_time']:.2f}s "
              f"({results['cpu_iterations']} iterations)")
        return
    
    print(f"\nComponent Performance:")
    print(f"  Objective:")
    print(f"    CPU: {results['cpu_objective_time']*1000:.2f} ms")
    print(f"    GPU: {results['gpu_objective_time']*1000:.2f} ms")
    print(f"    Speedup: {results['objective_speedup']:.2f}x")
    
    print(f"  Gradient:")
    print(f"    CPU: {results['cpu_gradient_time']*1000:.2f} ms")
    print(f"    GPU: {results['gpu_gradient_time']*1000:.2f} ms")
    print(f"    Speedup: {results['gradient_speedup']:.2f}x")
    
    print(f"\nFull Optimization:")
    print(f"  CPU: {results['cpu_total_time']:.2f}s ({results['cpu_iterations']} iter)")
    print(f"  GPU: {results['gpu_total_time']:.2f}s ({results['gpu_iterations']} iter)")
    print(f"  SPEEDUP: {results['total_speedup']:.2f}x")
    
    if results['total_speedup'] > 10:
        print(f"  🚀 MASSIVE SPEEDUP!")
    elif results['total_speedup'] > 5:
        print(f"  🎯 Excellent speedup!")
    elif results['total_speedup'] > 2:
        print(f"  ✅ Good speedup!")
    else:
        print(f"  ⚠️  Modest speedup")
    
    print(f"\nAccuracy:")
    print(f"  Max μ difference: {results['mu_diff']:.2e}")
    print(f"  Max Σ difference: {results['sigma_diff']:.2e}")


def main():
    """Run GPU showcase tests."""
    print("=" * 70)
    print("GPU SHOWCASE TEST")
    print("=" * 70)
    
    # Test configurations optimized for GPU
    test_configs = [
        # Config 1: Moderate size, few patterns (good GPU case)
        {
            'name': 'Moderate (2000×15, 3 patterns)',
            'n_obs': 2000,
            'n_vars': 15,
            'n_patterns': 3,
            'max_iter': 30
        },
        
        # Config 2: Large size, minimal patterns (ideal GPU case)
        {
            'name': 'Large (5000×20, 2 patterns)',
            'n_obs': 5000,
            'n_vars': 20,
            'n_patterns': 2,
            'max_iter': 25
        },
        
        # Config 3: Very large, single pattern (best GPU case)
        {
            'name': 'Very Large (10000×25, 1 pattern)',
            'n_obs': 10000,
            'n_vars': 25,
            'n_patterns': 1,
            'max_iter': 20
        },
        
        # Config 4: Massive (if you have enough memory)
        {
            'name': 'Massive (20000×30, 2 patterns)',
            'n_obs': 20000,
            'n_vars': 30,
            'n_patterns': 2,
            'max_iter': 15
        }
    ]
    
    # Let user choose
    print("\nAvailable test configurations:")
    for i, config in enumerate(test_configs, 1):
        print(f"{i}. {config['name']}")
    print(f"{len(test_configs)+1}. Run all (will take time)")
    
    try:
        choice = input(f"\nSelect configuration (1-{len(test_configs)+1}): ").strip()
        choice = int(choice)
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default configuration...")
        choice = 2
    
    if choice == len(test_configs) + 1:
        configs_to_run = test_configs
    elif 1 <= choice <= len(test_configs):
        configs_to_run = [test_configs[choice-1]]
    else:
        print(f"Invalid choice, using configuration 2...")
        configs_to_run = [test_configs[1]]
    
    # Run selected configurations
    for config in configs_to_run:
        print("\n" + "=" * 70)
        print(f"Testing: {config['name']}")
        print("=" * 70)
        
        # Create data
        print("\nGenerating test data...")
        data, info = create_gpu_optimal_data(
            n_obs=config['n_obs'],
            n_vars=config['n_vars'],
            n_patterns=config['n_patterns']
        )
        
        # Profile
        results = profile_gpu_vs_cpu(
            data, 
            max_iter=config['max_iter'],
            verbose=True
        )
        
        # Print results
        print_results(results, info)
        
        # Save results
        import json
        from datetime import datetime
        
        output_dir = Path("/mnt/artifacts/pymvnmle")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config['name'].replace(' ', '_').replace('×', 'x')
        output_file = output_dir / f"gpu_showcase_{config_name}_{timestamp}.json"
        
        save_data = {
            'config': config,
            'data_info': info,
            'results': results,
            'timestamp': timestamp
        }
        
        # Convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert_numpy(save_data), f, indent=2)
        
        print(f"\n📊 Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("GPU SHOWCASE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()