#!/usr/bin/env python3
"""
GPU profiling test to identify performance bottlenecks.

This test uses a carefully sized problem that takes ~2-3 seconds on CPU
and should show speedup on GPU. It profiles different parts of the
computation to identify where the slowdowns occur.
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path setup
from pymvnmle import mlest
from pymvnmle._objectives import get_objective


class GPUProfiler:
    """Profiler for GPU objective function performance."""
    
    def __init__(self, data: np.ndarray):
        """Initialize profiler with test data."""
        self.data = data
        self.n_obs, self.n_vars = data.shape
        self.timings = {}
        
        # Check GPU availability
        try:
            import torch
            self.torch = torch
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            self.gpu_available = self.has_cuda or self.has_mps
        except ImportError:
            self.torch = None
            self.gpu_available = False
            self.has_cuda = False
            self.has_mps = False
    
    def profile_initialization(self) -> Dict[str, float]:
        """Profile objective initialization."""
        timings = {}
        
        # CPU initialization
        start = time.perf_counter()
        cpu_obj = get_objective(self.data, backend='cpu')
        timings['cpu_init'] = time.perf_counter() - start
        
        if self.gpu_available:
            # GPU initialization
            start = time.perf_counter()
            gpu_obj = get_objective(self.data, backend='gpu', precision='fp32')
            timings['gpu_init'] = time.perf_counter() - start
            
            # Break down GPU initialization
            start = time.perf_counter()
            gpu_obj_new = get_objective(self.data, backend='gpu', precision='fp32')
            timings['gpu_init_second'] = time.perf_counter() - start
            
            return timings, cpu_obj, gpu_obj
        
        return timings, cpu_obj, None
    
    def profile_pattern_preparation(self, obj) -> Dict[str, float]:
        """Profile pattern preparation and data transfer."""
        timings = {}
        
        if hasattr(obj, 'patterns'):
            timings['n_patterns'] = obj.n_patterns
            timings['pattern_sizes'] = [p.n_obs for p in obj.patterns]
        
        if hasattr(obj, '_prepare_gpu_patterns'):
            # Time pattern preparation for GPU
            torch = self.torch
            if torch and self.gpu_available:
                # Clear any existing patterns
                if hasattr(obj, 'gpu_patterns'):
                    del obj.gpu_patterns
                    if self.has_cuda:
                        torch.cuda.empty_cache()
                
                start = time.perf_counter()
                obj._prepare_gpu_patterns()
                timings['gpu_pattern_prep'] = time.perf_counter() - start
                
                # Measure GPU memory usage
                if self.has_cuda:
                    timings['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
                    timings['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
        
        return timings
    
    def profile_objective_computation(self, obj, theta: np.ndarray, 
                                     n_iterations: int = 10) -> Dict[str, float]:
        """Profile objective function computation."""
        timings = {}
        
        # Warmup
        _ = obj.compute_objective(theta)
        
        # Time multiple iterations
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = obj.compute_objective(theta)
            times.append(time.perf_counter() - start)
        
        timings['mean'] = np.mean(times)
        timings['std'] = np.std(times)
        timings['min'] = np.min(times)
        timings['max'] = np.max(times)
        
        return timings
    
    def profile_gradient_computation(self, obj, theta: np.ndarray,
                                    n_iterations: int = 10) -> Dict[str, float]:
        """Profile gradient computation."""
        timings = {}
        
        # Warmup
        _ = obj.compute_gradient(theta)
        
        # Time multiple iterations
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = obj.compute_gradient(theta)
            times.append(time.perf_counter() - start)
        
        timings['mean'] = np.mean(times)
        timings['std'] = np.std(times)
        timings['min'] = np.min(times)
        timings['max'] = np.max(times)
        
        return timings
    
    def profile_pattern_contributions(self, gpu_obj, theta: np.ndarray) -> Dict[str, Any]:
        """Profile individual pattern contribution computations."""
        if not self.gpu_available or gpu_obj is None:
            return {}
        
        torch = self.torch
        timings = {'patterns': []}
        
        # Convert theta to GPU
        theta_gpu = torch.tensor(theta, device=gpu_obj.device, dtype=gpu_obj.dtype)
        
        # Unpack parameters
        mu_gpu, sigma_gpu = gpu_obj._unpack_gpu(theta_gpu)
        
        # Profile each pattern
        for i, pattern in enumerate(gpu_obj.gpu_patterns):
            if pattern['n_observed'] == 0:
                continue
            
            pattern_timing = {
                'pattern_id': i,
                'n_obs': pattern['n_obs'],
                'n_observed': pattern['n_observed']
            }
            
            # Extract submatrices
            obs_idx = pattern['observed_indices']
            mu_k = mu_gpu[obs_idx]
            sigma_k = sigma_gpu[obs_idx][:, obs_idx]
            
            # Add regularization for FP32
            if gpu_obj.dtype == torch.float32:
                sigma_k = sigma_k + gpu_obj.eps * torch.eye(
                    pattern['n_observed'], device=gpu_obj.device, dtype=gpu_obj.dtype
                )
            
            # Time pattern contribution
            start = time.perf_counter()
            for _ in range(10):
                _ = gpu_obj._compute_pattern_contribution_gpu(pattern, mu_k, sigma_k)
            pattern_timing['contribution_time'] = (time.perf_counter() - start) / 10
            
            timings['patterns'].append(pattern_timing)
        
        # Aggregate statistics
        if timings['patterns']:
            contrib_times = [p['contribution_time'] for p in timings['patterns']]
            timings['total_pattern_time'] = sum(contrib_times)
            timings['mean_pattern_time'] = np.mean(contrib_times)
            timings['max_pattern_time'] = np.max(contrib_times)
            
            # Weighted by observations
            weighted_times = [
                p['contribution_time'] * p['n_obs'] 
                for p in timings['patterns']
            ]
            timings['weighted_total'] = sum(weighted_times)
        
        return timings
    
    def profile_full_optimization(self, method: str = 'BFGS', 
                                 max_iter: int = 100) -> Dict[str, Any]:
        """Profile full optimization run."""
        timings = {}
        
        # CPU optimization
        start = time.perf_counter()
        cpu_result = mlest(self.data, method=method, max_iter=max_iter, 
                          backend='cpu', verbose=False)
        timings['cpu_total'] = time.perf_counter() - start
        timings['cpu_iterations'] = cpu_result.n_iter
        timings['cpu_converged'] = cpu_result.converged
        
        if self.gpu_available:
            # GPU optimization
            start = time.perf_counter()
            gpu_result = mlest(self.data, method=method, max_iter=max_iter,
                             backend='gpu', verbose=False)
            timings['gpu_total'] = time.perf_counter() - start
            timings['gpu_iterations'] = gpu_result.n_iter
            timings['gpu_converged'] = gpu_result.converged
            timings['speedup'] = timings['cpu_total'] / timings['gpu_total']
            
            # Compare results
            mu_diff = np.max(np.abs(cpu_result.muhat - gpu_result.muhat))
            sigma_diff = np.max(np.abs(cpu_result.sigmahat - gpu_result.sigmahat))
            timings['mu_diff'] = mu_diff
            timings['sigma_diff'] = sigma_diff
        
        return timings


def generate_mvn_data(n_obs: int, mu: np.ndarray, sigma: np.ndarray,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Generate multivariate normal data.
    
    Parameters
    ----------
    n_obs : int
        Number of observations
    mu : np.ndarray
        Mean vector
    sigma : np.ndarray
        Covariance matrix
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Generated data
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.multivariate_normal(mu, sigma, size=n_obs)


def add_missing_data(data: np.ndarray, missing_prob: float,
                    seed: Optional[int] = None) -> np.ndarray:
    """
    Add missing values to data.
    
    Parameters
    ----------
    data : np.ndarray
        Complete data
    missing_prob : float
        Probability of missing values
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Data with missing values
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_with_missing = data.copy()
    mask = np.random.random(data.shape) < missing_prob
    data_with_missing[mask] = np.nan
    
    # Ensure no completely missing rows or columns
    for i in range(data.shape[0]):
        if np.all(np.isnan(data_with_missing[i, :])):
            # Keep at least one value
            j = np.random.randint(data.shape[1])
            data_with_missing[i, j] = data[i, j]
    
    for j in range(data.shape[1]):
        if np.all(np.isnan(data_with_missing[:, j])):
            # Keep at least one value
            i = np.random.randint(data.shape[0])
            data_with_missing[i, j] = data[i, j]
    
    return data_with_missing


def create_test_data(n_obs: int = 150, n_vars: int = 8, 
                    missing_prob: float = 0.15,
                    seed: int = 42) -> np.ndarray:
    """
    Create test data sized for ~2-3 second CPU runtime.
    
    Parameters
    ----------
    n_obs : int
        Number of observations (150 gives ~2-3s on most CPUs)
    n_vars : int
        Number of variables (8 gives 44 parameters)
    missing_prob : float
        Probability of missing values
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Test data with missing values
    """
    np.random.seed(seed)
    
    # Create interesting covariance structure
    A = np.random.randn(n_vars, n_vars)
    true_sigma = A @ A.T + np.eye(n_vars)
    true_mu = np.random.randn(n_vars) * 5
    
    # Generate complete data
    complete_data = generate_mvn_data(n_obs, true_mu, true_sigma)
    
    # Add missing values
    data = add_missing_data(complete_data, missing_prob, seed=seed)
    
    return data, true_mu, true_sigma


def print_profiling_results(profiler: GPUProfiler, results: Dict[str, Any]):
    """Print formatted profiling results."""
    print("\n" + "=" * 70)
    print("GPU PROFILING RESULTS")
    print("=" * 70)
    
    # Initialization
    if 'init' in results:
        print("\n1. INITIALIZATION")
        print("-" * 40)
        print(f"CPU init: {results['init']['cpu_init']:.3f}s")
        if 'gpu_init' in results['init']:
            print(f"GPU init (first): {results['init']['gpu_init']:.3f}s")
            print(f"GPU init (second): {results['init']['gpu_init_second']:.3f}s")
            overhead = results['init']['gpu_init'] - results['init']['gpu_init_second']
            print(f"JIT compilation overhead: {overhead:.3f}s")
    
    # Pattern preparation
    if 'patterns' in results:
        print("\n2. PATTERN PREPARATION")
        print("-" * 40)
        print(f"Number of patterns: {results['patterns'].get('n_patterns', 'N/A')}")
        if 'pattern_sizes' in results['patterns']:
            sizes = results['patterns']['pattern_sizes']
            print(f"Pattern sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={np.mean(sizes):.1f}")
        if 'gpu_pattern_prep' in results['patterns']:
            print(f"GPU pattern preparation: {results['patterns']['gpu_pattern_prep']:.3f}s")
        if 'gpu_memory_allocated' in results['patterns']:
            print(f"GPU memory allocated: {results['patterns']['gpu_memory_allocated']:.1f} MB")
    
    # Objective computation
    if 'objective' in results:
        print("\n3. OBJECTIVE COMPUTATION (10 iterations)")
        print("-" * 40)
        cpu_obj = results['objective']['cpu']
        print(f"CPU: {cpu_obj['mean']*1000:.2f} ± {cpu_obj['std']*1000:.2f} ms")
        
        if 'gpu' in results['objective']:
            gpu_obj = results['objective']['gpu']
            print(f"GPU: {gpu_obj['mean']*1000:.2f} ± {gpu_obj['std']*1000:.2f} ms")
            speedup = cpu_obj['mean'] / gpu_obj['mean']
            print(f"Speedup: {speedup:.2f}x")
    
    # Gradient computation
    if 'gradient' in results:
        print("\n4. GRADIENT COMPUTATION (10 iterations)")
        print("-" * 40)
        cpu_grad = results['gradient']['cpu']
        print(f"CPU: {cpu_grad['mean']*1000:.2f} ± {cpu_grad['std']*1000:.2f} ms")
        
        if 'gpu' in results['gradient']:
            gpu_grad = results['gradient']['gpu']
            print(f"GPU: {gpu_grad['mean']*1000:.2f} ± {gpu_grad['std']*1000:.2f} ms")
            speedup = cpu_grad['mean'] / gpu_grad['mean']
            print(f"Speedup: {speedup:.2f}x")
    
    # Pattern contributions
    if 'pattern_contrib' in results and results['pattern_contrib']:
        print("\n5. PATTERN CONTRIBUTION BREAKDOWN")
        print("-" * 40)
        pc = results['pattern_contrib']
        print(f"Total pattern time: {pc['total_pattern_time']*1000:.2f} ms")
        print(f"Mean per pattern: {pc['mean_pattern_time']*1000:.3f} ms")
        print(f"Max pattern time: {pc['max_pattern_time']*1000:.3f} ms")
        print(f"Weighted total: {pc['weighted_total']*1000:.2f} ms")
        
        # Show top 3 slowest patterns
        if 'patterns' in pc and pc['patterns']:
            sorted_patterns = sorted(pc['patterns'], 
                                   key=lambda x: x['contribution_time'], 
                                   reverse=True)[:3]
            print("\nSlowest patterns:")
            for p in sorted_patterns:
                print(f"  Pattern {p['pattern_id']}: "
                      f"{p['contribution_time']*1000:.3f} ms "
                      f"(n_obs={p['n_obs']}, n_vars={p['n_observed']})")
    
    # Full optimization
    if 'optimization' in results:
        print("\n6. FULL OPTIMIZATION")
        print("-" * 40)
        opt = results['optimization']
        print(f"CPU: {opt['cpu_total']:.2f}s ({opt['cpu_iterations']} iterations)")
        
        if 'gpu_total' in opt:
            print(f"GPU: {opt['gpu_total']:.2f}s ({opt['gpu_iterations']} iterations)")
            print(f"SPEEDUP: {opt['speedup']:.2f}x")
            print(f"\nAccuracy:")
            print(f"  Max μ difference: {opt['mu_diff']:.2e}")
            print(f"  Max Σ difference: {opt['sigma_diff']:.2e}")


def identify_bottlenecks(results: Dict[str, Any]):
    """Identify and report bottlenecks."""
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    bottlenecks = []
    
    # Check initialization overhead
    if 'init' in results and 'gpu_init' in results['init']:
        overhead = results['init']['gpu_init'] - results['init']['gpu_init_second']
        if overhead > 0.5:
            bottlenecks.append(f"High JIT compilation overhead: {overhead:.2f}s")
    
    # Check objective speedup
    if 'objective' in results and 'gpu' in results['objective']:
        cpu_time = results['objective']['cpu']['mean']
        gpu_time = results['objective']['gpu']['mean']
        speedup = cpu_time / gpu_time
        if speedup < 1.0:
            bottlenecks.append(f"GPU objective SLOWER than CPU: {speedup:.2f}x")
    
    # Check gradient speedup
    if 'gradient' in results and 'gpu' in results['gradient']:
        cpu_time = results['gradient']['cpu']['mean']
        gpu_time = results['gradient']['gpu']['mean']
        speedup = cpu_time / gpu_time
        if speedup < 1.5:  # Should be much faster with autodiff
            bottlenecks.append(f"Poor gradient speedup: {speedup:.2f}x")
    
    # Check pattern overhead
    if 'pattern_contrib' in results and results['pattern_contrib']:
        pc = results['pattern_contrib']
        if 'objective' in results and 'gpu' in results['objective']:
            obj_time = results['objective']['gpu']['mean']
            pattern_time = pc['total_pattern_time']
            pattern_pct = (pattern_time / obj_time) * 100
            if pattern_pct > 80:
                bottlenecks.append(f"Pattern computation dominates: {pattern_pct:.0f}% of objective")
    
    # Check overall optimization
    if 'optimization' in results and 'speedup' in results['optimization']:
        if results['optimization']['speedup'] < 1.0:
            bottlenecks.append(f"Overall optimization SLOWER on GPU: "
                             f"{results['optimization']['speedup']:.2f}x")
    
    if bottlenecks:
        print("\n⚠️  IDENTIFIED BOTTLENECKS:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"{i}. {bottleneck}")
        
        print("\n📊 RECOMMENDATIONS:")
        if any("SLOWER" in b for b in bottlenecks):
            print("• GPU is slower than CPU - investigate memory transfer overhead")
            print("• Consider batching pattern computations")
            print("• Profile CUDA kernel launches vs computation time")
        if any("pattern" in b.lower() for b in bottlenecks):
            print("• Pattern computation is the bottleneck")
            print("• Consider processing all patterns in parallel")
            print("• Use CPU to prepare batched data for GPU")
        if any("JIT" in b for b in bottlenecks):
            print("• High JIT compilation overhead")
            print("• Consider torch.compile() or pre-compilation strategies")
    else:
        print("\n✅ No major bottlenecks identified!")


def main():
    """Run comprehensive GPU profiling."""
    print("=" * 70)
    print("GPU PERFORMANCE PROFILING TEST")
    print("=" * 70)
    
    # Create test data
    print("\nCreating test data...")
    data, true_mu, true_sigma = create_test_data(
        n_obs=150,  # Tuned for ~2-3s CPU runtime
        n_vars=8,   # 44 parameters
        missing_prob=0.15
    )
    
    n_obs, n_vars = data.shape
    n_params = n_vars + n_vars * (n_vars + 1) // 2
    n_missing = np.sum(np.isnan(data))
    
    print(f"Data shape: {n_obs} × {n_vars}")
    print(f"Parameters to estimate: {n_params}")
    print(f"Missing values: {n_missing}/{n_obs*n_vars} ({100*n_missing/(n_obs*n_vars):.1f}%)")
    
    # Initialize profiler
    profiler = GPUProfiler(data)
    
    if not profiler.gpu_available:
        print("\n⚠️  No GPU available for profiling!")
        print("Install PyTorch with CUDA or MPS support to profile GPU performance.")
        return
    
    device_name = "CUDA GPU" if profiler.has_cuda else "MPS (Apple Silicon)"
    print(f"\n✓ {device_name} detected")
    
    # Run profiling
    results = {}
    
    # 1. Profile initialization
    print("\nProfiling initialization...")
    init_timings, cpu_obj, gpu_obj = profiler.profile_initialization()
    results['init'] = init_timings
    
    if gpu_obj is None:
        print("Failed to create GPU objective!")
        return
    
    # 2. Profile pattern preparation
    print("Profiling pattern preparation...")
    pattern_timings = profiler.profile_pattern_preparation(gpu_obj)
    results['patterns'] = pattern_timings
    
    # Get initial parameters
    theta = cpu_obj.get_initial_parameters()
    
    # 3. Profile objective computation
    print("Profiling objective computation...")
    results['objective'] = {
        'cpu': profiler.profile_objective_computation(cpu_obj, theta),
        'gpu': profiler.profile_objective_computation(gpu_obj, theta)
    }
    
    # 4. Profile gradient computation
    print("Profiling gradient computation...")
    results['gradient'] = {
        'cpu': profiler.profile_gradient_computation(cpu_obj, theta),
        'gpu': profiler.profile_gradient_computation(gpu_obj, theta)
    }
    
    # 5. Profile pattern contributions
    print("Profiling pattern contributions...")
    results['pattern_contrib'] = profiler.profile_pattern_contributions(gpu_obj, theta)
    
    # 6. Profile full optimization
    print("\nRunning full optimization comparison...")
    print("(This may take a few seconds...)")
    results['optimization'] = profiler.profile_full_optimization(max_iter=100)
    
    # Print results
    print_profiling_results(profiler, results)
    
    # Identify bottlenecks
    identify_bottlenecks(results)
    
    # Save raw results for further analysis
    import json
    import os
    
    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("/mnt/artifacts/pymvnmle")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp for multiple runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = artifacts_dir / f"gpu_profiling_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n📊 Raw profiling data saved to: {output_file}")
    
    # Also save a latest symlink for easy access
    latest_file = artifacts_dir / "gpu_profiling_results_latest.json"
    if latest_file.exists():
        latest_file.unlink()
    latest_file.symlink_to(output_file.name)
    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()