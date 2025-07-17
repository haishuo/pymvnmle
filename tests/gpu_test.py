#!/usr/bin/env python3
"""
Fair GPU vs CPU Benchmark for PyMVNMLE
Includes warmup to avoid JIT compilation overhead
Tests realistic medium-sized problems
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pymvnmle import mlest
    PYMVNMLE_AVAILABLE = True
except ImportError:
    PYMVNMLE_AVAILABLE = False
    print("WARNING: PyMVNMLE not found in Python path")

def generate_test_data(n, p, missing_rate=0.15, seed=None):
    """Generate test data with specified dimensions."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate multivariate normal data with some correlation
    # Create a more realistic covariance structure
    A = np.random.randn(p, p) * 0.3
    cov = np.eye(p) + A @ A.T  # Ensure positive definite
    
    data = np.random.multivariate_normal(np.zeros(p), cov, n)
    
    # Add missingness
    missing_mask = np.random.random(data.shape) < missing_rate
    data[missing_mask] = np.nan
    
    return data

def warmup_gpu():
    """Warm up GPU to avoid JIT compilation in timing."""
    print("Warming up GPU (JIT compilation)...")
    small_data = generate_test_data(50, 5, seed=42)
    try:
        # Run once to trigger compilation
        result = mlest(small_data, backend='gpu', verbose=False, max_iter=20)
        print("✓ GPU warmup complete")
        return True
    except Exception as e:
        print(f"✗ GPU warmup failed: {e}")
        return False

def run_benchmark(data, backend, max_iter=100, runs=3):
    """Run benchmark multiple times and return average."""
    times = []
    converged_count = 0
    
    print(f"\n{backend.upper()} Backend ({runs} runs):")
    
    for i in range(runs):
        try:
            start = time.time()
            result = mlest(data, backend=backend, verbose=False, max_iter=max_iter)
            elapsed = time.time() - start
            
            times.append(elapsed)
            if result.converged:
                converged_count += 1
            
            print(f"  Run {i+1}: {elapsed:.3f}s (converged: {result.converged})")
            
            # Store result from first run
            if i == 0:
                first_result = result
                
        except Exception as e:
            print(f"  Run {i+1}: FAILED - {e}")
            return None, None, None
    
    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s (converged: {converged_count}/{runs})")
    
    return avg_time, first_result, times

def main():
    """Run fair GPU vs CPU benchmark."""
    print("=" * 70)
    print("FAIR GPU vs CPU BENCHMARK")
    print("=" * 70)
    
    if not PYMVNMLE_AVAILABLE:
        print("\n❌ ERROR: PyMVNMLE not installed")
        return
    
    # Test configurations - adjust these for your hardware
    test_configs = [
        # (n, p, name)
        (80, 6, "Small-Medium"),      # 55 parameters
        (100, 8, "Medium"),            # 135 parameters  
        (200, 10, "Medium-Large"),      # 230 parameters
    ]
    
    # Warm up GPU first
    if not warmup_gpu():
        print("GPU warmup failed, continuing anyway...")
    
    results = {}
    
    for n, p, name in test_configs:
        print(f"\n{'='*70}")
        print(f"TEST: {name} (n={n}, p={p})")
        print(f"Parameters to estimate: {p + p*(p+1)//2}")
        print(f"{'='*70}")
        
        # Generate data
        print(f"\nGenerating test data...")
        data = generate_test_data(n, p, missing_rate=0.15, seed=42)
        n_missing = np.sum(np.isnan(data))
        print(f"Data: {n} × {p}, missing: {n_missing}/{n*p} ({n_missing/(n*p)*100:.1f}%)")
        
        # CPU Benchmark
        cpu_time, cpu_result, cpu_times = run_benchmark(data, 'cpu', max_iter=200, runs=3)
        
        # GPU Benchmark  
        gpu_time, gpu_result, gpu_times = run_benchmark(data, 'gpu', max_iter=200, runs=3)
        
        # Compare results
        if cpu_result and gpu_result:
            print(f"\nSPEEDUP: {cpu_time/gpu_time:.1f}x")
            
            # Check numerical equivalence
            mu_diff = np.max(np.abs(cpu_result.muhat - gpu_result.muhat))
            sigma_diff = np.max(np.abs(cpu_result.sigmahat - gpu_result.sigmahat))
            ll_diff = abs(cpu_result.loglik - gpu_result.loglik)
            
            print(f"\nNumerical differences:")
            print(f"  Max μ diff: {mu_diff:.2e}")
            print(f"  Max Σ diff: {sigma_diff:.2e}")
            print(f"  Log-lik diff: {ll_diff:.2e}")
            
            # Store results
            results[name] = {
                'n': n, 'p': p,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time/gpu_time,
                'cpu_times': cpu_times,
                'gpu_times': gpu_times
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Problem':<15} {'CPU (avg)':<12} {'GPU (avg)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        print(f"{name:<15} {res['cpu_time']:<12.3f} {res['gpu_time']:<12.3f} {res['speedup']:<10.1f}x")
    
    # Analysis
    print("\nANALYSIS:")
    if results:
        speedups = [r['speedup'] for r in results.values()]
        print(f"  Average speedup: {np.mean(speedups):.1f}x")
        print(f"  Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x")
        
        # Check if speedup increases with problem size
        if len(results) > 1:
            sizes = [(r['n'] * r['p'], r['speedup']) for r in results.values()]
            sizes.sort()
            if sizes[-1][1] > sizes[0][1]:
                print("  ✓ Speedup increases with problem size (good!)")
            else:
                print("  ✗ Speedup doesn't scale with size (check implementation)")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()