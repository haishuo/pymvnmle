#!/usr/bin/env python3
"""
PyMVNMLE Large Dataset GPU Performance Test
n=1000, p=20 - Where CPU chokes but GPU shines!
RTX 5070 Ti should handle this like a champ
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

def generate_large_test_data(n=1000, p=20, missing_rate=0.15, seed=42):
    """Generate large test data that will stress CPU but not GPU."""
    np.random.seed(seed)
    
    # Generate multivariate normal data with some correlation structure
    # Make it more realistic with a structured covariance
    A = np.random.randn(p, p)
    cov = A @ A.T  # Ensure positive definite
    cov = cov / np.max(np.abs(cov)) * 2  # Scale it reasonably
    
    data = np.random.multivariate_normal(np.zeros(p), cov, n)
    
    # Add missingness
    missing_mask = np.random.random(data.shape) < missing_rate
    data[missing_mask] = np.nan
    
    # Report stats
    n_missing = np.sum(missing_mask)
    total_values = n * p
    actual_missing_rate = n_missing / total_values
    
    print(f"Generated data: {n} observations × {p} variables")
    print(f"Missing values: {n_missing:,}/{total_values:,} ({actual_missing_rate:.1%})")
    print(f"Number of parameters to estimate: {p + p*(p+1)//2}")
    
    # Estimate number of patterns (rough)
    unique_patterns = len(np.unique(missing_mask.astype(int) @ (2**np.arange(p)), return_counts=True)[0])
    print(f"Estimated missingness patterns: ~{unique_patterns}")
    
    return data

def test_large_gpu_performance():
    """Test GPU vs CPU performance on large dataset."""
    print("=" * 70)
    print("PyMVNMLE LARGE DATASET GPU TEST")
    print("n=1000, p=20 - The CPU Killer!")
    print("Running on: NVIDIA GeForce RTX 5070 Ti")
    print("=" * 70)
    
    if not PYMVNMLE_AVAILABLE:
        print("\n❌ ERROR: PyMVNMLE not installed or not in path")
        return
    
    # Generate large test data
    print("\nGenerating large test data...")
    data = generate_large_test_data(n=1000, p=20, missing_rate=0.15)
    
    # Test GPU backend FIRST (since CPU might take forever)
    print("\n" + "-" * 50)
    print("GPU BACKEND TEST (RTX 5070 Ti)")
    print("-" * 50)
    
    try:
        start_time = time.time()
        result_gpu = mlest(data, backend='gpu', verbose=True, max_iter=1000)
        gpu_time = time.time() - start_time
        
        print(f"\n✅ GPU Success!")
        print(f"  Time: {gpu_time:.3f} seconds")
        print(f"  Converged: {result_gpu.converged}")
        print(f"  Iterations: {result_gpu.n_iter}")
        print(f"  Log-likelihood: {result_gpu.loglik:.6f}")
        print(f"  Backend used: {result_gpu.backend}")
        
        if gpu_time < 10:
            print(f"\n🚀 GPU handled this large problem in under 10 seconds!")
        elif gpu_time < 30:
            print(f"\n✨ GPU solved this in reasonable time!")
        
    except Exception as e:
        print(f"\n❌ GPU Failed: {type(e).__name__}: {e}")
        gpu_time = None
        result_gpu = None
    
    # Test CPU backend (with timeout warning)
    print("\n" + "-" * 50)
    print("CPU BACKEND TEST")
    print("-" * 50)
    print("⚠️  WARNING: This might take a VERY long time...")
    print("    (n=1000, p=20 means 230 parameters with finite differences)")
    print("    Press Ctrl+C to skip if it's taking too long")
    
    try:
        start_time = time.time()
        
        # Add a progress indicator
        print("\nStarting CPU optimization...")
        result_cpu = mlest(data, backend='cpu', verbose=True, max_iter=200)  # Lower max_iter
        cpu_time = time.time() - start_time
        
        print(f"\n✅ CPU Success!")
        print(f"  Time: {cpu_time:.3f} seconds")
        print(f"  Converged: {result_cpu.converged}")
        print(f"  Iterations: {result_cpu.n_iter}")
        print(f"  Log-likelihood: {result_cpu.loglik:.6f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  CPU test interrupted by user")
        cpu_time = time.time() - start_time
        print(f"  Time before interruption: {cpu_time:.1f} seconds")
        cpu_time = None
        result_cpu = None
    except Exception as e:
        print(f"\n❌ CPU Failed: {type(e).__name__}: {e}")
        cpu_time = None
        result_cpu = None
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\nProblem size: n=1000 × p=20 (230 parameters)")
    print(f"Missing data: ~15%")
    
    if gpu_time is not None:
        print(f"\nGPU Time (RTX 5070 Ti): {gpu_time:.3f} seconds")
        
        if gpu_time < 5:
            print("🏆 BLAZING FAST! Under 5 seconds for this large problem!")
        elif gpu_time < 10:
            print("🚀 Excellent! Under 10 seconds!")
        elif gpu_time < 30:
            print("✨ Good performance for this problem size")
    
    if cpu_time is not None and gpu_time is not None:
        speedup = cpu_time / gpu_time
        print(f"\nCPU Time: {cpu_time:.3f} seconds")
        print(f"\n💥 GPU SPEEDUP: {speedup:.1f}x faster!")
        
        if speedup > 50:
            print("🤯 OVER 50X SPEEDUP! The GPU is crushing it!")
        elif speedup > 20:
            print("🔥 Massive speedup! This is why we need GPUs!")
    elif cpu_time is None and gpu_time is not None:
        print(f"\nCPU: Did not complete (too slow)")
        print(f"💀 CPU CHOKED on this problem size!")
        print(f"🚀 GPU completed in {gpu_time:.1f}s while CPU couldn't finish!")
    
    # Show why GPU is so much better
    if result_gpu is not None:
        print("\n" + "-" * 50)
        print("WHY GPU DOMINATES:")
        print("-" * 50)
        print(f"1. Analytical gradients: Only {result_gpu.n_iter} iterations needed")
        print(f"2. Parallel tensor ops: RTX 5070 Ti tensor cores")
        print(f"3. No finite difference overhead: Direct gradient computation")
        print(f"4. Memory bandwidth: GPU can handle large matrices efficiently")
    
    # Numerical equivalence check (if both completed)
    if result_cpu is not None and result_gpu is not None:
        print("\n" + "-" * 50)
        print("NUMERICAL EQUIVALENCE CHECK")
        print("-" * 50)
        
        mu_diff = np.max(np.abs(result_cpu.muhat - result_gpu.muhat))
        sigma_diff = np.max(np.abs(result_cpu.sigmahat - result_gpu.sigmahat))
        ll_diff = abs(result_cpu.loglik - result_gpu.loglik)
        
        print(f"Max μ difference: {mu_diff:.2e}")
        print(f"Max Σ difference: {sigma_diff:.2e}")
        print(f"Log-likelihood difference: {ll_diff:.2e}")
        
        if mu_diff < 1e-6 and sigma_diff < 1e-6:
            print("\n✅ Results are statistically equivalent!")

if __name__ == "__main__":
    test_large_gpu_performance()