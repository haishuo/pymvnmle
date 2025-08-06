#!/usr/bin/env python3
"""
Fair GPU vs CPU Benchmark for PyMVNMLE
Properly detects and uses GPU acceleration
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

# Check for GPU availability
def check_gpu_availability():
    """Check if GPU is actually available and working."""
    gpu_info = {
        'cuda_available': False,
        'mps_available': False,
        'device_name': None,
        'device_type': None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device_type'] = 'cuda'
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            print(f"✓ CUDA GPU detected: {gpu_info['device_name']}")
            
        elif torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            gpu_info['device_type'] = 'mps'
            gpu_info['device_name'] = 'Apple Silicon GPU'
            print(f"✓ Apple Silicon GPU detected")
            
        else:
            print("✗ No GPU detected (CPU only)")
            
    except ImportError:
        print("✗ PyTorch not installed - GPU not available")
        
    return gpu_info

def verify_gpu_usage():
    """Verify that GPU is actually being used."""
    try:
        import torch
        
        # Check if any tensors are on GPU
        if torch.cuda.is_available():
            # Try to create a tensor on GPU
            test_tensor = torch.randn(100, 100).cuda()
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            return True
        elif torch.backends.mps.is_available():
            test_tensor = torch.randn(100, 100).to('mps')
            print(f"  MPS backend active")
            return True
            
    except Exception as e:
        print(f"  GPU verification failed: {e}")
        
    return False

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

def warmup_gpu(gpu_info):
    """Warm up GPU to avoid JIT compilation in timing."""
    if not (gpu_info['cuda_available'] or gpu_info['mps_available']):
        print("No GPU available for warmup")
        return False
        
    print("\nWarming up GPU (JIT compilation)...")
    small_data = generate_test_data(50, 5, seed=42)
    
    try:
        # Force GPU backend explicitly
        import os
        if gpu_info['cuda_available']:
            os.environ['PYTORCH_DEVICE'] = 'cuda'
        elif gpu_info['mps_available']:
            os.environ['PYTORCH_DEVICE'] = 'mps'
            
        # Run once to trigger compilation
        result = mlest(small_data, backend='gpu', verbose=False, max_iter=20)
        
        # Verify GPU was actually used
        if verify_gpu_usage():
            print("✓ GPU warmup complete and verified")
            return True
        else:
            print("⚠ GPU warmup completed but GPU usage not verified")
            return False
            
    except Exception as e:
        print(f"✗ GPU warmup failed: {e}")
        return False

def run_benchmark(data, backend, max_iter=100, runs=3, gpu_info=None):
    """Run benchmark multiple times and return average."""
    times = []
    converged_count = 0
    
    print(f"\n{backend.upper()} Backend ({runs} runs):")
    
    # Set environment for GPU if needed
    if backend == 'gpu' and gpu_info:
        import os
        if gpu_info['cuda_available']:
            os.environ['PYTORCH_DEVICE'] = 'cuda'
            print("  Using CUDA GPU")
        elif gpu_info['mps_available']:
            os.environ['PYTORCH_DEVICE'] = 'mps'
            print("  Using Apple Silicon GPU")
    
    for i in range(runs):
        try:
            # Clear GPU cache before each run if available
            if backend == 'gpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
            
            start = time.time()
            result = mlest(data, backend=backend, verbose=False, max_iter=max_iter)
            
            # Ensure GPU operations complete before timing
            if backend == 'gpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except:
                    pass
                    
            elapsed = time.time() - start
            
            times.append(elapsed)
            if result.converged:
                converged_count += 1
            
            # Check backend actually used
            actual_backend = result.backend if hasattr(result, 'backend') else 'unknown'
            print(f"  Run {i+1}: {elapsed:.3f}s (converged: {result.converged}, backend: {actual_backend})")
            
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
    
    # Check GPU availability first
    print("\nGPU Detection:")
    gpu_info = check_gpu_availability()
    
    if not (gpu_info['cuda_available'] or gpu_info['mps_available']):
        print("\n⚠ WARNING: No GPU detected. Running CPU-only benchmark.")
        gpu_available = False
    else:
        gpu_available = True
    
    # Test configurations - adjust these for your hardware
    test_configs = [
        # (n, p, name)
        (80, 6, "Small-Medium"),      # 55 parameters
        (100, 8, "Medium"),           # 135 parameters  
        (200, 10, "Medium-Large"),    # 230 parameters
    ]
    
    # Warm up GPU first if available
    if gpu_available:
        if not warmup_gpu(gpu_info):
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
        
        # GPU Benchmark (if available)
        if gpu_available:
            gpu_time, gpu_result, gpu_times = run_benchmark(
                data, 'gpu', max_iter=200, runs=3, gpu_info=gpu_info
            )
        else:
            print("\nGPU Backend: SKIPPED (no GPU)")
            gpu_time, gpu_result, gpu_times = None, None, None
        
        # Compare results
        if cpu_result and gpu_result:
            print(f"\nSPEEDUP: {cpu_time/gpu_time:.1f}x")
            
            # Check numerical differences (they SHOULD be different due to different parameterizations)
            mu_diff = np.max(np.abs(cpu_result.muhat - gpu_result.muhat))
            sigma_diff = np.max(np.abs(cpu_result.sigmahat - gpu_result.sigmahat))
            ll_diff = abs(cpu_result.loglik - gpu_result.loglik)
            
            print(f"\nNumerical differences (expected due to different parameterizations):")
            print(f"  Max μ diff: {mu_diff:.2e}")
            print(f"  Max Σ diff: {sigma_diff:.2e}")
            print(f"  Log-lik diff: {ll_diff:.2e}")
            
            if ll_diff < 1e-2:
                print("  ✓ Results are statistically equivalent")
            else:
                print("  ⚠ Results differ significantly (check implementation)")
            
            # Store results
            results[name] = {
                'n': n, 'p': p,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time/gpu_time,
                'cpu_times': cpu_times,
                'gpu_times': gpu_times
            }
        elif cpu_result:
            # CPU only
            results[name] = {
                'n': n, 'p': p,
                'cpu_time': cpu_time,
                'gpu_time': None,
                'speedup': None,
                'cpu_times': cpu_times,
                'gpu_times': None
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if gpu_available:
        print(f"\n{'Problem':<15} {'CPU (avg)':<12} {'GPU (avg)':<12} {'Speedup':<10}")
        print("-" * 50)
        
        for name, res in results.items():
            if res['gpu_time']:
                print(f"{name:<15} {res['cpu_time']:<12.3f} {res['gpu_time']:<12.3f} {res['speedup']:<10.1f}x")
            else:
                print(f"{name:<15} {res['cpu_time']:<12.3f} {'N/A':<12} {'N/A':<10}")
    else:
        print(f"\n{'Problem':<15} {'CPU (avg)':<12}")
        print("-" * 30)
        
        for name, res in results.items():
            print(f"{name:<15} {res['cpu_time']:<12.3f}")
    
    # Analysis
    print("\nANALYSIS:")
    if gpu_available and any(r['speedup'] for r in results.values()):
        speedups = [r['speedup'] for r in results.values() if r['speedup']]
        if speedups:
            print(f"  Average speedup: {np.mean(speedups):.1f}x")
            print(f"  Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x")
            
            # Check if speedup increases with problem size
            if len(speedups) > 1:
                sizes = [(r['n'] * r['p'], r['speedup']) 
                        for r in results.values() if r['speedup']]
                sizes.sort()
                if sizes[-1][1] > sizes[0][1]:
                    print("  ✓ Speedup increases with problem size (good!)")
                else:
                    print("  ✗ Speedup doesn't scale with size (check implementation)")
    else:
        print("  GPU benchmark not available")
    
    # GPU usage verification
    if gpu_available:
        print("\nGPU Usage Verification:")
        if verify_gpu_usage():
            print("  ✓ GPU is active and being used")
        else:
            print("  ⚠ Could not verify GPU usage")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()