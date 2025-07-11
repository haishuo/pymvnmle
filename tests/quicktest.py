from pymvnmle._backends import print_backend_summary, get_available_backends

# Fixed CuPy test:
import cupy as cp
print("CuPy version:", cp.__version__)
device_count = cp.cuda.runtime.getDeviceCount()
print("CUDA devices:", device_count)

for i in range(device_count):
    with cp.cuda.Device(i):
        props = cp.cuda.runtime.getDeviceProperties(i)
        # Use the correct memory info API
        mempool = cp.get_default_memory_pool()
        print(f"GPU {i}: {props['name'].decode()}")
        print(f"  Compute capability: {props['major']}.{props['minor']}")
        print(f"  Total memory: {props['totalGlobalMem'] // 1024**3} GB")
print("==================================================")
print_backend_summary()  # See what's available on your system
print(get_available_backends())  # Should show at least ['numpy']

print("\n🔥 CPU vs GPU Performance:")
from pymvnmle._backends import benchmark_backends

# Test different matrix sizes
for size in [500, 1000, 2000, 3000, 5000, 8000, 10000]:
    print(f"\nMatrix {size}×{size}:")
    times = benchmark_backends(matrix_size=size, operation='cholesky')
    
    # Check for ANY GPU backend, not just CuPy
    gpu_backend = None
    gpu_time = None
    
    # Check in preference order: JAX > CuPy > Metal
    for backend in ['jax', 'cupy', 'metal']:
        if backend in times:
            gpu_backend = backend
            gpu_time = times[backend]
            break
    
    if gpu_backend and 'numpy' in times:
        numpy_time = times['numpy']
        speedup = numpy_time / gpu_time
        
        print(f"  Intel MKL (CPU): {numpy_time:.3f}s")
        print(f"  {gpu_backend.upper()} (GPU): {gpu_time:.3f}s")
        print(f"  GPU Speedup:      {speedup:.1f}x")
    else:
        print(f"  Times: {times}")