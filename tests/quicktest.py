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

print("\n🔥 CPU (Intel MKL) vs GPU (RTX 5070 Ti) Performance:")
from pymvnmle._backends import benchmark_backends

# Test different matrix sizes
for size in [500, 1000, 2000, 3000]:
    print(f"\nMatrix {size}×{size}:")
    times = benchmark_backends(matrix_size=size, operation='cholesky')
    
    if 'cupy' in times and 'numpy' in times:
        numpy_time = times['numpy']
        cupy_time = times['cupy']
        speedup = numpy_time / cupy_time
        
        print(f"  Intel MKL (CPU): {numpy_time:.3f}s")
        print(f"  RTX 5070 Ti:      {cupy_time:.3f}s")
        print(f"  GPU Speedup:      {speedup:.1f}x")
    else:
        print(f"  Times: {times}")