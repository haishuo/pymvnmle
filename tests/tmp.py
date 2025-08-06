#!/usr/bin/env python3
"""
Direct test of GPU objectives, bypassing mlest entirely.
This will tell us if the GPU objectives actually work.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("DIRECT GPU OBJECTIVE TEST")
print("=" * 70)

# Generate test data
np.random.seed(42)
n, p = 50, 5
data = np.random.randn(n, p)
data[np.random.rand(n, p) < 0.15] = np.nan

print(f"\nTest data: {n} × {p}, {np.sum(np.isnan(data))} missing values")

# Test 1: Try to create GPU objective directly
print("\n1. Creating GPU objectives directly:")

try:
    from pymvnmle._objectives import get_objective
    
    # Try factory function
    print("\n   Using get_objective factory:")
    gpu_obj = get_objective(data, backend='gpu', precision='fp32')
    print(f"   ✓ Created: {type(gpu_obj)}")
    print(f"   Device info: {gpu_obj.get_device_info()}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Force lazy import and try direct instantiation
print("\n2. Forcing lazy import and direct instantiation:")

try:
    from pymvnmle._objectives import _lazy_import_gpu
    
    success = _lazy_import_gpu()
    print(f"   Lazy import success: {success}")
    
    if success:
        from pymvnmle._objectives import GPUObjectiveFP32
        print(f"   GPUObjectiveFP32 type: {type(GPUObjectiveFP32)}")
        
        if GPUObjectiveFP32 is not None:
            gpu_obj = GPUObjectiveFP32(data)
            print(f"   ✓ Created directly: {type(gpu_obj)}")
        else:
            print("   ✗ GPUObjectiveFP32 is still None!")
            
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Import the file directly
print("\n3. Direct file import (bypassing __init__.py):")

try:
    from pymvnmle._objectives.gpu_fp32_objective import GPUObjectiveFP32
    
    print(f"   ✓ Imported GPUObjectiveFP32: {GPUObjectiveFP32}")
    
    gpu_obj = GPUObjectiveFP32(data)
    print(f"   ✓ Created: {type(gpu_obj)}")
    
    # Check it's actually on GPU
    device_info = gpu_obj.get_device_info()
    print(f"   Device: {device_info.get('device', 'unknown')}")
    print(f"   GPU name: {device_info.get('gpu_name', 'unknown')}")
    print(f"   Memory allocated: {device_info.get('memory_allocated', 0)} bytes")
    
    # Test computation
    print("\n4. Testing GPU computation:")
    
    # Get initial parameters
    theta = gpu_obj.get_initial_parameters()
    print(f"   Parameters: {len(theta)} values")
    
    # Compute objective
    start = time.time()
    obj_val = gpu_obj.compute_objective(theta)
    gpu_time = time.time() - start
    print(f"   Objective: {obj_val:.4f} (computed in {gpu_time:.4f}s)")
    
    # Compute gradient
    start = time.time()
    grad = gpu_obj.compute_gradient(theta)
    grad_time = time.time() - start
    print(f"   Gradient norm: {np.linalg.norm(grad):.4f} (computed in {grad_time:.4f}s)")
    
    # Check GPU memory again
    device_info = gpu_obj.get_device_info()
    print(f"   Memory after computation: {device_info.get('memory_allocated', 0)} bytes")
    
    # Compare with CPU
    print("\n5. Comparing with CPU:")
    from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
    
    cpu_obj = CPUObjectiveFP64(data)
    theta_cpu = cpu_obj.get_initial_parameters()
    
    start = time.time()
    obj_val_cpu = cpu_obj.compute_objective(theta_cpu)
    cpu_time = time.time() - start
    print(f"   CPU objective: {obj_val_cpu:.4f} (computed in {cpu_time:.4f}s)")
    
    # Extract parameters to compare
    mu_gpu, sigma_gpu, _ = gpu_obj.extract_parameters(theta)
    mu_cpu, sigma_cpu, _ = cpu_obj.extract_parameters(theta_cpu)
    
    print(f"\n   Mean difference: {np.max(np.abs(mu_gpu - mu_cpu)):.2e}")
    print(f"   Sigma difference: {np.max(np.abs(sigma_gpu - sigma_cpu)):.2e}")
    
    if abs(obj_val - obj_val_cpu) < 1e-10:
        print("\n   ⚠️ WARNING: Objectives are identical - GPU might be falling back to CPU!")
    else:
        print(f"\n   ✓ Different objectives (expected due to different parameterizations)")
        
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check PyTorch GPU usage directly
print("\n6. PyTorch GPU check:")

try:
    import torch
    
    # Create a test tensor on GPU
    test = torch.randn(100, 100, device='cuda')
    print(f"   ✓ Created tensor on GPU")
    print(f"   CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Do a computation
    result = torch.matmul(test, test)
    print(f"   ✓ Matrix multiply on GPU worked")
    
    # Clear
    del test, result
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ✗ PyTorch GPU failed: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)