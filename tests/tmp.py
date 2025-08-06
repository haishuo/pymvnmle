"""
Debug why GPU backend is giving wrong results.
"""

import numpy as np
from pymvnmle import datasets

print("=" * 70)
print("GPU BACKEND DEBUGGING")
print("=" * 70)

# First, check if GPU objectives are even being loaded
print("\n1. Checking GPU availability:")
try:
    from pymvnmle._objectives import GPU_AVAILABLE
    print(f"   GPU_AVAILABLE: {GPU_AVAILABLE}")
except:
    print("   Can't import GPU_AVAILABLE")

# Check what backend 'auto' selects
print("\n2. Backend auto-selection:")
from pymvnmle import mlest

# Create a tiny test to see backend selection
tiny_data = np.array([[1, 2], [3, 4], [5, np.nan]])
result = mlest(tiny_data, backend='auto', verbose=True, max_iter=1)
print(f"   Selected backend: {result.backend if hasattr(result, 'backend') else 'unknown'}")

# Now test GPU directly
print("\n3. Testing GPU backend directly:")
apple_data = datasets.apple

try:
    result_gpu = mlest(apple_data, backend='gpu', verbose=False)
    print(f"   GPU log-lik: {result_gpu.loglik}")
except Exception as e:
    print(f"   GPU backend error: {e}")

# Test GPU with different precisions
print("\n4. Testing GPU precisions:")
try:
    # Try FP32
    from pymvnmle._objectives import get_objective
    obj_fp32 = get_objective(apple_data, backend='gpu', precision='fp32')
    print(f"   Created GPU FP32 objective")
    
    # Get initial params and evaluate
    theta_init = obj_fp32.get_initial_parameters()
    obj_val = obj_fp32.compute_objective(theta_init)
    print(f"   FP32 initial obj: {obj_val}")
    print(f"   FP32 initial log-lik: {-obj_val/2}")
    
except Exception as e:
    print(f"   FP32 error: {e}")

try:
    # Try FP64
    obj_fp64 = get_objective(apple_data, backend='gpu', precision='fp64')
    print(f"\n   Created GPU FP64 objective")
    
    theta_init = obj_fp64.get_initial_parameters()
    obj_val = obj_fp64.compute_objective(theta_init)
    print(f"   FP64 initial obj: {obj_val}")
    print(f"   FP64 initial log-lik: {-obj_val/2}")
    
except Exception as e:
    print(f"   FP64 error: {e}")

# The issue might be in mlest's backend selection logic
print("\n5. Checking mlest's backend selection:")
print("   When backend='auto', mlest might be:")
print("   1. Incorrectly selecting GPU when it shouldn't")
print("   2. Using wrong precision for GPU")
print("   3. GPU backend has different parameterization that's incompatible")

print("\n" + "=" * 70)
print("HYPOTHESIS:")
print("The GPU backend uses DIFFERENT parameterization (Cholesky)")
print("instead of R's Inverse Cholesky, causing wrong results!")
print("=" * 70)