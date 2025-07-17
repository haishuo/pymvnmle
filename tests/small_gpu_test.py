#!/usr/bin/env python3
"""
Simple GPU acceleration test for PyMVNMLE
Tests basic GPU backend functionality and reports errors
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import mlest, datasets

def test_gpu_backend():
    """Test basic GPU backend functionality."""
    print("=" * 60)
    print("PyMVNMLE GPU ACCELERATION TEST")
    print("=" * 60)
    
    # Test data - use small apple dataset
    data = datasets.apple
    print(f"\nTest data: Apple dataset ({data.shape[0]} obs × {data.shape[1]} vars)")
    print(f"Missing rate: {np.sum(np.isnan(data)) / data.size:.1%}")
    
    # Test 1: Try explicit GPU request
    print("\n" + "-" * 50)
    print("TEST 1: Explicit GPU request (backend='gpu')")
    print("-" * 50)
    try:
        result = mlest(data, backend='gpu', verbose=True)
        print(f"\n✅ SUCCESS! GPU backend worked")
        print(f"  Backend used: {result.backend}")
        print(f"  Method used: {result.method}")
        print(f"  Log-likelihood: {result.loglik:.6f}")
        print(f"  Converged: {result.converged}")
    except Exception as e:
        print(f"\n❌ FAILED with {type(e).__name__}: {e}")
    
    # Test 2: Try PyTorch backend directly
    print("\n" + "-" * 50)
    print("TEST 2: PyTorch backend (backend='pytorch')")
    print("-" * 50)
    try:
        result = mlest(data, backend='pytorch', verbose=True)
        print(f"\n✅ SUCCESS! PyTorch backend worked")
        print(f"  Backend used: {result.backend}")
        print(f"  GPU accelerated: {result.gpu_accelerated}")
    except Exception as e:
        print(f"\n❌ FAILED with {type(e).__name__}: {e}")
    
    # Test 3: Try JAX backend
    print("\n" + "-" * 50)
    print("TEST 3: JAX backend (backend='jax')")
    print("-" * 50)
    try:
        result = mlest(data, backend='jax', verbose=True)
        print(f"\n✅ SUCCESS! JAX backend worked")
        print(f"  Backend used: {result.backend}")
    except Exception as e:
        print(f"\n❌ FAILED with {type(e).__name__}: {e}")
    
    # Test 4: Check what backends are available
    print("\n" + "-" * 50)
    print("TEST 4: Available backends check")
    print("-" * 50)
    try:
        from pymvnmle._backends import get_available_backends
        available = get_available_backends()
        print(f"Available backends: {available}")
        
        # Try to import GPU backends directly
        print("\nDirect import tests:")
        
        try:
            import torch
            print(f"  ✓ PyTorch {torch.__version__} available")
            if torch.cuda.is_available():
                print(f"    - CUDA available: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"    - Metal/MPS available")
            else:
                print(f"    - No GPU acceleration available in PyTorch")
        except ImportError:
            print("  ✗ PyTorch not installed")
        
        try:
            import jax
            print(f"  ✓ JAX {jax.__version__} available")
            print(f"    - Devices: {jax.devices()}")
        except ImportError:
            print("  ✗ JAX not installed")
            
    except Exception as e:
        print(f"Failed to check backends: {e}")
    
    print("\n" + "=" * 60)
    print("GPU TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_backend()