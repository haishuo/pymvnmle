#!/usr/bin/env python3
"""
Quick test for missvals dataset convergence issue
Run this to debug without waiting for full test suite
"""

import numpy as np
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import mlest, datasets

def load_r_reference(filename):
    """Load R reference results."""
    ref_path = Path(__file__).parent / "references" / filename
    with open(ref_path, 'r') as f:
        return json.load(f)

def test_missvals_debug():
    """Debug test for missvals convergence issue."""
    print("=" * 60)
    print("MISSVALS DATASET DEBUG TEST")
    print("=" * 60)
    
    # Load R reference
    r_ref = load_r_reference('missvals_reference.json')
    print(f"\nR Reference:")
    print(f"  Log-likelihood: {r_ref['loglik']:.12f}")
    print(f"  Iterations: {r_ref['iterations']}")
    print(f"  Converged: {r_ref.get('converged', 'Unknown')}")
    
    # Test with method='auto' (what the failing test uses)
    print(f"\nTest 1: method='auto' (failing case)")
    print("-" * 40)
    try:
        result = mlest(datasets.missvals, method='auto', backend='auto', max_iter=400, verbose=False)
        
        print(f"\nResults:")
        print(f"  Log-likelihood: {result.loglik:.12f}")
        print(f"  R log-likelihood: {r_ref['loglik']:.12f}")
        print(f"  Difference: {abs(result.loglik - r_ref['loglik']):.2e}")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.n_iter}")
        print(f"  Method used: {result.method}")
        print(f"  Backend used: {result.backend}")
        
        if hasattr(result, 'gradient') and result.gradient is not None:
            grad_norm = np.linalg.norm(result.gradient)
            print(f"  Final gradient norm: {grad_norm:.2e}")
        
        # Check both conditions
        loglik_pass = abs(result.loglik - r_ref['loglik']) < 1e-6
        conv_pass = result.converged
        
        print(f"\n  Log-likelihood test: {'PASS' if loglik_pass else 'FAIL'}")
        print(f"  Convergence test: {'PASS' if conv_pass else 'FAIL'}")
        print(f"  Overall: {'PASS' if (loglik_pass and conv_pass) else 'FAIL'}")
        
    except Exception as e:
        print(f"  FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with method='BFGS' (should work)
    print(f"\n\nTest 2: method='BFGS' (expected to work)")
    print("-" * 40)
    try:
        result = mlest(datasets.missvals, method='BFGS', backend='auto', max_iter=400, verbose=False)
        
        print(f"Results:")
        print(f"  Log-likelihood: {result.loglik:.12f}")
        print(f"  Difference: {abs(result.loglik - r_ref['loglik']):.2e}")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.n_iter}")
        
        loglik_pass = abs(result.loglik - r_ref['loglik']) < 1e-6
        conv_pass = result.converged
        
        print(f"\n  Log-likelihood test: {'PASS' if loglik_pass else 'FAIL'}")
        print(f"  Convergence test: {'PASS' if conv_pass else 'FAIL'}")
        print(f"  Overall: {'PASS' if (loglik_pass and conv_pass) else 'FAIL'}")
        
    except Exception as e:
        print(f"  FAILED WITH ERROR: {e}")
    
    # Quick test on Apple dataset to ensure we didn't break that
    print(f"\n\nTest 3: Apple dataset sanity check")
    print("-" * 40)
    try:
        result = mlest(datasets.apple, method='auto', verbose=False)
        r_ref_apple = load_r_reference('apple_reference.json')
        
        loglik_diff = abs(result.loglik - r_ref_apple['loglik'])
        print(f"  Log-likelihood difference: {loglik_diff:.2e}")
        print(f"  Converged: {result.converged}")
        print(f"  Result: {'PASS' if loglik_diff < 1e-7 else 'FAIL'}")
        
    except Exception as e:
        print(f"  FAILED WITH ERROR: {e}")

if __name__ == "__main__":
    test_missvals_debug()