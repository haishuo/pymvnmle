#!/usr/bin/env python3
"""
Test script for PyMVNMLE implementation
Quick validation against R references
"""

import numpy as np
import json
import sys
import traceback
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Try direct imports first
    from pymvnmle.mlest import mlest
    print("✓ Successfully imported mlest")
    
    # For now, create datasets directly using the CORRECT R data
    apple = np.array([
        [8.0, 59.0],
        [6.0, 58.0], 
        [11.0, 56.0],
        [22.0, 53.0],
        [14.0, 50.0],
        [17.0, 45.0],
        [18.0, 43.0],
        [24.0, 42.0],
        [19.0, 39.0],
        [23.0, 38.0],
        [26.0, 30.0],
        [40.0, 27.0],
        [4.0, np.nan],
        [4.0, np.nan],
        [5.0, np.nan],
        [6.0, np.nan],
        [8.0, np.nan],
        [10.0, np.nan]
    ])
    
    missvals = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.1, 3.1, np.nan, 5.1],
        [1.2, np.nan, 3.2, 4.2, 5.2],
        [np.nan, 2.3, 3.3, 4.3, 5.3],
        [1.4, 2.4, np.nan, 4.4, 5.4],
        [1.5, 2.5, 3.5, 4.5, np.nan],
        [np.nan, np.nan, 3.6, 4.6, 5.6],
        [1.7, 2.7, 3.7, np.nan, np.nan],
        [1.8, np.nan, np.nan, 4.8, 5.8],
        [np.nan, 2.9, 3.9, 4.9, 5.9],
        [1.10, 2.10, 3.10, 4.10, 5.10],
        [1.11, np.nan, 3.11, 4.11, np.nan],
        [np.nan, 2.12, np.nan, 4.12, 5.12]
    ])
    
    print("✓ Successfully created test datasets")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Available files:")
    for f in project_root.glob("**/*.py"):
        print(f"  {f}")
    sys.exit(1)


def load_r_reference(filename):
    """Load R reference results from JSON."""
    ref_path = Path("tests/references") / filename
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    
    with open(ref_path) as f:
        return json.load(f)


def test_apple_dataset():
    """Test against R reference: mlest(apple)"""
    print("🍎 Testing Apple Dataset...")
    
    try:
        # Load reference
        r_ref = load_r_reference("apple_reference.json")
        
        # Python implementation
        result = mlest(apple, verbose=True)
        
        # Compare results
        print("\n📊 Results Comparison:")
        print(f"Mean estimates:")
        print(f"  Python: {result.muhat}")
        print(f"  R ref:  {r_ref['muhat']}")
        print(f"  Diff:   {np.abs(result.muhat - r_ref['muhat'])}")
        
        print(f"\nCovariance matrix:")
        print(f"  Python diagonal: {np.diag(result.sigmahat)}")
        print(f"  R ref diagonal:  {np.diag(r_ref['sigmahat'])}")
        print(f"  Diff diagonal:   {np.abs(np.diag(result.sigmahat) - np.diag(r_ref['sigmahat']))}")
        
        print(f"\nLog-likelihood:")
        print(f"  Python: {result.loglik:.6f}")
        print(f"  R ref:  {r_ref['loglik']:.6f}")
        print(f"  Diff:   {abs(result.loglik - r_ref['loglik']):.2e}")
        
        # Numerical validation
        mu_close = np.allclose(result.muhat, r_ref['muhat'], rtol=1e-10)
        sigma_close = np.allclose(result.sigmahat, r_ref['sigmahat'], rtol=1e-10)
        loglik_close = np.isclose(result.loglik, r_ref['loglik'], rtol=1e-10)
        
        print(f"\n✓ Validation Results:")
        print(f"  Mean estimates match R: {mu_close}")
        print(f"  Covariance matrix matches R: {sigma_close}")
        print(f"  Log-likelihood matches R: {loglik_close}")
        
        if mu_close and sigma_close and loglik_close:
            print("🎉 APPLE TEST PASSED!")
            return True
        else:
            print("❌ APPLE TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Apple test failed with error: {e}")
        traceback.print_exc()
        return False


def test_missvals_dataset():
    """Test against R reference: mlest(missvals, iterlim=400)"""
    print("\n📊 Testing Missvals Dataset...")
    
    try:
        # Load reference
        r_ref = load_r_reference("missvals_reference.json")
        
        # Python implementation (with higher iteration limit like R)
        result = mlest(missvals, max_iter=400, verbose=True)
        
        # Compare results
        print("\n📊 Results Comparison:")
        print(f"Mean estimates:")
        print(f"  Python: {result.muhat}")
        print(f"  R ref:  {r_ref['muhat']}")
        print(f"  Diff:   {np.abs(result.muhat - r_ref['muhat'])}")
        
        print(f"\nLog-likelihood:")
        print(f"  Python: {result.loglik:.6f}")
        print(f"  R ref:  {r_ref['loglik']:.6f}")
        print(f"  Diff:   {abs(result.loglik - r_ref['loglik']):.2e}")
        
        # Numerical validation (more lenient for this dataset)
        mu_close = np.allclose(result.muhat, r_ref['muhat'], rtol=1e-8)
        sigma_close = np.allclose(result.sigmahat, r_ref['sigmahat'], rtol=1e-8)
        loglik_close = np.isclose(result.loglik, r_ref['loglik'], rtol=1e-8)
        
        print(f"\n✓ Validation Results:")
        print(f"  Mean estimates match R: {mu_close}")
        print(f"  Covariance matrix matches R: {sigma_close}")
        print(f"  Log-likelihood matches R: {loglik_close}")
        
        if mu_close and sigma_close and loglik_close:
            print("🎉 MISSVALS TEST PASSED!")
            return True
        else:
            print("❌ MISSVALS TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Missvals test failed with error: {e}")
        traceback.print_exc()
        return False


def test_simple_case():
    """Test with a simple manually constructed case."""
    print("\n🔧 Testing Simple Case...")
    
    try:
        # Create simple test data
        np.random.seed(42)
        n_obs, n_vars = 20, 3
        
        # Generate some data with missing values
        true_mu = np.array([1.0, 2.0, 3.0])
        true_sigma = np.array([[1.0, 0.5, 0.2],
                              [0.5, 2.0, 0.3],
                              [0.2, 0.3, 1.5]])
        
        # Generate complete data
        data = np.random.multivariate_normal(true_mu, true_sigma, n_obs)
        
        # Introduce missing values
        missing_mask = np.random.random((n_obs, n_vars)) < 0.2
        data[missing_mask] = np.nan
        
        print(f"Generated data: {n_obs}×{n_vars}")
        print(f"Missing rate: {np.sum(missing_mask) / (n_obs * n_vars):.1%}")
        
        # Run estimation
        result = mlest(data, verbose=True)
        
        print(f"\n📊 Results:")
        print(f"True mean:      {true_mu}")
        print(f"Estimated mean: {result.muhat}")
        print(f"Mean error:     {np.abs(result.muhat - true_mu)}")
        
        print(f"\nTrue covariance diagonal:      {np.diag(true_sigma)}")
        print(f"Estimated covariance diagonal: {np.diag(result.sigmahat)}")
        
        print(f"\nConverged: {result.converged}")
        print(f"Iterations: {result.n_iter}")
        print(f"Backend: {result.backend}")
        print(f"GPU accelerated: {result.gpu_accelerated}")
        
        # Basic sanity checks
        if result.converged and not np.any(np.isnan(result.muhat)):
            print("🎉 SIMPLE TEST PASSED!")
            return True
        else:
            print("❌ SIMPLE TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Simple test failed with error: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🚀 PyMVNMLE Implementation Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Apple dataset
    results.append(test_apple_dataset())
    
    # Test 2: Missvals dataset  
    results.append(test_missvals_dataset())
    
    # Test 3: Simple case
    results.append(test_simple_case())
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! PyMVNMLE implementation is working correctly!")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)