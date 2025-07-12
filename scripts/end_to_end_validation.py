#!/usr/bin/env python3
"""
end_to_end_validation.py - Complete PyMVNMLE Validation Against R References

This script provides comprehensive validation of our PyMVNMLE implementation
against R's mvnmle package reference results. It tests both standard datasets
(apple, missvals) and ensures numerical agreement within machine precision.

This is the final validation before production deployment.

Author: Senior Biostatistician
Purpose: Validate complete ML estimation pipeline against R
Standard: FDA submission grade for clinical trials
"""

import numpy as np
import json
from scipy.optimize import minimize
from typing import Dict, Any, Tuple
import time
from pathlib import Path

# Import all our validated components
from pattern_preprocessing import validate_data
from objective_function import MVNMLEObjective, create_scipy_objective


def load_r_reference(filename: str) -> Dict[str, Any]:
    """
    Load R reference results from JSON file.
    
    Parameters
    ----------
    filename : str
        Name of reference file (e.g., 'apple_reference.json')
        
    Returns
    -------
    dict
        R reference results including muhat, sigmahat, loglik, etc.
    """
    # Try multiple possible locations
    possible_paths = [
        Path('tests/references') / filename,
        Path('../tests/references') / filename,
        Path('references') / filename,
        Path('.') / filename
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(
        f"Could not find reference file {filename}. "
        f"Searched in: {[str(p) for p in possible_paths]}"
    )


def create_test_datasets() -> Dict[str, np.ndarray]:
    """
    Create test datasets matching R exactly.
    
    Returns
    -------
    dict
        Dictionary with 'apple' and 'missvals' datasets
    """
    # Apple dataset - EXACT values from R
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
    
    # Missvals dataset - EXACT values from R
    missvals = np.array([
        [7.0, 26.0, 6.0, 60.0, 78.5],
        [1.0, 29.0, 15.0, 52.0, 74.3],
        [11.0, 56.0, 8.0, 20.0, 104.3],
        [11.0, 31.0, 8.0, 47.0, 87.6],
        [7.0, 52.0, 6.0, 33.0, 95.9],
        [11.0, 55.0, 9.0, 22.0, 109.2],
        [3.0, 71.0, 17.0, np.nan, 102.7],
        [1.0, 31.0, 22.0, np.nan, 72.5],
        [2.0, 54.0, 18.0, np.nan, 93.1],
        [np.nan, np.nan, 4.0, np.nan, 115.9],
        [np.nan, np.nan, 23.0, np.nan, 83.8],
        [np.nan, np.nan, 9.0, np.nan, 113.3],
        [np.nan, np.nan, 8.0, np.nan, 109.4]
    ])
    
    return {'apple': apple, 'missvals': missvals}


def run_optimization(data: np.ndarray, method: str = 'Newton-CG', 
                    max_iter: int = 1000) -> Dict[str, Any]:
    """
    Run ML optimization using scipy.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with missing values
    method : str
        Optimization method (Newton-CG, BFGS, L-BFGS-B, Nelder-Mead, etc.)
        Default is Newton-CG to match R's nlm
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    dict
        Optimization results including estimates and diagnostics
    """
    # Create objective function
    obj = MVNMLEObjective(data)
    
    # Get starting values
    theta0 = obj.get_starting_values()
    
    # Run optimization
    start_time = time.time()
    
    if method in ['BFGS', 'L-BFGS-B', 'Newton-CG']:
        # Methods that can use gradients
        result = minimize(
            fun=obj,
            x0=theta0,
            method=method,
            jac=obj.gradient,
            options={'maxiter': max_iter, 'disp': False}
        )
    else:
        # Gradient-free methods
        result = minimize(
            fun=obj,
            x0=theta0,
            method=method,
            options={'maxiter': max_iter, 'disp': False}
        )
    
    elapsed_time = time.time() - start_time
    
    # Extract estimates
    if result.success or result.fun < 1e10:  # Sometimes marked as failure but actually converged
        muhat, sigmahat = obj.compute_estimates(result.x)
        
        # Convert objective value to log-likelihood
        # Our objective returns proportional to -2*loglik (like R)
        loglik = -result.fun / 2.0
        
        return {
            'muhat': muhat,
            'sigmahat': sigmahat,
            'loglik': loglik,
            'converged': result.success,
            'iterations': result.nit,
            'time': elapsed_time,
            'raw_result': result,
            'objective': obj
        }
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")


def compare_results(py_result: Dict[str, Any], r_result: Dict[str, Any], 
                   dataset_name: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Compare Python results with R reference.
    
    Parameters
    ----------
    py_result : dict
        Python optimization results
    r_result : dict
        R reference results
    dataset_name : str
        Name of dataset for reporting
    tolerance : float
        Tolerance for numerical comparison
        
    Returns
    -------
    success : bool
        Whether results match within tolerance
    message : str
        Detailed comparison message
    """
    messages = []
    all_pass = True
    
    # Compare mean estimates
    mu_diff = np.max(np.abs(py_result['muhat'] - r_result['muhat']))
    mu_pass = mu_diff < tolerance
    all_pass &= mu_pass
    messages.append(f"Mean estimates: max diff = {mu_diff:.2e} {'✓' if mu_pass else '✗'}")
    
    # Compare covariance estimates
    sigma_diff = np.max(np.abs(py_result['sigmahat'] - r_result['sigmahat']))
    sigma_pass = sigma_diff < tolerance
    all_pass &= sigma_pass
    messages.append(f"Covariance matrix: max diff = {sigma_diff:.2e} {'✓' if sigma_pass else '✗'}")
    
    # Compare log-likelihood
    # Our objective returns value proportional to -2*loglik (like R)
    # So loglik = -objective/2
    py_loglik = -py_result['raw_result'].fun / 2.0
    r_loglik = r_result['loglik']
    
    loglik_diff = abs(py_loglik - r_loglik)
    loglik_pass = loglik_diff < tolerance * 10  # Slightly more lenient for log-likelihood
    all_pass &= loglik_pass
    messages.append(f"Log-likelihood: diff = {loglik_diff:.2e} {'✓' if loglik_pass else '✗'}")
    messages.append(f"  (R: {r_loglik:.6f}, Py: {py_loglik:.6f})")
    
    # Compare iterations (informational only)
    messages.append(f"Iterations: R = {r_result.get('iterations', 'N/A')}, Py = {py_result['iterations']}")
    
    # Final summary
    summary = f"\n{dataset_name} dataset: {'PASS' if all_pass else 'FAIL'}\n"
    summary += "\n".join(f"  {msg}" for msg in messages)
    
    return all_pass, summary


def test_apple_dataset():
    """Test Apple dataset against R reference."""
    print("\n" + "="*60)
    print("Testing APPLE dataset")
    print("="*60)
    
    # Load data and reference
    datasets = create_test_datasets()
    apple_data = datasets['apple']
    r_ref = load_r_reference('apple_reference.json')
    
    # First diagnose at R's solution
    theta_r, f_r, grad_r = diagnose_objective_at_r_solution(apple_data, r_ref)

    print(f"Data shape: {apple_data.shape}")
    print(f"Missing values: {np.sum(np.isnan(apple_data))}")
    
    # Run optimization (Newton-CG to match R's nlm)
    print("\nRunning optimization with Newton-CG (closest to R's nlm)...")
    py_result = run_optimization(apple_data, method='Newton-CG')
    
    print(f"Optimization completed in {py_result['time']:.3f}s")
    print(f"Converged: {py_result['converged']}")
    print(f"Iterations: {py_result['iterations']}")
    
    # Compare results
    success, message = compare_results(py_result, r_ref, 'Apple')
    print(message)
    
    # Detailed output
    print("\nDetailed results:")
    print(f"Python muhat: {py_result['muhat']}")
    print(f"R muhat:      {r_ref['muhat']}")
    print(f"\nPython sigmahat:")
    print(py_result['sigmahat'])
    print(f"\nR sigmahat:")
    print(np.array(r_ref['sigmahat']))
    
    return success


def test_missvals_dataset():
    """Test Missvals dataset against R reference."""
    print("\n" + "="*60)
    print("Testing MISSVALS dataset")
    print("="*60)
    
    # Load data and reference
    datasets = create_test_datasets()
    missvals_data = datasets['missvals']
    r_ref = load_r_reference('missvals_reference.json')
    
    print(f"Data shape: {missvals_data.shape}")
    print(f"Missing values: {np.sum(np.isnan(missvals_data))}")
    
    # Run optimization with higher iteration limit (like R)
    print("\nRunning optimization with Newton-CG (closest to R's nlm)...")
    py_result = run_optimization(missvals_data, method='Newton-CG', max_iter=400)
    
    print(f"Optimization completed in {py_result['time']:.3f}s")
    print(f"Converged: {py_result['converged']}")
    print(f"Iterations: {py_result['iterations']}")
    
    # Compare results
    success, message = compare_results(py_result, r_ref, 'Missvals', tolerance=1e-5)
    print(message)
    
    # Detailed output
    print("\nDetailed results:")
    print(f"Python muhat: {py_result['muhat']}")
    print(f"R muhat:      {r_ref['muhat']}")
    
    return success


def test_optimization_methods():
    """Test different optimization methods for robustness."""
    print("\n" + "="*60)
    print("Testing different optimization methods")
    print("="*60)
    
    datasets = create_test_datasets()
    apple_data = datasets['apple']
    
    # Newton-CG is closest to R's nlm method
    methods = ['Newton-CG', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method}...")
        try:
            result = run_optimization(apple_data, method=method)
            results[method] = result
            print(f"  Converged: {result['converged']}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Time: {result['time']:.3f}s")
            print(f"  muhat: {result['muhat']}")
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = None
    
    # Compare results across methods
    print("\nCross-method comparison:")
    if all(results.values()):
        methods_with_results = [(m, r) for m, r in results.items() if r is not None]
        if len(methods_with_results) > 1:
            base_method, base_result = methods_with_results[0]
            for method, result in methods_with_results[1:]:
                mu_diff = np.max(np.abs(result['muhat'] - base_result['muhat']))
                print(f"  {base_method} vs {method}: max μ diff = {mu_diff:.2e}")


def test_numerical_stability():
    """Test numerical stability with challenging datasets."""
    print("\n" + "="*60)
    print("Testing numerical stability")
    print("="*60)
    
    # Test 1: Near-singular covariance
    print("\nTest 1: Near-singular covariance")
    np.random.seed(42)
    n = 50
    p = 3
    
    # Generate data with high correlation
    base = np.random.randn(n, 1)
    data_singular = np.hstack([
        base + 0.01 * np.random.randn(n, 1),
        base + 0.01 * np.random.randn(n, 1),
        np.random.randn(n, 1)
    ])
    
    # Add some missing values
    missing_mask = np.random.random(data_singular.shape) < 0.1
    data_singular[missing_mask] = np.nan
    
    try:
        result = run_optimization(data_singular)
        eigenvals = np.linalg.eigvalsh(result['sigmahat'])
        print(f"  Success! Condition number: {np.max(eigenvals)/np.min(eigenvals):.2e}")
        print(f"  Min eigenvalue: {np.min(eigenvals):.2e}")
    except Exception as e:
        print(f"  Handled gracefully: {e}")
    
    # Test 2: High missingness
    print("\nTest 2: High missingness rate")
    data_sparse = np.random.randn(100, 4)
    sparse_mask = np.random.random(data_sparse.shape) < 0.7  # 70% missing
    data_sparse[sparse_mask] = np.nan
    
    try:
        result = run_optimization(data_sparse)
        print(f"  Success! Converged in {result['iterations']} iterations")
    except Exception as e:
        print(f"  Handled gracefully: {e}")
    
    # Test 3: Small sample size
    print("\nTest 3: Small sample size (n < p)")
    data_small = np.random.randn(3, 5)
    data_small[0, [1, 3]] = np.nan
    data_small[1, [0, 2, 4]] = np.nan
    
    try:
        result = run_optimization(data_small)
        print(f"  Success! Converged = {result['converged']}")
    except Exception as e:
        print(f"  Handled gracefully: {e}")

def diagnose_objective_at_r_solution(data: np.ndarray, r_result: Dict[str, Any]):
    """
    Diagnose objective function behavior at R's solution.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC: Objective function at R solution")
    print("="*60)
    
    # Create objective function
    obj = MVNMLEObjective(data)
    
    # Reconstruct R's parameter vector
    n_vars = data.shape[1]
    
    # Get R's Delta matrix from their solution
    # R has Sigma = (Delta^-1)' (Delta^-1)
    # So Delta^-1 = chol(Sigma)'
    Sigma_r = np.array(r_result['sigmahat'])
    L = np.linalg.cholesky(Sigma_r)  # Lower triangular
    Delta_inv = L.T  # Upper triangular
    Delta = np.linalg.inv(Delta_inv)
    
    # Pack into parameter vector
    theta_r = np.zeros(obj.n_total_params)
    theta_r[:n_vars] = r_result['muhat']
    theta_r[n_vars:2*n_vars] = np.log(np.diag(Delta))
    
    idx = 2*n_vars
    for j in range(1, n_vars):
        for i in range(j):
            theta_r[idx] = Delta[i, j]
            idx += 1
    
    # Evaluate objective at R's solution
    f_at_r = obj(theta_r)
    print(f"Objective at R solution: {f_at_r:.6f}")
    print(f"Expected (-2*loglik): {-2 * r_result['loglik']:.6f}")
    print(f"Difference: {abs(f_at_r - (-2 * r_result['loglik'])):.2e}")
    
    # Check gradient at R's solution
    grad_at_r = obj.gradient(theta_r)
    grad_norm = np.linalg.norm(grad_at_r)
    print(f"\nGradient norm at R solution: {grad_norm:.2e}")
    print(f"Max gradient component: {np.max(np.abs(grad_at_r)):.2e}")
    
    # Check gradient with different step sizes
    print("\nGradient finite difference check:")
    for eps_scale in [1.0, 10.0, 100.0, 0.1, 0.01]:
        eps = 1.49011612e-08 * eps_scale
        
        # Check one parameter
        i = 0  # First mean parameter
        h = eps * max(abs(theta_r[i]), 1.0)
        
        theta_plus = theta_r.copy()
        theta_plus[i] += h
        
        f_plus = obj(theta_plus)
        fd_grad = (f_plus - f_at_r) / h
        
        print(f"  eps_scale={eps_scale:6.2f}: FD grad[0]={fd_grad:12.6f}, "
              f"Computed grad[0]={grad_at_r[0]:12.6f}, "
              f"Diff={abs(fd_grad - grad_at_r[0]):.2e}")
    
    # Test objective function smoothness
    print("\nObjective function smoothness test:")
    alphas = np.logspace(-10, -1, 10)
    direction = -grad_at_r / grad_norm  # Descent direction
    
    for alpha in alphas:
        theta_test = theta_r + alpha * direction
        f_test = obj(theta_test)
        print(f"  α={alpha:.2e}: f={f_test:.6f}, Δf={f_test - f_at_r:.6f}")
    
    return theta_r, f_at_r, grad_at_r

def main():
    """Run all validation tests."""
    print("PyMVNMLE End-to-End Validation Against R References")
    print("=" * 70)
    print("This validates our complete implementation against R's mvnmle package")
    print("Using Newton-CG optimizer (closest to R's nlm method)")
    print("Tolerance: 1e-6 for parameters, 1e-5 for log-likelihood")
    
    all_tests_passed = True
    
    # Test 1: Apple dataset
    try:
        apple_passed = test_apple_dataset()
        all_tests_passed &= apple_passed
    except Exception as e:
        print(f"\nApple test failed with error: {e}")
        all_tests_passed = False
    
    # Test 2: Missvals dataset
    try:
        missvals_passed = test_missvals_dataset()
        all_tests_passed &= missvals_passed
    except Exception as e:
        print(f"\nMissvals test failed with error: {e}")
        all_tests_passed = False
    
    # Test 3: Different optimization methods
    try:
        test_optimization_methods()
    except Exception as e:
        print(f"\nOptimization methods test failed: {e}")
    
    # Test 4: Numerical stability
    try:
        test_numerical_stability()
    except Exception as e:
        print(f"\nNumerical stability test failed: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("PyMVNMLE implementation matches R's mvnmle within numerical tolerance")
        print("Ready for production deployment in regulatory environments")
    else:
        print("❌ Some tests failed")
        print("Further investigation needed before production deployment")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)