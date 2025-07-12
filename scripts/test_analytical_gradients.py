#!/usr/bin/env python3
"""
test_analytical_gradients.py - Test if our analytical gradients were actually correct

Quick test to see if the analytical gradients we implemented (and then abandoned)
were actually correct all along. We'll test:
1. Compare analytical vs finite differences at multiple points
2. Test optimization with BFGS using analytical gradients
3. Check convergence speed vs finite differences

Author: Biostatistician who might have been debugging the wrong thing
"""

import numpy as np
import time
from scipy.optimize import minimize
import sys
sys.path.append('.')

from objective_function import MVNMLEObjective
from end_to_end_validation import create_test_datasets
from analytical_gradients import compute_analytical_gradients, PatternData


def test_gradient_accuracy(data: np.ndarray, n_test_points: int = 10):
    """
    Test analytical gradients vs finite differences at random points.
    """
    print("Testing gradient accuracy at random points...")
    print("=" * 60)
    
    # Create objective function
    obj = MVNMLEObjective(data)
    
    # Convert to PatternData format for analytical gradients
    patterns = []
    for pattern_obj in obj.patterns:
        pattern = PatternData(
            observed_indices=pattern_obj.observed_indices,
            n_k=pattern_obj.n_k,
            data_k=pattern_obj.data_k
        )
        patterns.append(pattern)
    
    # Test at multiple random points
    np.random.seed(42)
    max_rel_error = 0.0
    
    for i in range(n_test_points):
        # Generate random parameters
        theta = obj.get_starting_values()
        theta += 0.1 * np.random.randn(len(theta))  # Perturb from starting values
        
        # Compute finite difference gradient
        grad_fd = obj.gradient(theta)
        
        # Compute analytical gradient
        try:
            grad_analytical = compute_analytical_gradients(theta, patterns)
        except Exception as e:
            print(f"Analytical gradient failed: {e}")
            return False
        
        # Compare
        diff = np.abs(grad_analytical - grad_fd)
        rel_error = diff / (np.abs(grad_fd) + 1e-10)
        max_rel_error = max(max_rel_error, np.max(rel_error))
        
        print(f"Point {i+1}: max absolute diff = {np.max(diff):.2e}, "
              f"max relative error = {np.max(rel_error):.2e}")
        
        # Check if they're close
        if not np.allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-8):
            print(f"  MISMATCH at point {i+1}!")
            print(f"  Analytical: {grad_analytical[:5]}...")
            print(f"  Finite diff: {grad_fd[:5]}...")
            return False
    
    print(f"\nOverall max relative error: {max_rel_error:.2e}")
    return max_rel_error < 1e-4


def test_optimization_with_analytical(data: np.ndarray):
    """
    Test optimization using analytical gradients with different methods.
    """
    print("\nTesting optimization with analytical gradients...")
    print("=" * 60)
    
    # Create objective function
    obj = MVNMLEObjective(data)
    
    # Convert patterns for analytical gradient
    patterns = []
    for pattern_obj in obj.patterns:
        pattern = PatternData(
            observed_indices=pattern_obj.observed_indices,
            n_k=pattern_obj.n_k,
            data_k=pattern_obj.data_k
        )
        patterns.append(pattern)
    
    # Create gradient function
    def grad_func(theta):
        return compute_analytical_gradients(theta, patterns)
    
    # Get starting values
    theta0 = obj.get_starting_values()
    
    # Test different optimizers
    methods = ['Newton-CG', 'BFGS', 'L-BFGS-B']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} with analytical gradients...")
        
        try:
            start_time = time.time()
            result = minimize(
                fun=obj,
                x0=theta0,
                method=method,
                jac=grad_func,
                options={'maxiter': 1000, 'disp': False}
            )
            elapsed = time.time() - start_time
            
            # Extract mean estimates
            muhat = result.x[:obj.n_vars]
            
            results[method] = {
                'success': result.success,
                'iterations': result.nit,
                'time': elapsed,
                'muhat': muhat,
                'fun': result.fun,
                'message': result.message
            }
            
            print(f"  Converged: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Objective: {result.fun:.6f}")
            print(f"  Mean estimates: {muhat}")
            
        except Exception as e:
            print(f"  Failed with error: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    return results


def compare_gradient_methods(data: np.ndarray):
    """
    Compare convergence speed: analytical vs finite difference gradients.
    """
    print("\nComparing analytical vs finite difference gradients...")
    print("=" * 60)
    
    # Create objective function
    obj = MVNMLEObjective(data)
    theta0 = obj.get_starting_values()
    
    # Setup analytical gradients
    patterns = []
    for pattern_obj in obj.patterns:
        pattern = PatternData(
            observed_indices=pattern_obj.observed_indices,
            n_k=pattern_obj.n_k,
            data_k=pattern_obj.data_k
        )
        patterns.append(pattern)
    
    def grad_analytical(theta):
        return compute_analytical_gradients(theta, patterns)
    
    # Test BFGS with both gradient types
    print("\nBFGS with finite differences:")
    start = time.time()
    result_fd = minimize(
        fun=obj,
        x0=theta0,
        method='BFGS',
        jac=obj.gradient,  # Finite differences
        options={'maxiter': 1000, 'disp': False}
    )
    time_fd = time.time() - start
    
    print(f"  Iterations: {result_fd.nit}")
    print(f"  Time: {time_fd:.3f}s")
    print(f"  Success: {result_fd.success}")
    
    print("\nBFGS with analytical gradients:")
    start = time.time()
    result_analytical = minimize(
        fun=obj,
        x0=theta0,
        method='BFGS',
        jac=grad_analytical,
        options={'maxiter': 1000, 'disp': False}
    )
    time_analytical = time.time() - start
    
    print(f"  Iterations: {result_analytical.nit}")
    print(f"  Time: {time_analytical:.3f}s")
    print(f"  Success: {result_analytical.success}")
    
    # Compare results
    print(f"\nSpeedup: {time_fd/time_analytical:.2f}x")
    print(f"Same result? {np.allclose(result_fd.x, result_analytical.x, rtol=1e-6)}")


def main():
    """Run all analytical gradient tests."""
    print("Testing Analytical Gradients")
    print("=" * 70)
    print("Were our analytical gradients correct all along?")
    print("Let's find out...\n")
    
    # Load test data
    datasets = create_test_datasets()
    apple_data = datasets['apple']
    
    # Test 1: Gradient accuracy
    print("\nTEST 1: Gradient Accuracy")
    gradients_accurate = test_gradient_accuracy(apple_data)
    
    if not gradients_accurate:
        print("\n❌ Analytical gradients don't match finite differences!")
        print("They really were wrong. Mystery solved.")
        return
    
    print("\n✅ Analytical gradients match finite differences!")
    
    # Test 2: Optimization with analytical gradients
    print("\nTEST 2: Optimization Performance")
    results = test_optimization_with_analytical(apple_data)
    
    # Check if Newton-CG works now
    if results.get('Newton-CG', {}).get('success'):
        print("\n🎉 Newton-CG WORKS with analytical gradients!")
        print("We were debugging the wrong thing all along!")
    else:
        print("\n🤔 Newton-CG still fails even with analytical gradients")
        print("The problem is deeper than just gradient accuracy")
    
    # Test 3: Speed comparison
    print("\nTEST 3: Speed Comparison")
    compare_gradient_methods(apple_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if gradients_accurate:
        print("✅ Analytical gradients were mathematically correct")
        
        if results.get('Newton-CG', {}).get('success'):
            print("✅ Newton-CG works with analytical gradients")
            print("💡 We should use analytical gradients for better performance!")
        else:
            print("❌ Newton-CG fails even with correct gradients")
            print("💡 Newton-CG just doesn't like this problem")
        
        if results.get('BFGS', {}).get('success'):
            print("✅ BFGS works with analytical gradients")
            print("💡 Could get significant speedup using them")
    else:
        print("❌ Analytical gradients were actually wrong")
        print("💡 Good thing we switched to finite differences")


if __name__ == "__main__":
    main()