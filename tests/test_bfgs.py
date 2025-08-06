#!/usr/bin/env python3
"""
Test what the BFGS optimizer is actually returning.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import datasets
from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
from pymvnmle._methods import BFGSOptimizer


def test_bfgs_return_value():
    """Test what BFGS returns vs scipy."""
    print("=" * 70)
    print("TESTING BFGS OPTIMIZER RETURN VALUE")
    print("=" * 70)
    
    # Load Apple data
    data = datasets.apple
    obj = CPUObjectiveFP64(data)
    initial = obj.get_initial_parameters()
    
    print(f"\n1. INITIAL VALUES")
    print("-" * 40)
    print(f"Initial parameters shape: {initial.shape}")
    initial_obj = obj.compute_objective(initial)
    print(f"Initial objective: {initial_obj}")
    print(f"Initial log-likelihood: {-initial_obj/2}")
    
    print(f"\n2. PYMVNMLE BFGS OPTIMIZER")
    print("-" * 40)
    
    # Create BFGS optimizer
    optimizer = BFGSOptimizer(
        max_iter=1000,
        gtol=1e-6,
        ftol=1e-9,
        step_size_init=1.0,
        armijo_c1=1e-4,
        wolfe_c2=0.9,
        max_line_search=20,
        verbose=True
    )
    
    # Run optimization
    result = optimizer.optimize(
        objective_fn=obj.compute_objective,
        gradient_fn=obj.compute_gradient,
        x0=initial
    )
    
    print(f"\nBFGS Result:")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_iter']}")
    print(f"  result['fun']: {result['fun']}")
    print(f"  Log-likelihood from result['fun']: {-result['fun']/2}")
    
    # Manually compute objective at final point
    final_obj = obj.compute_objective(result['x'])
    print(f"\n  Actual objective at result['x']: {final_obj}")
    print(f"  Actual log-likelihood: {-final_obj/2}")
    
    print(f"\n  DISCREPANCY: result['fun'] = {result['fun']}, actual = {final_obj}")
    
    print(f"\n3. SCIPY COMPARISON")
    print("-" * 40)
    from scipy.optimize import minimize
    
    scipy_result = minimize(
        fun=obj.compute_objective,
        x0=initial,
        method='BFGS',
        options={'gtol': 1e-6, 'maxiter': 1000}
    )
    
    print(f"Scipy Result:")
    print(f"  Converged: {scipy_result.success}")
    print(f"  Iterations: {scipy_result.nit}")
    print(f"  scipy_result.fun: {scipy_result.fun}")
    print(f"  Log-likelihood: {-scipy_result.fun/2}")
    
    # Verify objective at scipy's final point
    scipy_final_obj = obj.compute_objective(scipy_result.x)
    print(f"\n  Actual objective at scipy's x: {scipy_final_obj}")
    print(f"  Match: {np.abs(scipy_result.fun - scipy_final_obj) < 1e-10}")
    
    print(f"\n4. PARAMETER COMPARISON")
    print("-" * 40)
    param_diff = np.max(np.abs(result['x'] - scipy_result.x))
    print(f"Max parameter difference: {param_diff}")
    
    print(f"\n5. DIAGNOSIS")
    print("-" * 40)
    if np.abs(result['fun'] - final_obj) > 1e-10:
        print("❌ BUG FOUND: BFGS optimizer returns wrong objective value!")
        print("   result['fun'] doesn't match objective at result['x']")
    else:
        print("✓ BFGS optimizer returns correct objective value")
    
    return result, scipy_result


if __name__ == "__main__":
    bfgs_result, scipy_result = test_bfgs_return_value()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    print(f"Expected log-likelihood: -74.217476")
    print(f"BFGS returns: {-bfgs_result['fun']/2}")
    print(f"Scipy returns: {-scipy_result.fun/2}")