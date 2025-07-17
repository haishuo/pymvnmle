"""
Test if PyTorch gradients lead to convergence with proper optimization.
"""

import numpy as np
from scipy.optimize import minimize
from pymvnmle._objectives import get_objective
from pymvnmle import datasets

# Test function that uses line search
def test_gradient_descent_with_line_search():
    """Test if gradients are valid descent directions with appropriate step size."""
    
    print("🧪 Testing PyTorch Gradient Descent with Line Search")
    print("=" * 60)
    
    # Use apple dataset
    data = datasets.apple
    
    # Create objectives
    numpy_obj = get_objective(data, backend='numpy')
    torch_obj = get_objective(data, backend='pytorch')
    
    # Starting point
    np.random.seed(42)
    n_vars = data.shape[1]
    n_params = n_vars + (n_vars * (n_vars + 1)) // 2
    theta = np.random.randn(n_params)
    
    print(f"Starting objective: {numpy_obj(theta):.6f}")
    
    # Test gradient descent with line search
    print("\n📉 Gradient descent test (5 iterations):")
    
    for i in range(5):
        # Get gradient
        grad = torch_obj.gradient(theta)
        grad_norm = np.linalg.norm(grad)
        
        # Line search to find appropriate step size
        obj_current = numpy_obj(theta)
        
        # Try different step sizes
        for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            theta_new = theta - alpha * grad
            obj_new = numpy_obj(theta_new)
            
            if np.isfinite(obj_new) and obj_new < obj_current:
                theta = theta_new
                print(f"  Iter {i+1}: obj = {obj_new:.6f}, ||grad|| = {grad_norm:.2e}, step = {alpha:.0e}")
                break
        else:
            print(f"  Iter {i+1}: No valid step found, stopping")
            break
    
    print("\n" + "="*60)
    
    # Now test full optimization
    print("\n🎯 Full Optimization Test")
    print("="*60)
    
    # Reset starting point
    theta0 = np.random.randn(n_params) * 0.1  # Smaller initial values
    
    # Add bounds to prevent numerical issues
    bounds = []
    for i in range(n_params):
        if i < n_vars:
            # Mean parameters: unbounded
            bounds.append((None, None))
        elif i < 2 * n_vars:
            # Log-diagonal parameters: keep reasonable
            bounds.append((-10, 10))
        else:
            # Off-diagonal parameters: keep reasonable
            bounds.append((-50, 50))
    
    print("🔹 NumPy optimization:")
    result_numpy = minimize(
        numpy_obj,
        theta0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    print(f"  Converged: {result_numpy.success}")
    print(f"  Iterations: {result_numpy.nit}")
    print(f"  Final objective: {result_numpy.fun:.6f}")
    
    # Extract estimates
    mu_np, sigma_np, loglik_np = numpy_obj.extract_parameters(result_numpy.x)
    
    print("\n🔹 PyTorch optimization (with autodiff):")
    
    def torch_obj_and_grad(theta):
        """Combined objective and gradient for L-BFGS-B."""
        obj = torch_obj(theta)
        grad = torch_obj.gradient(theta)
        return obj, grad
    
    result_torch = minimize(
        torch_obj_and_grad,
        theta0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    print(f"  Converged: {result_torch.success}")
    print(f"  Iterations: {result_torch.nit}")
    print(f"  Final objective: {result_torch.fun:.6f}")
    
    # Extract estimates
    mu_torch, sigma_torch, loglik_torch = torch_obj.extract_parameters(result_torch.x)
    
    # Compare results
    print("\n📊 RESULTS COMPARISON:")
    print("-" * 40)
    
    print("\nMean estimates:")
    print(f"  NumPy:   {mu_np}")
    print(f"  PyTorch: {mu_torch}")
    print(f"  Max diff: {np.max(np.abs(mu_np - mu_torch)):.6f}")
    
    print("\nCovariance estimates:")
    print(f"  NumPy diagonal:   {np.diag(sigma_np)}")
    print(f"  PyTorch diagonal: {np.diag(sigma_torch)}")
    print(f"  Max diff: {np.max(np.abs(sigma_np - sigma_torch)):.6f}")
    
    print("\nLog-likelihood:")
    print(f"  NumPy:   {loglik_np:.6f}")
    print(f"  PyTorch: {loglik_torch:.6f}")
    print(f"  Diff: {abs(loglik_np - loglik_torch):.6f}")
    
    # Statistical equivalence test
    mu_rel_err = np.max(np.abs(mu_np - mu_torch)) / (np.max(np.abs(mu_np)) + 1e-10)
    sigma_rel_err = np.max(np.abs(sigma_np - sigma_torch)) / (np.max(np.abs(sigma_np)) + 1e-10)
    
    print(f"\n📈 Relative errors:")
    print(f"  Mean: {mu_rel_err:.2%}")
    print(f"  Covariance: {sigma_rel_err:.2%}")
    
    if mu_rel_err < 0.01 and sigma_rel_err < 0.01:
        print("\n✅ SUCCESS! Estimates are statistically equivalent (< 1% difference)")
        print("   PyTorch autodiff gradients work for optimization!")
        
        # Performance comparison
        if result_torch.nit < result_numpy.nit:
            print(f"\n🚀 BONUS: PyTorch converged {result_numpy.nit - result_torch.nit} iterations faster!")
        
        return True
    else:
        print("\n❌ Estimates differ by more than 1%")
        return False

# Run the test
if __name__ == "__main__":
    success = test_gradient_descent_with_line_search()
    
    if success:
        print("\n🎉 PyTorch autodiff implementation is working correctly!")
        print("   We have achieved analytical gradients for MLE with missing data!")
    else:
        print("\n🔧 More work needed on the gradient computation...")