"""
Test what actually matters: Do we get the same statistical results?
"""

import numpy as np
from scipy.optimize import minimize
from pymvnmle._objectives import get_objective
from pymvnmle import datasets

# Test on a real dataset
print("🧪 Testing Statistical Equivalence: CPU vs GPU")
print("=" * 60)

# Use the apple dataset (small but real)
data = datasets.apple
n_obs, n_vars = data.shape
n_params = n_vars + (n_vars * (n_vars + 1)) // 2

print(f"\n📊 Dataset: Apple ({n_obs} observations, {n_vars} variables)")
print(f"Parameters to estimate: {n_params}")

# Create objectives
numpy_obj = get_objective(data, backend='numpy')
torch_obj = get_objective(data, backend='pytorch')

# Starting values (same for both)
np.random.seed(42)
theta0 = np.random.randn(n_params)

print(f"\n🎯 Starting objective values:")
print(f"NumPy: {numpy_obj(theta0):.6f}")
print(f"PyTorch: {torch_obj(theta0):.6f}")

# Optimize with NumPy backend
print(f"\n⚙️ Optimizing with NumPy backend...")
result_numpy = minimize(
    numpy_obj,
    theta0,
    method='BFGS',
    options={'maxiter': 1000, 'gtol': 1e-6}
)
print(f"Converged: {result_numpy.success} in {result_numpy.nit} iterations")
print(f"Final objective: {result_numpy.fun:.6f}")

# Extract estimates
mu_numpy, sigma_numpy, loglik_numpy = numpy_obj.extract_parameters(result_numpy.x)
print(f"\nNumPy estimates:")
print(f"μ = {mu_numpy}")
print(f"Σ = \n{sigma_numpy}")

# Optimize with PyTorch backend using autodiff gradients
print(f"\n⚙️ Optimizing with PyTorch backend (autodiff gradients)...")

def torch_obj_with_grad(theta):
    """Objective and gradient for scipy."""
    obj_val = torch_obj(theta)
    grad = torch_obj.gradient(theta)
    return obj_val, grad

result_torch = minimize(
    torch_obj_with_grad,
    theta0,
    method='BFGS',
    jac=True,  # We're providing gradients
    options={'maxiter': 1000, 'gtol': 1e-6}
)
print(f"Converged: {result_torch.success} in {result_torch.nit} iterations")
print(f"Final objective: {result_torch.fun:.6f}")

# Extract estimates
mu_torch, sigma_torch, loglik_torch = torch_obj.extract_parameters(result_torch.x)
print(f"\nPyTorch estimates:")
print(f"μ = {mu_torch}")
print(f"Σ = \n{sigma_torch}")

# Compare results
print(f"\n📈 STATISTICAL COMPARISON:")
print(f"-" * 40)

# Mean comparison
mu_diff = np.max(np.abs(mu_numpy - mu_torch))
mu_rel_diff = mu_diff / (np.max(np.abs(mu_numpy)) + 1e-10)
print(f"Mean difference: {mu_diff:.6f} (relative: {mu_rel_diff:.2%})")

# Covariance comparison
sigma_diff = np.max(np.abs(sigma_numpy - sigma_torch))
sigma_rel_diff = sigma_diff / (np.max(np.abs(sigma_numpy)) + 1e-10)
print(f"Covariance difference: {sigma_diff:.6f} (relative: {sigma_rel_diff:.2%})")

# Log-likelihood comparison
loglik_diff = abs(loglik_numpy - loglik_torch)
print(f"Log-likelihood difference: {loglik_diff:.6f}")

# Statistical significance test
# If estimates are within 0.1% relative error, they're statistically equivalent
tolerance = 0.001  # 0.1%

if mu_rel_diff < tolerance and sigma_rel_diff < tolerance:
    print(f"\n✅ SUCCESS: Estimates are statistically equivalent!")
    print(f"   Both methods converged to the same solution.")
    print(f"   Autodiff gradients are working correctly for optimization!")
else:
    print(f"\n❌ FAILURE: Estimates differ significantly.")
    print(f"   This indicates a problem with the gradient computation.")

# Performance comparison
print(f"\n⏱️ PERFORMANCE:")
print(f"NumPy iterations: {result_numpy.nit}")
print(f"PyTorch iterations: {result_torch.nit}")

if result_torch.nit < result_numpy.nit:
    print(f"🚀 PyTorch converged {result_numpy.nit - result_torch.nit} iterations faster!")
elif result_torch.nit > result_numpy.nit:
    print(f"🐌 PyTorch took {result_torch.nit - result_numpy.nit} more iterations.")
else:
    print(f"🤝 Both methods took the same number of iterations.")