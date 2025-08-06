#!/usr/bin/env python3
"""
Check if optimization is converging properly.
"""

import numpy as np
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generate test data
np.random.seed(42)
n = 500
p = 5

true_mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
true_sigma = np.eye(p) + 0.3 * np.ones((p, p))
data = np.random.multivariate_normal(true_mu, true_sigma, n)

# Add missing
mask = np.random.random((n, p)) < 0.05
data[mask] = np.nan

from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
from pymvnmle._objectives.gpu_fp32_objective import GPUObjectiveFP32

cpu_obj = CPUObjectiveFP64(data)
gpu_obj = GPUObjectiveFP32(data)

theta_cpu = cpu_obj.get_initial_parameters()
theta_gpu = gpu_obj.get_initial_parameters()

# Track optimization
class OptimizerMonitor:
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name
        self.history = []
        self.param_history = []
        
    def callback(self, theta):
        obj_val = self.obj.compute_objective(theta)
        mu, sigma, _ = self.obj.extract_parameters(theta)
        self.history.append(obj_val)
        self.param_history.append((mu.copy(), sigma.copy()))
        
        if len(self.history) % 5 == 0:
            print(f"  {self.name} iter {len(self.history)}: obj={obj_val:.2f}, "
                  f"Σ_max={np.max(np.abs(sigma)):.2f}")

# Run CPU optimization
print("CPU Optimization:")
cpu_monitor = OptimizerMonitor(cpu_obj, "CPU")
result_cpu = minimize(
    cpu_obj.compute_objective,
    theta_cpu,
    method='BFGS',
    jac=cpu_obj.compute_gradient,
    callback=cpu_monitor.callback,
    options={'maxiter': 30, 'gtol': 1e-5}
)

# Run GPU optimization
print("\nGPU Optimization:")
gpu_monitor = OptimizerMonitor(gpu_obj, "GPU")
result_gpu = minimize(
    gpu_obj.compute_objective,
    theta_gpu,
    method='L-BFGS-B',  # Try L-BFGS-B for better stability
    jac=gpu_obj.compute_gradient,
    callback=gpu_monitor.callback,
    options={'maxiter': 30, 'ftol': 1e-5, 'gtol': 1e-5}
)

# Compare final results
mu_cpu_final, sigma_cpu_final, _ = cpu_obj.extract_parameters(result_cpu.x)
mu_gpu_final, sigma_gpu_final, _ = gpu_obj.extract_parameters(result_gpu.x)

print(f"\nFinal objectives:")
print(f"  CPU: {result_cpu.fun:.4f} ({result_cpu.nit} iterations)")
print(f"  GPU: {result_gpu.fun:.4f} ({result_gpu.nit} iterations)")

print(f"\nFinal Σ max values:")
print(f"  CPU: {np.max(np.abs(sigma_cpu_final)):.4f}")
print(f"  GPU: {np.max(np.abs(sigma_gpu_final)):.4f}")

print(f"\nParameter differences:")
print(f"  μ max diff: {np.max(np.abs(mu_cpu_final - mu_gpu_final)):.2e}")
print(f"  Σ max diff: {np.max(np.abs(sigma_cpu_final - sigma_gpu_final)):.2e}")

if np.max(np.abs(sigma_gpu_final)) > 100:
    print("\n⚠️ GPU Σ exploded - optimization diverged!")
    print("Possible causes:")
    print("- Cholesky parameterization needs bounds")
    print("- FP32 precision insufficient for large covariances")
    print("- Line search overshooting")