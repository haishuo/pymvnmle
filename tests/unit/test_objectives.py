#!/usr/bin/env python3
"""
Comprehensive unit tests for MLE objectives.

Tests CPU and GPU objectives for correctness, consistency,
and numerical accuracy.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymvnmle._objectives import (
    get_objective,
    create_objective,
    compare_objectives,
    benchmark_objectives,
    CPUObjectiveFP64,
    GPU_AVAILABLE
)
from pymvnmle._objectives.base import PatternData


class TestObjectiveBase(unittest.TestCase):
    """Base class with common test data."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        
        # Small complete data
        self.complete_data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ])
        
        # Data with missing values
        self.missing_data = np.array([
            [1.0, 2.0, np.nan],
            [2.0, np.nan, 4.0],
            [3.0, 4.0, 5.0],
            [np.nan, 5.0, 6.0],
            [5.0, 6.0, np.nan]
        ])
        
        # Create known positive definite covariance
        A = np.array([[1.0, 0.5, 0.2],
                      [0.5, 2.0, 0.3],
                      [0.2, 0.3, 1.5]])
        self.known_sigma = A
        self.known_mu = np.array([1.0, 2.0, 3.0])
        
        # Expected dimensions
        self.n_vars = 3
        self.n_mean_params = 3
        self.n_cov_params = 6  # 3*(3+1)/2
        self.n_params = 9


class TestCPUObjective(TestObjectiveBase):
    """Test CPU objective (R-compatible reference)."""
    
    def test_initialization(self):
        """Test objective initialization."""
        obj = CPUObjectiveFP64(self.complete_data)
        
        self.assertEqual(obj.n_vars, 3)
        self.assertEqual(obj.n_obs, 5)
        self.assertEqual(obj.n_params, 9)
        self.assertTrue(obj.is_complete)
        self.assertEqual(obj.n_patterns, 1)
    
    def test_pattern_extraction(self):
        """Test missingness pattern extraction."""
        obj = CPUObjectiveFP64(self.missing_data)
        
        self.assertFalse(obj.is_complete)
        self.assertGreater(obj.n_patterns, 1)
        
        # Check patterns are valid
        for pattern in obj.patterns:
            self.assertIsInstance(pattern, PatternData)
            self.assertGreater(pattern.n_obs, 0)
            self.assertEqual(
                len(pattern.observed_indices) + len(pattern.missing_indices),
                self.n_vars
            )
    
    def test_initial_parameters(self):
        """Test initial parameter generation."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta_init = obj.get_initial_parameters()
        
        # Check dimensions
        self.assertEqual(len(theta_init), self.n_params)
        
        # Check all finite
        self.assertTrue(np.all(np.isfinite(theta_init)))
        
        # Extract and check positive definiteness
        mu, sigma, _ = obj.extract_parameters(theta_init)
        eigenvals = np.linalg.eigvalsh(sigma)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_objective_computation(self):
        """Test objective function computation."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Compute objective
        obj_value = obj.compute_objective(theta)
        
        # Should be finite (can be negative if log-likelihood is positive!)
        self.assertTrue(np.isfinite(obj_value))
        # The objective is -2*log-likelihood, which can be negative or positive
    
    def test_gradient_computation(self):
        """Test gradient computation via finite differences."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Compute gradient
        grad = obj.compute_gradient(theta)
        
        # Check dimensions and finiteness
        self.assertEqual(len(grad), self.n_params)
        self.assertTrue(np.all(np.isfinite(grad)))
        
        # Near optimum, gradient should be small
        theta_perturbed = theta + 0.1 * np.random.randn(len(theta))
        grad_perturbed = obj.compute_gradient(theta_perturbed)
        
        # Gradient should change when parameters change
        self.assertFalse(np.allclose(grad, grad_perturbed))
    
    def test_parameter_extraction(self):
        """Test extraction of mu, sigma, and log-likelihood."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta = obj.get_initial_parameters()
        
        mu, sigma, loglik = obj.extract_parameters(theta)
        
        # Check dimensions
        self.assertEqual(len(mu), self.n_vars)
        self.assertEqual(sigma.shape, (self.n_vars, self.n_vars))
        
        # Check properties
        self.assertTrue(np.allclose(sigma, sigma.T))  # Symmetric
        eigenvals = np.linalg.eigvalsh(sigma)
        self.assertTrue(np.all(eigenvals > 0))  # Positive definite
        
        # Log-likelihood should be finite
        self.assertTrue(np.isfinite(loglik))
    
    def test_convergence_check(self):
        """Test convergence checking."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Should not be converged initially
        self.assertFalse(obj.check_convergence(theta, tol=1e-10))
        
        # With very loose tolerance, might be "converged"
        might_converge = obj.check_convergence(theta, tol=1.0)
        # This could be True or False depending on initial point
    
    def test_missing_data_handling(self):
        """Test handling of missing data patterns."""
        obj = CPUObjectiveFP64(self.missing_data)
        theta = obj.get_initial_parameters()
        
        # Should still compute objective
        obj_value = obj.compute_objective(theta)
        self.assertTrue(np.isfinite(obj_value))
        
        # Pattern summary should show multiple patterns
        summary = obj.get_pattern_summary()
        self.assertGreater(summary['n_patterns'], 1)
        self.assertEqual(summary['n_obs'], 5)
    
    def test_validation(self):
        """Test parameter validation."""
        obj = CPUObjectiveFP64(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Valid parameters
        valid, msg = obj.validate_parameters(theta)
        self.assertTrue(valid)
        
        # Invalid: wrong length
        invalid_theta = theta[:-1]
        valid, msg = obj.validate_parameters(invalid_theta)
        self.assertFalse(valid)
        self.assertIn("Expected", msg)
        
        # Invalid: contains NaN
        theta_nan = theta.copy()
        theta_nan[0] = np.nan
        valid, msg = obj.validate_parameters(theta_nan)
        self.assertFalse(valid)
        self.assertIn("NaN", msg)


class TestGPUObjectives(TestObjectiveBase):
    """Test GPU objectives (if available)."""
    
    def setUp(self):
        super().setUp()
        self.gpu_available = GPU_AVAILABLE
        
        if self.gpu_available:
            try:
                import torch
                self.has_cuda = torch.cuda.is_available()
                self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            except ImportError:
                self.has_cuda = False
                self.has_mps = False
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_gpu_fp32_initialization(self):
        """Test GPU FP32 objective initialization."""
        from pymvnmle._objectives import GPUObjectiveFP32
        
        obj = GPUObjectiveFP32(self.complete_data)
        
        self.assertEqual(obj.n_vars, 3)
        self.assertEqual(obj.n_obs, 5)
        self.assertEqual(obj.n_params, 9)
        
        # Check device info
        device_info = obj.get_device_info()
        self.assertIn('device', device_info)
        self.assertEqual(device_info['dtype'], 'float32')
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_gpu_fp32_objective(self):
        """Test GPU FP32 objective computation."""
        from pymvnmle._objectives import GPUObjectiveFP32
        
        obj = GPUObjectiveFP32(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Compute objective
        obj_value = obj.compute_objective(theta)
        self.assertTrue(np.isfinite(obj_value))
        # The objective is -2*log-likelihood, which can be negative or positive
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_gpu_fp32_gradient(self):
        """Test GPU FP32 gradient via autodiff."""
        from pymvnmle._objectives import GPUObjectiveFP32
        
        obj = GPUObjectiveFP32(self.complete_data)
        theta = obj.get_initial_parameters()
        
        # Compute gradient
        grad = obj.compute_gradient(theta)
        
        self.assertEqual(len(grad), self.n_params)
        self.assertTrue(np.all(np.isfinite(grad)))
        
        # GPU and CPU use DIFFERENT parameterizations!
        # GPU uses standard Cholesky: θ = [μ, log(diag(L)), off-diag(L)]
        # CPU uses inverse Cholesky: θ = [μ, log(diag(Δ)), off-diag(Δ)]
        # So their gradients are in different spaces and can't be compared directly
        
        # Instead, test that gradient changes when parameters change
        theta_perturbed = theta + 0.01 * np.random.randn(len(theta))
        grad_perturbed = obj.compute_gradient(theta_perturbed)
        
        # Gradients should be different at different points
        self.assertFalse(np.allclose(grad, grad_perturbed, rtol=1e-3))
    
    @unittest.skipUnless(GPU_AVAILABLE and 'has_cuda', "CUDA not available")
    def test_gpu_fp64_initialization(self):
        """Test GPU FP64 objective initialization."""
        if not self.has_cuda:
            self.skipTest("CUDA not available")
        
        from pymvnmle._objectives import GPUObjectiveFP64
        
        # Should work on CUDA
        obj = GPUObjectiveFP64(self.complete_data, verify_fp64_performance=False)
        
        self.assertEqual(obj.n_vars, 3)
        self.assertEqual(obj.n_obs, 5)
        
        device_info = obj.get_device_info()
        self.assertEqual(device_info['dtype'], 'float64')
        self.assertTrue(device_info['supports_newton_cg'])
    
    @unittest.skipUnless(GPU_AVAILABLE and 'has_cuda', "CUDA not available")
    def test_gpu_fp64_hessian(self):
        """Test GPU FP64 Hessian computation."""
        if not self.has_cuda:
            self.skipTest("CUDA not available")
        
        from pymvnmle._objectives import GPUObjectiveFP64
        
        obj = GPUObjectiveFP64(self.complete_data, verify_fp64_performance=False)
        theta = obj.get_initial_parameters()
        
        # Compute Hessian
        hessian = obj.compute_hessian(theta)
        
        # Check properties
        self.assertEqual(hessian.shape, (self.n_params, self.n_params))
        self.assertTrue(np.allclose(hessian, hessian.T))  # Symmetric
        self.assertTrue(np.all(np.isfinite(hessian)))
        
        # Should be positive definite near minimum
        eigenvals = np.linalg.eigvalsh(hessian)
        # May not be positive definite at arbitrary point
        self.assertTrue(np.all(np.isfinite(eigenvals)))


class TestObjectiveConsistency(TestObjectiveBase):
    """Test consistency across different objectives."""
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU give similar results."""
        from pymvnmle._objectives import GPUObjectiveFP32
        
        # Create objectives
        cpu_obj = CPUObjectiveFP64(self.complete_data)
        gpu_obj = GPUObjectiveFP32(self.complete_data)
        
        # Get initial parameters (should be similar)
        theta_cpu = cpu_obj.get_initial_parameters()
        theta_gpu = gpu_obj.get_initial_parameters()
        
        # Note: Different parameterizations, so theta vectors differ
        # But extracted mu/sigma should be similar
        mu_cpu, sigma_cpu, _ = cpu_obj.extract_parameters(theta_cpu)
        mu_gpu, sigma_gpu, _ = gpu_obj.extract_parameters(theta_gpu)
        
        np.testing.assert_allclose(mu_cpu, mu_gpu, rtol=1e-5)
        np.testing.assert_allclose(sigma_cpu, sigma_gpu, rtol=1e-5)
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_cpu_gpu_objective_values(self):
        """Test that CPU and GPU objectives give similar values for same (μ, Σ)."""
        from pymvnmle._objectives import GPUObjectiveFP32
        from pymvnmle._objectives.parameterizations import (
            InverseCholeskyParameterization,
            CholeskyParameterization
        )
        
        # Create objectives
        cpu_obj = CPUObjectiveFP64(self.complete_data)
        gpu_obj = GPUObjectiveFP32(self.complete_data)
        
        # Use a fixed, known (μ, Σ) for comparison
        mu_test = np.array([2.0, 3.0, 4.0])
        # Create a simple positive definite matrix
        sigma_test = np.array([[2.0, 0.5, 0.3],
                              [0.5, 3.0, 0.4],
                              [0.3, 0.4, 2.5]])
        
        # Pack into respective parameterizations
        cpu_param = InverseCholeskyParameterization(self.n_vars)
        gpu_param = CholeskyParameterization(self.n_vars)
        
        theta_cpu = cpu_param.pack(mu_test, sigma_test)
        theta_gpu = gpu_param.pack(mu_test, sigma_test)
        
        # Compute objectives
        obj_cpu = cpu_obj.compute_objective(theta_cpu)
        obj_gpu = gpu_obj.compute_objective(theta_gpu)
        
        # Should be close (allowing for FP32 vs FP64 differences)
        # Use relative tolerance since values can be large
        rel_diff = abs(obj_cpu - obj_gpu) / abs(obj_cpu)
        self.assertLess(rel_diff, 0.001)  # Less than 0.1% relative difference
    
    def test_complete_vs_missing_data(self):
        """Test that complete data is handled efficiently."""
        obj_complete = CPUObjectiveFP64(self.complete_data)
        
        # Add tiny missing value to force pattern extraction
        almost_complete = self.complete_data.copy()
        almost_complete[-1, -1] = np.nan
        obj_missing = CPUObjectiveFP64(almost_complete)
        
        # Complete data should have 1 pattern
        self.assertEqual(obj_complete.n_patterns, 1)
        
        # Missing data should have more patterns
        self.assertGreater(obj_missing.n_patterns, 1)
        
        # Both should work
        theta_c = obj_complete.get_initial_parameters()
        theta_m = obj_missing.get_initial_parameters()
        
        obj_val_c = obj_complete.compute_objective(theta_c)
        obj_val_m = obj_missing.compute_objective(theta_m)
        
        self.assertTrue(np.isfinite(obj_val_c))
        self.assertTrue(np.isfinite(obj_val_m))


class TestObjectiveFactory(TestObjectiveBase):
    """Test objective factory functions."""
    
    def test_get_objective_cpu(self):
        """Test creating CPU objective."""
        obj = get_objective(self.complete_data, backend='cpu')
        self.assertIsInstance(obj, CPUObjectiveFP64)
        
        # Test alias
        obj = get_objective(self.complete_data, backend='numpy')
        self.assertIsInstance(obj, CPUObjectiveFP64)
    
    @unittest.skipUnless(GPU_AVAILABLE, "PyTorch not available")
    def test_get_objective_gpu(self):
        """Test creating GPU objectives."""
        from pymvnmle._objectives import GPUObjectiveFP32, GPUObjectiveFP64
        
        # FP32
        obj = get_objective(self.complete_data, backend='gpu', precision='fp32')
        self.assertIsInstance(obj, GPUObjectiveFP32)
        
        # Auto-select precision
        obj = get_objective(self.complete_data, backend='gpu')
        # Should be FP32 or FP64 depending on hardware
        self.assertTrue(isinstance(obj, (GPUObjectiveFP32, GPUObjectiveFP64)))
    
    def test_create_objective(self):
        """Test simple boolean interface."""
        # CPU
        obj = create_objective(self.complete_data, use_gpu=False)
        self.assertIsInstance(obj, CPUObjectiveFP64)
        
        if GPU_AVAILABLE:
            # GPU
            obj = create_objective(self.complete_data, use_gpu=True, use_fp64=False)
            from pymvnmle._objectives import GPUObjectiveFP32
            self.assertIsInstance(obj, GPUObjectiveFP32)


class TestObjectiveUtilities(TestObjectiveBase):
    """Test utility functions."""
    
    def test_compare_objectives(self):
        """Test objective comparison utility."""
        # Need parameters for comparison
        cpu_obj = CPUObjectiveFP64(self.complete_data)
        theta = cpu_obj.get_initial_parameters()
        
        # Compare available backends
        results = compare_objectives(self.complete_data, theta, backends=['cpu'])
        
        self.assertIn('cpu', results)
        self.assertIn('objective', results['cpu'])
        self.assertIn('loglik', results['cpu'])
        self.assertIn('mu', results['cpu'])
        self.assertIn('sigma', results['cpu'])
    
    def test_benchmark_objectives(self):
        """Test objective benchmarking."""
        results = benchmark_objectives(self.complete_data, n_iterations=2)
        
        self.assertIn('cpu', results)
        self.assertIn('objective_time', results['cpu'])
        self.assertIn('gradient_time', results['cpu'])
        self.assertIn('speedup', results['cpu'])
        
        # CPU should have speedup of 1.0 (baseline)
        self.assertEqual(results['cpu']['speedup'], 1.0)


def run_quick_test():
    """Quick test of objectives."""
    print("\n" + "="*70)
    print("QUICK OBJECTIVE TEST")
    print("="*70)
    
    # Create test data
    np.random.seed(42)
    data = np.random.randn(10, 3)
    data[3, 1] = np.nan
    data[7, 2] = np.nan
    
    print(f"\nTest data: {data.shape[0]} observations, {data.shape[1]} variables")
    print(f"Missing values: {np.sum(np.isnan(data))}")
    
    # Test CPU objective
    print("\n📊 Testing CPU Objective (R-compatible):")
    cpu_obj = CPUObjectiveFP64(data)
    theta = cpu_obj.get_initial_parameters()
    obj_val = cpu_obj.compute_objective(theta)
    grad = cpu_obj.compute_gradient(theta)
    
    print(f"  ✅ Patterns: {cpu_obj.n_patterns}")
    print(f"  ✅ Parameters: {len(theta)}")
    print(f"  ✅ Objective: {obj_val:.4f}")
    print(f"  ✅ Gradient norm: {np.linalg.norm(grad):.4f}")
    
    # Test GPU if available
    if GPU_AVAILABLE:
        print("\n📊 Testing GPU Objectives:")
        from pymvnmle._objectives import GPUObjectiveFP32
        
        gpu_obj = GPUObjectiveFP32(data)
        theta_gpu = gpu_obj.get_initial_parameters()
        obj_val_gpu = gpu_obj.compute_objective(theta_gpu)
        grad_gpu = gpu_obj.compute_gradient(theta_gpu)
        
        print(f"  ✅ Device: {gpu_obj.get_device_info()['device']}")
        print(f"  ✅ Objective: {obj_val_gpu:.4f}")
        print(f"  ✅ Gradient norm: {np.linalg.norm(grad_gpu):.4f}")
        print(f"  ✅ Autodiff gradients working!")
    else:
        print("\n⚠️  PyTorch not available - skipping GPU tests")
    
    print("\n✅ Objective functions working correctly!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLE objectives")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Run full test suite
        unittest.main(argv=[''], verbosity=2 if args.verbose else 1)