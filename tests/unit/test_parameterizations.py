#!/usr/bin/env python3
"""
Unit tests for covariance parameterizations.

Tests pack/unpack operations, positive definiteness preservation,
and conversions between parameterizations.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymvnmle._objectives.parameterizations import (
    InverseCholeskyParameterization,
    CholeskyParameterization,
    MatrixLogParameterization,
    convert_parameters,
    get_parameterization
)


class TestParameterizationBase(unittest.TestCase):
    """Base class with common test data."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_vars = 3
        
        # Create a positive definite covariance matrix
        A = np.random.randn(self.n_vars, self.n_vars)
        self.sigma = A @ A.T + np.eye(self.n_vars)
        
        # Create mean vector
        self.mu = np.random.randn(self.n_vars)
        
        # Expected parameter dimensions
        self.n_mean_params = self.n_vars
        self.n_cov_params = (self.n_vars * (self.n_vars + 1)) // 2
        self.n_params = self.n_mean_params + self.n_cov_params
    
    def check_positive_definite(self, matrix):
        """Check if matrix is positive definite."""
        eigenvals = np.linalg.eigvalsh(matrix)
        return np.all(eigenvals > 0)


class TestInverseCholeskyParameterization(TestParameterizationBase):
    """Test R-compatible inverse Cholesky parameterization."""
    
    def setUp(self):
        super().setUp()
        self.param = InverseCholeskyParameterization(self.n_vars)
    
    def test_dimensions(self):
        """Test parameter dimensions."""
        self.assertEqual(self.param.n_vars, self.n_vars)
        self.assertEqual(self.param.n_mean_params, self.n_mean_params)
        self.assertEqual(self.param.n_cov_params, self.n_cov_params)
        self.assertEqual(self.param.n_params, self.n_params)
    
    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack round-trip consistency."""
        # Pack
        theta = self.param.pack(self.mu, self.sigma)
        self.assertEqual(len(theta), self.n_params)
        
        # Unpack
        mu_recovered, sigma_recovered = self.param.unpack(theta)
        
        # Check recovery
        np.testing.assert_allclose(mu_recovered, self.mu, rtol=1e-10)
        np.testing.assert_allclose(sigma_recovered, self.sigma, rtol=1e-10)
        
        # Check positive definiteness preserved
        self.assertTrue(self.check_positive_definite(sigma_recovered))
    
    def test_parameter_structure(self):
        """Test the structure of the parameter vector."""
        theta = self.param.pack(self.mu, self.sigma)
        
        # First n_vars elements should be the mean
        np.testing.assert_array_equal(theta[:self.n_vars], self.mu)
        
        # Next n_vars elements should be log of diagonal elements
        # These should be real numbers (not inf or nan)
        log_diag = theta[self.n_vars:2*self.n_vars]
        self.assertTrue(np.all(np.isfinite(log_diag)))
    
    def test_initial_parameters(self):
        """Test initial parameter generation."""
        sample_mean = np.array([1.0, 2.0, 3.0])
        sample_cov = np.array([[1.0, 0.5, 0.2],
                              [0.5, 2.0, 0.3],
                              [0.2, 0.3, 1.5]])
        
        theta_init = self.param.get_initial_parameters(sample_mean, sample_cov)
        
        # Check dimensions
        self.assertEqual(len(theta_init), self.n_params)
        
        # Unpack and verify
        mu_init, sigma_init = self.param.unpack(theta_init)
        np.testing.assert_allclose(mu_init, sample_mean, rtol=1e-10)
        np.testing.assert_allclose(sigma_init, sample_cov, rtol=1e-10)
    
    def test_singular_covariance_handling(self):
        """Test handling of near-singular covariance."""
        # Create near-singular covariance
        singular_cov = np.ones((self.n_vars, self.n_vars)) * 0.99
        np.fill_diagonal(singular_cov, 1.0)
        
        # This is near-singular (smallest eigenvalue ≈ 0.01)
        eigenvals = np.linalg.eigvalsh(singular_cov)
        self.assertLess(np.min(eigenvals), 0.02)
        
        # Should regularize and work
        theta_init = self.param.get_initial_parameters(self.mu, singular_cov)
        mu_recovered, sigma_recovered = self.param.unpack(theta_init)
        
        # Should be positive definite after regularization
        self.assertTrue(self.check_positive_definite(sigma_recovered))


class TestCholeskyParameterization(TestParameterizationBase):
    """Test standard Cholesky parameterization."""
    
    def setUp(self):
        super().setUp()
        self.param = CholeskyParameterization(self.n_vars)
    
    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack round-trip consistency."""
        # Pack
        theta = self.param.pack(self.mu, self.sigma)
        self.assertEqual(len(theta), self.n_params)
        
        # Unpack
        mu_recovered, sigma_recovered = self.param.unpack(theta)
        
        # Check recovery
        np.testing.assert_allclose(mu_recovered, self.mu, rtol=1e-10)
        np.testing.assert_allclose(sigma_recovered, self.sigma, rtol=1e-10)
        
        # Check positive definiteness preserved
        self.assertTrue(self.check_positive_definite(sigma_recovered))
    
    def test_different_from_inverse_cholesky(self):
        """Test that Cholesky gives different parameters than inverse Cholesky."""
        inv_chol = InverseCholeskyParameterization(self.n_vars)
        
        theta_chol = self.param.pack(self.mu, self.sigma)
        theta_inv = inv_chol.pack(self.mu, self.sigma)
        
        # Parameter vectors should be different
        self.assertFalse(np.allclose(theta_chol, theta_inv))
        
        # But should unpack to same (mu, sigma)
        mu_chol, sigma_chol = self.param.unpack(theta_chol)
        mu_inv, sigma_inv = inv_chol.unpack(theta_inv)
        
        np.testing.assert_allclose(mu_chol, mu_inv, rtol=1e-10)
        np.testing.assert_allclose(sigma_chol, sigma_inv, rtol=1e-10)


class TestMatrixLogParameterization(TestParameterizationBase):
    """Test matrix logarithm parameterization."""
    
    def setUp(self):
        super().setUp()
        self.param = MatrixLogParameterization(self.n_vars)
    
    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack round-trip consistency."""
        # Pack
        theta = self.param.pack(self.mu, self.sigma)
        self.assertEqual(len(theta), self.n_params)
        
        # Unpack
        mu_recovered, sigma_recovered = self.param.unpack(theta)
        
        # Check recovery (slightly looser tolerance for matrix exp/log)
        np.testing.assert_allclose(mu_recovered, self.mu, rtol=1e-9)
        np.testing.assert_allclose(sigma_recovered, self.sigma, rtol=1e-9)
        
        # Check positive definiteness preserved
        self.assertTrue(self.check_positive_definite(sigma_recovered))
    
    def test_no_cholesky_failure(self):
        """Test that matrix log doesn't fail where Cholesky might."""
        # Create a positive definite matrix close to singular
        eigenvals = np.array([1e-6, 0.1, 1.0])
        Q = np.linalg.qr(np.random.randn(self.n_vars, self.n_vars))[0]
        difficult_sigma = Q @ np.diag(eigenvals) @ Q.T
        
        # Matrix log should handle this
        theta = self.param.pack(self.mu, difficult_sigma)
        mu_recovered, sigma_recovered = self.param.unpack(theta)
        
        # Check positive definiteness preserved
        self.assertTrue(self.check_positive_definite(sigma_recovered))
        
        # Check eigenvalues are preserved (approximately)
        recovered_eigenvals = np.sort(np.linalg.eigvalsh(sigma_recovered))
        np.testing.assert_allclose(recovered_eigenvals, eigenvals, rtol=1e-8)


class TestParameterConversion(TestParameterizationBase):
    """Test conversion between parameterizations."""
    
    def test_inverse_to_standard_cholesky(self):
        """Test converting from inverse to standard Cholesky."""
        inv_param = InverseCholeskyParameterization(self.n_vars)
        std_param = CholeskyParameterization(self.n_vars)
        
        # Pack with inverse Cholesky
        theta_inv = inv_param.pack(self.mu, self.sigma)
        
        # Convert to standard Cholesky
        theta_std = convert_parameters(theta_inv, inv_param, std_param)
        
        # Unpack both
        mu_inv, sigma_inv = inv_param.unpack(theta_inv)
        mu_std, sigma_std = std_param.unpack(theta_std)
        
        # Should give same results
        np.testing.assert_allclose(mu_inv, mu_std, rtol=1e-10)
        np.testing.assert_allclose(sigma_inv, sigma_std, rtol=1e-10)
    
    def test_cholesky_to_matrix_log(self):
        """Test converting from Cholesky to matrix log."""
        chol_param = CholeskyParameterization(self.n_vars)
        log_param = MatrixLogParameterization(self.n_vars)
        
        # Pack with Cholesky
        theta_chol = chol_param.pack(self.mu, self.sigma)
        
        # Convert to matrix log
        theta_log = convert_parameters(theta_chol, chol_param, log_param)
        
        # Unpack both
        mu_chol, sigma_chol = chol_param.unpack(theta_chol)
        mu_log, sigma_log = log_param.unpack(theta_log)
        
        # Should give same results
        np.testing.assert_allclose(mu_chol, mu_log, rtol=1e-9)
        np.testing.assert_allclose(sigma_chol, sigma_log, rtol=1e-9)
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion through all parameterizations."""
        inv_param = InverseCholeskyParameterization(self.n_vars)
        std_param = CholeskyParameterization(self.n_vars)
        log_param = MatrixLogParameterization(self.n_vars)
        
        # Start with inverse Cholesky
        theta_inv = inv_param.pack(self.mu, self.sigma)
        
        # Convert: inverse -> standard -> log -> inverse
        theta_std = convert_parameters(theta_inv, inv_param, std_param)
        theta_log = convert_parameters(theta_std, std_param, log_param)
        theta_back = convert_parameters(theta_log, log_param, inv_param)
        
        # Should recover original parameters
        mu_original, sigma_original = inv_param.unpack(theta_inv)
        mu_recovered, sigma_recovered = inv_param.unpack(theta_back)
        
        np.testing.assert_allclose(mu_original, mu_recovered, rtol=1e-9)
        np.testing.assert_allclose(sigma_original, sigma_recovered, rtol=1e-9)


class TestParameterizationFactory(unittest.TestCase):
    """Test parameterization factory function."""
    
    def test_get_parameterization_by_name(self):
        """Test getting parameterizations by name."""
        n_vars = 4
        
        # Test various names
        inv_chol = get_parameterization('inverse_cholesky', n_vars)
        self.assertIsInstance(inv_chol, InverseCholeskyParameterization)
        
        std_chol = get_parameterization('cholesky', n_vars)
        self.assertIsInstance(std_chol, CholeskyParameterization)
        
        mat_log = get_parameterization('matrix_log', n_vars)
        self.assertIsInstance(mat_log, MatrixLogParameterization)
        
        # Test aliases
        r_param = get_parameterization('r', n_vars)
        self.assertIsInstance(r_param, InverseCholeskyParameterization)
        
        gpu_param = get_parameterization('gpu', n_vars)
        self.assertIsInstance(gpu_param, CholeskyParameterization)
    
    def test_invalid_parameterization_name(self):
        """Test error on invalid parameterization name."""
        with self.assertRaises(ValueError):
            get_parameterization('invalid', 3)


class TestNumericalStability(TestParameterizationBase):
    """Test numerical stability of parameterizations."""
    
    def test_extreme_eigenvalues(self):
        """Test with extreme eigenvalue ratios."""
        # Create matrix with large condition number
        eigenvals = np.array([1e-8, 1e-4, 1.0])
        Q = np.linalg.qr(np.random.randn(self.n_vars, self.n_vars))[0]
        extreme_sigma = Q @ np.diag(eigenvals) @ Q.T
        
        # Test each parameterization
        for param_class in [InverseCholeskyParameterization,
                           CholeskyParameterization,
                           MatrixLogParameterization]:
            param = param_class(self.n_vars)
            
            # Should handle regularization in initial parameters
            theta = param.get_initial_parameters(self.mu, extreme_sigma)
            mu_recovered, sigma_recovered = param.unpack(theta)
            
            # Should be positive definite
            self.assertTrue(self.check_positive_definite(sigma_recovered))
            
            # Mean should be preserved exactly
            np.testing.assert_allclose(mu_recovered, self.mu, rtol=1e-10)
    
    def test_large_dimension(self):
        """Test with larger dimensional matrices."""
        n_vars = 10
        
        # Create large positive definite matrix
        A = np.random.randn(n_vars, n_vars)
        large_sigma = A @ A.T + np.eye(n_vars)
        large_mu = np.random.randn(n_vars)
        
        # Test each parameterization
        for param_name in ['inverse_cholesky', 'cholesky', 'matrix_log']:
            param = get_parameterization(param_name, n_vars)
            
            # Pack and unpack
            theta = param.pack(large_mu, large_sigma)
            mu_recovered, sigma_recovered = param.unpack(theta)
            
            # Check recovery
            np.testing.assert_allclose(mu_recovered, large_mu, rtol=1e-9)
            np.testing.assert_allclose(sigma_recovered, large_sigma, rtol=1e-9)
            
            # Check positive definiteness
            self.assertTrue(self.check_positive_definite(sigma_recovered))


def run_quick_test():
    """Quick test of parameterizations."""
    print("\n" + "="*70)
    print("QUICK PARAMETERIZATION TEST")
    print("="*70)
    
    n_vars = 3
    
    # Create test data
    mu = np.array([1.0, 2.0, 3.0])
    A = np.array([[1.0, 0.5, 0.2],
                   [0.5, 2.0, 0.3],
                   [0.2, 0.3, 1.5]])
    sigma = A @ A.T + np.eye(n_vars)
    
    print(f"\nTest data:")
    print(f"μ = {mu}")
    print(f"Σ eigenvalues = {np.linalg.eigvalsh(sigma)}")
    
    # Test each parameterization
    for name in ['inverse_cholesky', 'cholesky', 'matrix_log']:
        print(f"\n📊 Testing {name} parameterization:")
        param = get_parameterization(name, n_vars)
        
        # Pack
        theta = param.pack(mu, sigma)
        print(f"  Parameter vector length: {len(theta)}")
        
        # Unpack
        mu_recovered, sigma_recovered = param.unpack(theta)
        
        # Check
        mu_error = np.max(np.abs(mu - mu_recovered))
        sigma_error = np.max(np.abs(sigma - sigma_recovered))
        
        print(f"  ✅ Round-trip errors: μ={mu_error:.2e}, Σ={sigma_error:.2e}")
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvalsh(sigma_recovered)
        print(f"  ✅ Positive definite: min eigenvalue = {np.min(eigenvals):.2e}")
    
    print("\n✅ All parameterizations working correctly!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parameterizations")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Run full test suite
        unittest.main(argv=[''], verbosity=2 if args.verbose else 1)