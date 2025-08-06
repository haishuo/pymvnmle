#!/usr/bin/env python3
"""
Unit tests for computational backends.

Tests each backend's operations for correctness and precision.
Includes tests for CPU (NumPy) and GPU (PyTorch) backends.
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymvnmle._backends.base import BackendFactory, BackendBase
from pymvnmle._backends.cpu_fp64_backend import NumpyBackendFP64
from pymvnmle._backends import get_backend, list_available_backends


class TestBackendBase(unittest.TestCase):
    """Test base backend functionality."""
    
    def test_backend_factory_cpu(self):
        """Test CPU backend creation."""
        backend = BackendFactory.create_backend(use_fp64=True, device_type='cpu')
        
        self.assertIsInstance(backend, NumpyBackendFP64)
        self.assertEqual(backend.precision, 'fp64')
        self.assertEqual(backend.device_type, 'cpu')
        self.assertTrue(backend.is_available())
    
    def test_backend_factory_invalid_device(self):
        """Test factory with invalid device type."""
        with self.assertRaises(ValueError):
            BackendFactory.create_backend(use_fp64=True, device_type='invalid')
    
    def test_backend_factory_metal_fp64_error(self):
        """Test that FP64 on Metal raises error."""
        with self.assertRaises(RuntimeError) as context:
            BackendFactory.create_backend(use_fp64=True, device_type='metal')
        
        self.assertIn("FP64 not supported on Apple Metal", str(context.exception))


class TestNumpyBackendFP64(unittest.TestCase):
    """Test NumPy CPU backend operations."""
    
    def setUp(self):
        """Create backend and test data."""
        self.backend = NumpyBackendFP64()
        
        # Create test matrices
        np.random.seed(42)
        self.size = 5
        
        # Positive definite matrix
        A = np.random.randn(self.size, self.size)
        self.pos_def = A @ A.T + np.eye(self.size)
        
        # Vector
        self.vector = np.random.randn(self.size)
        
        # Another matrix for operations
        self.matrix_b = np.random.randn(self.size, self.size)
    
    def test_initialization(self):
        """Test backend initialization."""
        self.assertEqual(self.backend.name, 'numpy_fp64')
        self.assertEqual(self.backend.precision, 'fp64')
        self.assertEqual(self.backend.dtype, np.float64)
        self.assertTrue(self.backend.is_available())
    
    def test_to_device_conversion(self):
        """Test array conversion to FP64."""
        # Test FP32 to FP64 conversion
        array_fp32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        array_fp64 = self.backend.to_device(array_fp32)
        
        self.assertEqual(array_fp64.dtype, np.float64)
        np.testing.assert_array_equal(array_fp64, array_fp32)
        
        # Test FP64 passthrough
        array_fp64_orig = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        array_fp64_result = self.backend.to_device(array_fp64_orig)
        
        self.assertIs(array_fp64_result, array_fp64_orig)  # Should be same object
    
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition."""
        # Lower triangular
        L = self.backend.cholesky(self.pos_def, upper=False)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, self.pos_def, rtol=1e-10)
        
        # Upper triangular
        U = self.backend.cholesky(self.pos_def, upper=True)
        reconstructed = U.T @ U
        np.testing.assert_allclose(reconstructed, self.pos_def, rtol=1e-10)
        
        # Test with non-positive definite matrix (should raise)
        non_pd = np.array([[1, 2], [2, 1]])  # Not positive definite
        with self.assertRaises(np.linalg.LinAlgError):
            self.backend.cholesky(non_pd, upper=False)
    
    def test_solve_triangular(self):
        """Test triangular system solver."""
        # Create lower triangular system
        L = np.tril(np.random.randn(self.size, self.size))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 1)  # Ensure non-singular
        b = np.random.randn(self.size)
        
        # Solve L @ x = b
        x = self.backend.solve_triangular(L, b, upper=False, trans='N')
        np.testing.assert_allclose(L @ x, b, rtol=1e-10)
        
        # Solve L.T @ x = b
        x = self.backend.solve_triangular(L, b, upper=False, trans='T')
        np.testing.assert_allclose(L.T @ x, b, rtol=1e-10)
        
        # Test upper triangular
        U = L.T
        x = self.backend.solve_triangular(U, b, upper=True, trans='N')
        np.testing.assert_allclose(U @ x, b, rtol=1e-10)
    
    def test_matmul(self):
        """Test matrix multiplication."""
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        
        C = self.backend.matmul(A, B)
        C_expected = A @ B
        
        np.testing.assert_allclose(C, C_expected, rtol=1e-12)
        self.assertEqual(C.shape, (3, 5))
    
    def test_inv(self):
        """Test matrix inversion."""
        inv = self.backend.inv(self.pos_def)
        
        # Check A @ A^-1 = I
        identity = self.backend.matmul(self.pos_def, inv)
        np.testing.assert_allclose(identity, np.eye(self.size), rtol=1e-10, atol=1e-14)
        
        # Check A^-1 @ A = I
        identity = self.backend.matmul(inv, self.pos_def)
        np.testing.assert_allclose(identity, np.eye(self.size), rtol=1e-10, atol=1e-14)
    
    def test_log_det(self):
        """Test log determinant computation."""
        # Compute using backend
        log_det = self.backend.log_det(self.pos_def)
        
        # Compute reference using NumPy
        sign, log_det_ref = np.linalg.slogdet(self.pos_def)
        
        self.assertAlmostEqual(log_det, log_det_ref, places=10)
        
        # Test with known determinant
        A = np.diag([1, 2, 3, 4])
        log_det = self.backend.log_det(A)
        expected = np.log(24)  # det = 1*2*3*4 = 24
        self.assertAlmostEqual(log_det, expected, places=12)
    
    def test_quadratic_form(self):
        """Test quadratic form computation."""
        result = self.backend.quadratic_form(self.vector, self.pos_def)
        
        # Compute reference
        expected = self.vector.T @ self.pos_def @ self.vector
        
        self.assertAlmostEqual(result, expected, places=10)
        self.assertIsInstance(result, float)
    
    def test_solve_posdef(self):
        """Test positive definite system solver."""
        b = np.random.randn(self.size)
        
        # Solve A @ x = b
        x = self.backend.solve_posdef(self.pos_def, b)
        
        # Check solution
        np.testing.assert_allclose(self.pos_def @ x, b, rtol=1e-10)
        
        # Compare with direct solver
        x_direct = np.linalg.solve(self.pos_def, b)
        np.testing.assert_allclose(x, x_direct, rtol=1e-10)
    
    def test_eigenvalues(self):
        """Test eigenvalue computation."""
        eigenvals = self.backend.eigenvalues(self.pos_def)
        
        # Should all be positive for positive definite matrix
        self.assertTrue(np.all(eigenvals > 0))
        
        # Check against NumPy
        eigenvals_ref = np.linalg.eigvalsh(self.pos_def)
        np.testing.assert_allclose(eigenvals, eigenvals_ref, rtol=1e-10)
    
    def test_is_positive_definite(self):
        """Test positive definite check."""
        # Positive definite matrix
        self.assertTrue(self.backend.is_positive_definite(self.pos_def))
        
        # Not positive definite
        non_pd = np.array([[1, 2], [2, 1]])
        self.assertFalse(self.backend.is_positive_definite(non_pd))
        
        # Singular matrix
        singular = np.ones((3, 3))
        self.assertFalse(self.backend.is_positive_definite(singular))
    
    def test_make_positive_definite(self):
        """Test making matrix positive definite."""
        # Start with non-PD matrix
        non_pd = np.array([[1, 2], [2, 1]])
        
        # Make it PD
        pd = self.backend.make_positive_definite(non_pd, min_eigenval=0.1)
        
        # Check it's now PD
        self.assertTrue(self.backend.is_positive_definite(pd))
        
        # Check eigenvalues are at least min_eigenval
        eigenvals = self.backend.eigenvalues(pd)
        self.assertTrue(np.all(eigenvals >= 0.1))
        
        # Check symmetry
        np.testing.assert_allclose(pd, pd.T, rtol=1e-12)
    
    def test_device_info(self):
        """Test device information retrieval."""
        info = self.backend.get_device_info()
        
        self.assertEqual(info['device_type'], 'cpu')
        self.assertEqual(info['precision'], 'fp64')
        self.assertIn('memory_total', info)
        self.assertIn('cpu_count', info)
    
    def test_backend_info(self):
        """Test backend information retrieval."""
        info = self.backend.get_info()
        
        self.assertEqual(info['backend_name'], 'numpy_fp64')
        self.assertEqual(info['precision'], 'fp64')
        self.assertEqual(info['optimization_method'], 'BFGS')
        self.assertFalse(info['supports_autodiff'])
        self.assertIn('numpy_version', info)
        self.assertIn('scipy_version', info)


class TestPyTorchBackends(unittest.TestCase):
    """Test PyTorch GPU backends (if available)."""
    
    def setUp(self):
        """Check if PyTorch is available."""
        try:
            import torch
            self.torch_available = True
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            self.torch_available = False
            self.has_cuda = False
            self.has_mps = False
    
    @unittest.skipUnless('torch_available', "PyTorch not installed")
    def test_fp32_backend_creation(self):
        """Test FP32 backend creation."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        from pymvnmle._backends.gpu_fp32_backend import PyTorchBackendFP32
        
        # Should work even without GPU (falls back to CPU)
        backend = PyTorchBackendFP32()
        
        self.assertEqual(backend.precision, 'fp32')
        self.assertEqual(backend.optimization_method, 'BFGS')
        self.assertTrue(backend.supports_autodiff())
    
    @unittest.skipUnless('torch_available', "PyTorch not installed")
    def test_fp32_backend_operations(self):
        """Test FP32 backend operations."""
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        from pymvnmle._backends.gpu_fp32_backend import PyTorchBackendFP32
        import torch
        
        backend = PyTorchBackendFP32()
        
        # Create test data
        np.random.seed(42)
        size = 4
        A = np.random.randn(size, size).astype(np.float32)
        pos_def = A @ A.T + np.eye(size, dtype=np.float32)
        
        # Transfer to device
        pos_def_gpu = backend.to_device(pos_def)
        self.assertEqual(pos_def_gpu.dtype, torch.float32)
        
        # Cholesky
        L = backend.cholesky(pos_def_gpu, upper=False)
        
        # Transfer back and check
        L_cpu = backend.to_numpy(L)
        reconstructed = L_cpu @ L_cpu.T
        np.testing.assert_allclose(reconstructed, pos_def, rtol=1e-5)  # Lower precision for FP32
        
        # Log determinant
        log_det = backend.log_det(pos_def_gpu)
        self.assertIsInstance(log_det, float)
    
    @unittest.skipUnless('has_cuda', "CUDA not available")
    def test_fp64_backend_creation(self):
        """Test FP64 backend creation on CUDA."""
        if not self.has_cuda:
            self.skipTest("CUDA not available")
        
        from pymvnmle._backends.gpu_fp64_backend import PyTorchBackendFP64
        
        # Should work on CUDA
        backend = PyTorchBackendFP64(verify_performance=False)  # Skip perf test for speed
        
        self.assertEqual(backend.precision, 'fp64')
        self.assertEqual(backend.optimization_method, 'Newton-CG')
        self.assertTrue(backend.supports_autodiff())
        self.assertEqual(backend.device.type, 'cuda')
    
    def test_fp64_backend_no_cuda_error(self):
        """Test FP64 backend fails without CUDA."""
        if self.has_cuda:
            self.skipTest("CUDA is available")
        
        if not self.torch_available:
            self.skipTest("PyTorch not available")
        
        from pymvnmle._backends.gpu_fp64_backend import PyTorchBackendFP64
        
        # Should raise error without CUDA
        with self.assertRaises(RuntimeError) as context:
            PyTorchBackendFP64()
        
        self.assertIn("FP64 backend requires CUDA", str(context.exception))


class TestBackendNumericalConsistency(unittest.TestCase):
    """Test numerical consistency across backends."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(123)  # Different seed for variety
        self.size = 6
        
        # Create positive definite matrix
        A = np.random.randn(self.size, self.size)
        self.pos_def = A @ A.T + 2 * np.eye(self.size)
        
        # Ensure it's really positive definite
        eigenvals = np.linalg.eigvalsh(self.pos_def)
        self.assertTrue(np.all(eigenvals > 0))
        
        self.vector = np.random.randn(self.size)
        self.matrix_b = np.random.randn(self.size, self.size)
    
    def test_cpu_backend_consistency(self):
        """Test CPU backend gives consistent results."""
        backend = NumpyBackendFP64()
        
        # Run operations twice - should get identical results
        log_det1 = backend.log_det(self.pos_def)
        log_det2 = backend.log_det(self.pos_def)
        self.assertEqual(log_det1, log_det2)
        
        L1 = backend.cholesky(self.pos_def, upper=False)
        L2 = backend.cholesky(self.pos_def, upper=False)
        np.testing.assert_array_equal(L1, L2)
    
    def test_positive_definite_preservation(self):
        """Test that operations preserve positive definiteness."""
        backend = NumpyBackendFP64()
        
        # Cholesky and reconstruct
        L = backend.cholesky(self.pos_def, upper=False)
        reconstructed = L @ L.T
        
        # Should still be positive definite
        self.assertTrue(backend.is_positive_definite(reconstructed))
        
        # Make PD should preserve PD property
        made_pd = backend.make_positive_definite(self.pos_def)
        self.assertTrue(backend.is_positive_definite(made_pd))


def run_quick_backend_test():
    """Quick test to verify backends are working."""
    print("\n" + "="*70)
    print("QUICK BACKEND TEST")
    print("="*70)
    
    # Test CPU backend
    print("\n📊 Testing CPU Backend:")
    cpu_backend = NumpyBackendFP64()
    print(f"  ✅ CPU backend created: {cpu_backend.name}")
    
    # Test basic operation
    A = np.array([[4, 2], [2, 3]], dtype=np.float64)
    L = cpu_backend.cholesky(A, upper=False)
    print(f"  ✅ Cholesky decomposition works")
    
    # Test GPU backends if available
    try:
        import torch
        
        if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            from pymvnmle._backends.gpu_fp32_backend import PyTorchBackendFP32
            
            print("\n📊 Testing GPU FP32 Backend:")
            gpu32_backend = PyTorchBackendFP32()
            print(f"  ✅ GPU FP32 backend created: {gpu32_backend.name}")
            print(f"  Device: {gpu32_backend.device}")
            
            # Test operation
            A_gpu = gpu32_backend.to_device(A.astype(np.float32))
            L_gpu = gpu32_backend.cholesky(A_gpu, upper=False)
            print(f"  ✅ GPU Cholesky decomposition works")
        else:
            print("\n⚠️  No GPU available for testing")
            
    except ImportError:
        print("\n⚠️  PyTorch not installed - skipping GPU tests")
    
    print("\n✅ Basic backend functionality verified!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test computational backends")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick backend test only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_backend_test()
    else:
        # Run full test suite
        unittest.main(argv=[''], verbosity=2 if args.verbose else 1)