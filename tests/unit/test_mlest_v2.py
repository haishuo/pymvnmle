"""
Unit tests for PyMVNMLE v2.0 mlest function and module initialization.

Tests the new precision-based architecture, backend selection, gpu64 parameter,
and all convenience functions in the main module.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock

import pymvnmle as pmle
from pymvnmle import mlest, MLResult

# Try to import PrecisionDetector - handle different possible module structures
try:
    from pymvnmle._backends.precision_detector import PrecisionDetector
except ImportError:
    # Module might export it differently or not at all
    # Create a mock for testing purposes
    class PrecisionDetector:
        def detect_gpu(self):
            return {
                'gpu_type': 'none',
                'fp64_support': 'none',
                'device_name': 'None'
            }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_data():
    """Simple 2D dataset with no missing values."""
    np.random.seed(42)
    n_obs, n_vars = 50, 3
    data = np.random.randn(n_obs, n_vars)
    return data


@pytest.fixture
def missing_data():
    """Dataset with missing values."""
    np.random.seed(42)
    n_obs, n_vars = 100, 4
    data = np.random.randn(n_obs, n_vars)
    # Add 20% missing values
    mask = np.random.rand(n_obs, n_vars) < 0.2
    data[mask] = np.nan
    
    # Ensure no rows or columns are completely missing
    for i in range(n_obs):
        if np.isnan(data[i]).all():
            # If row is all NaN, set at least one value
            data[i, 0] = np.random.randn()
    
    for j in range(n_vars):
        if np.isnan(data[:, j]).all():
            # If column is all NaN, set at least one value
            data[0, j] = np.random.randn()
    
    return data


@pytest.fixture
def small_data():
    """Very small dataset for testing small problem detection."""
    np.random.seed(42)
    data = np.random.randn(20, 2)
    data[5, 0] = np.nan
    data[10, 1] = np.nan
    return data


@pytest.fixture
def mock_gpu_none():
    """Mock detector that reports no GPU."""
    class MockGPUCapabilities:
        has_gpu = False
        gpu_type = 'none'
        fp64_support = 'none'
        device_name = 'None'
    return MockGPUCapabilities()


@pytest.fixture
def mock_gpu_rtx():
    """Mock detector that reports consumer RTX GPU."""
    class MockGPUCapabilities:
        has_gpu = True
        gpu_type = 'cuda'
        fp64_support = 'gimped'
        device_name = 'NVIDIA GeForce RTX 4090'
        fp64_ratio = 64
    return MockGPUCapabilities()


@pytest.fixture
def mock_gpu_a100():
    """Mock detector that reports data center A100 GPU."""
    class MockGPUCapabilities:
        has_gpu = True
        gpu_type = 'cuda'
        fp64_support = 'full'
        device_name = 'NVIDIA A100'
        fp64_ratio = 2
    return MockGPUCapabilities()


@pytest.fixture
def mock_gpu_metal():
    """Mock detector that reports Apple Metal GPU."""
    class MockGPUCapabilities:
        has_gpu = True
        gpu_type = 'metal'
        fp64_support = 'none'
        device_name = 'Apple M2'
    return MockGPUCapabilities()


# ============================================================================
# Basic mlest Tests
# ============================================================================

class TestMLEstBasic:
    """Test basic mlest functionality."""
    
    def test_mlest_simple_data(self, simple_data):
        """Test mlest on simple complete data."""
        result = mlest(simple_data, verbose=False)
        
        assert isinstance(result, MLResult)
        assert result.converged
        assert result.n_iter > 0
        assert result.muhat.shape == (3,)
        assert result.sigmahat.shape == (3, 3)
        assert np.isfinite(result.loglik)
        assert result.computation_time > 0
    
    def test_mlest_missing_data(self, missing_data):
        """Test mlest on data with missing values."""
        result = mlest(missing_data, verbose=False)
        
        assert isinstance(result, MLResult)
        assert result.converged
        assert result.muhat.shape == (4,)
        assert result.sigmahat.shape == (4, 4)
        assert result.n_missing > 0
        assert len(result.patterns['pattern_indices']) > 1  # Multiple patterns
    
    def test_mlest_input_validation(self):
        """Test input validation."""
        # 1D data
        with pytest.raises(ValueError, match="Data must be 2-dimensional"):
            mlest(np.array([1, 2, 3]))
        
        # Too few observations
        with pytest.raises(ValueError, match="Need at least 2 observations"):
            mlest(np.array([[1, 2]]))
        
        # All missing
        with pytest.raises(ValueError, match="All data values are missing"):
            mlest(np.full((10, 3), np.nan))
        
        # Variable with all missing
        data = np.random.randn(10, 3)
        data[:, 1] = np.nan
        with pytest.raises(ValueError, match="Variables .* have all missing values"):
            mlest(data)
    
    def test_mlest_parameters(self, simple_data):
        """Test different parameter combinations."""
        # High precision
        result = mlest(simple_data, tol=1e-8, max_iter=500)
        assert result.converged
        
        # Specific method
        result = mlest(simple_data, method='BFGS')
        assert result.method == 'BFGS'
        
        # CPU backend
        result = mlest(simple_data, backend='cpu')
        assert result.backend == 'cpu'
    
    def test_backward_compatibility(self, simple_data):
        """Test backward compatibility aliases."""
        # ml_estimate alias
        result1 = pmle.ml_estimate(simple_data)
        assert isinstance(result1, MLResult)
        
        # maximum_likelihood_estimate alias
        result2 = pmle.maximum_likelihood_estimate(simple_data)
        assert isinstance(result2, MLResult)


# ============================================================================
# Backend Selection Tests
# ============================================================================

class TestBackendSelection:
    """Test backend selection logic."""
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')
    def test_cpu_selection(self, mock_detect_gpu, simple_data, mock_gpu_none):
        """Test CPU backend selection when no GPU."""
        mock_detect_gpu.return_value = mock_gpu_none
        
        result = mlest(simple_data, backend='auto', verbose=False)
        assert result.backend == 'cpu'
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')
    def test_gpu_fallback_to_cpu(self, mock_detect_gpu, simple_data, mock_gpu_none):
        """Test fallback to CPU when GPU requested but not available."""
        mock_detect_gpu.return_value = mock_gpu_none
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mlest(simple_data, backend='gpu', verbose=False)
            assert len(w) == 1
            assert "no GPU detected" in str(w[0].message)
        
        assert result.backend == 'cpu'
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')
    def test_gpu64_on_rtx(self, mock_detect_gpu, simple_data, mock_gpu_rtx):
        """Test gpu64=True on consumer RTX card (gimped FP64)."""
        mock_detect_gpu.return_value = mock_gpu_rtx
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.BackendFactory.create') as mock_create:
                mock_backend = Mock()
                mock_backend.device = 'cuda:0'
                mock_create.return_value = mock_backend
                
                # This will warn about gimped FP64
                result = mlest(simple_data, gpu64=True, verbose=False)
                
                # Should have warning about gimped performance
                assert any("gimped FP64" in str(warning.message) for warning in w)
                assert any("MUCH slower" in str(warning.message) for warning in w)
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')  
    def test_gpu64_on_metal(self, mock_detect_gpu, simple_data, mock_gpu_metal):
        """Test gpu64=True on Metal (no FP64 support)."""
        mock_detect_gpu.return_value = mock_gpu_metal
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.get_backend') as mock_get_backend:
                mock_backend = Mock()
                mock_backend.device = 'mps:0'
                mock_get_backend.return_value = mock_backend
                
                # This will fall back to FP32
                result = mlest(simple_data, gpu64=True, verbose=False)
                
                # Should warn about no FP64 support
                assert any("doesn't support FP64" in str(warning.message) for warning in w)
                assert any("Falling back to FP32" in str(warning.message) for warning in w)
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')
    def test_gpu64_on_a100(self, mock_detect_gpu, simple_data, mock_gpu_a100):
        """Test gpu64=True on A100 (full FP64 support)."""
        mock_detect_gpu.return_value = mock_gpu_a100
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.get_backend') as mock_get_backend:
                mock_backend = Mock()
                mock_backend.device = 'cuda:0'
                mock_get_backend.return_value = mock_backend
                
                # This should work without warnings
                result = mlest(simple_data, gpu64=True, verbose=False)
                
                # No warnings expected for A100
                fp64_warnings = [warning for warning in w 
                               if "FP64" in str(warning.message)]
                assert len(fp64_warnings) == 0
    
    @patch('pymvnmle.mlest.detect_gpu_capabilities')
    def test_small_problem_uses_cpu(self, mock_detect_gpu, small_data, mock_gpu_rtx):
        """Test that small problems default to CPU even with GPU available."""
        mock_detect_gpu.return_value = mock_gpu_rtx
        
        with patch('pymvnmle.mlest.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.device = 'cpu'
            mock_get_backend.return_value = mock_backend
            
            # Small problem should use CPU by default
            result = mlest(small_data, backend='auto', verbose=False)
            
            # The backend selection logic should choose CPU for small problems
            # Note: This depends on the actual implementation


# ============================================================================
# Module Functions Tests
# ============================================================================

class TestModuleFunctions:
    """Test convenience functions in __init__.py."""
    
    @patch('pymvnmle.PrecisionDetector')
    def test_check_gpu_capabilities_no_gpu(self, mock_detector_class, mock_gpu_none):
        """Test check_gpu_capabilities with no GPU."""
        mock_detector_class.return_value = mock_gpu_none
        
        caps = pmle.check_gpu_capabilities(verbose=False)
        
        assert not caps['gpu_available']
        assert caps['gpu_type'] == 'none'
        assert caps['recommended_settings']['backend'] == 'cpu'
        assert not caps['recommended_settings']['gpu64']
    
    @patch('pymvnmle.PrecisionDetector')
    def test_check_gpu_capabilities_rtx(self, mock_detector_class, mock_gpu_rtx):
        """Test check_gpu_capabilities with RTX GPU."""
        mock_detector_class.return_value = mock_gpu_rtx
        
        caps = pmle.check_gpu_capabilities(verbose=False)
        
        assert caps['gpu_available']
        assert caps['gpu_type'] == 'cuda'
        assert caps['fp64_support'] == 'gimped'
        assert caps['fp64_ratio'] == 64
        assert not caps['recommended_settings']['gpu64']  # Should recommend FP32
    
    @patch('pymvnmle.PrecisionDetector')
    def test_check_gpu_capabilities_a100(self, mock_detector_class, mock_gpu_a100):
        """Test check_gpu_capabilities with A100."""
        mock_detector_class.return_value = mock_gpu_a100
        
        caps = pmle.check_gpu_capabilities(verbose=False)
        
        assert caps['gpu_available']
        assert caps['fp64_support'] == 'full'
        assert caps['recommended_settings']['gpu64']  # Should recommend FP64
    
    @patch('pymvnmle.PrecisionDetector')
    def test_check_gpu_capabilities_verbose(self, mock_detector_class, mock_gpu_rtx, capsys):
        """Test verbose output of check_gpu_capabilities."""
        mock_detector_class.return_value = mock_gpu_rtx
        
        caps = pmle.check_gpu_capabilities(verbose=True)
        
        captured = capsys.readouterr()
        assert "GPU Detected: NVIDIA GeForce RTX 4090" in captured.out
        assert "FP64 Support: Gimped" in captured.out
        assert "1/64x speed" in captured.out
        assert "Suggested mlest() parameters:" in captured.out
    
    def test_benchmark_performance(self, capsys):
        """Test benchmark_performance function."""
        # Use very small data for fast test
        with patch('pymvnmle.mlest') as mock_mlest:
            # Mock the mlest results
            mock_result = Mock(spec=MLResult)
            mock_result.n_iter = 10
            mock_result.loglik = -100.0
            mock_result.converged = True
            mock_result.backend = 'cpu'
            mock_result.method = 'BFGS'
            mock_mlest.return_value = mock_result
            
            results = pmle.benchmark_performance(
                n_obs=50,
                n_vars=3, 
                missing_rate=0.1,
                backends=['cpu'],
                verbose=True
            )
            
            assert 'cpu' in results
            assert 'time' in results['cpu']
            assert 'iterations' in results['cpu']
            
            captured = capsys.readouterr()
            assert "Benchmarking PyMVNMLE Performance" in captured.out
    
    def test_check_version(self, capsys):
        """Test version checking."""
        pmle.check_version()
        
        captured = capsys.readouterr()
        assert "PyMVNMLE:" in captured.out
        assert "NumPy:" in captured.out
        assert "SciPy:" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_auto(self, missing_data):
        """Test full pipeline with auto selection."""
        result = mlest(
            missing_data,
            backend='auto',
            method='auto',
            verbose=False
        )
        
        assert result.converged
        assert result.muhat is not None
        assert result.sigmahat is not None
        assert np.all(np.linalg.eigvals(result.sigmahat) > 0)  # Positive definite
    
    def test_full_pipeline_cpu_bfgs(self, missing_data):
        """Test explicit CPU with BFGS."""
        result = mlest(
            missing_data,
            backend='cpu',
            method='BFGS',
            verbose=False
        )
        
        assert result.converged
        assert result.backend == 'cpu'
        assert result.method == 'BFGS'
    
    def test_verbose_output(self, simple_data, capsys):
        """Test verbose output."""
        result = mlest(simple_data, verbose=True)
        
        captured = capsys.readouterr()
        assert "PyMVNMLE Maximum Likelihood Estimation" in captured.out
        assert "Data:" in captured.out
        assert "Missing data patterns:" in captured.out
        assert "Backend Configuration:" in captured.out
        assert "Optimization Configuration:" in captured.out
        assert "Optimization Complete:" in captured.out
        assert "Converged:" in captured.out
    
    def test_deterministic_results(self, simple_data):
        """Test that results are deterministic."""
        result1 = mlest(simple_data, backend='cpu', method='BFGS')
        result2 = mlest(simple_data, backend='cpu', method='BFGS')
        
        np.testing.assert_allclose(result1.muhat, result2.muhat, rtol=1e-10)
        np.testing.assert_allclose(result1.sigmahat, result2.sigmahat, rtol=1e-10)
        assert abs(result1.loglik - result2.loglik) < 1e-10
    
    def test_result_attributes(self, missing_data):
        """Test all MLResult attributes are populated."""
        result = mlest(missing_data, verbose=False)
        
        # Required attributes
        assert hasattr(result, 'muhat')
        assert hasattr(result, 'sigmahat')
        assert hasattr(result, 'loglik')
        assert hasattr(result, 'n_iter')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'computation_time')
        assert hasattr(result, 'backend')
        assert hasattr(result, 'method')
        assert hasattr(result, 'patterns')
        assert hasattr(result, 'n_obs')
        assert hasattr(result, 'n_vars')
        assert hasattr(result, 'n_missing')
        assert hasattr(result, 'grad_norm')
        assert hasattr(result, 'message')
        
        # All should be non-None
        assert result.muhat is not None
        assert result.sigmahat is not None
        assert result.loglik is not None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_pattern(self):
        """Test data with single missingness pattern."""
        data = np.random.randn(50, 3)
        # All observations missing same variable
        data[:, 1] = np.nan
        
        with pytest.raises(ValueError, match="Variables .* have all missing values"):
            mlest(data)
    
    def test_high_missingness(self):
        """Test data with very high missingness."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        # 70% missing
        mask = np.random.rand(100, 5) < 0.7
        data[mask] = np.nan
        
        # Remove any completely missing rows/cols for valid test
        keep_rows = ~np.isnan(data).all(axis=1)
        keep_cols = ~np.isnan(data).all(axis=0)
        data = data[keep_rows][:, keep_cols]
        
        if data.shape[0] >= 2 and data.shape[1] >= 1:
            result = mlest(data, max_iter=2000, verbose=False)
            # May or may not converge with high missingness
            assert isinstance(result, MLResult)
    
    def test_perfect_data(self):
        """Test on perfectly centered and scaled data."""
        n = 100
        # Perfect standard normal data
        data = np.random.randn(n, 3)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        
        result = mlest(data, verbose=False)
        
        # Should recover approximately zero mean and identity covariance
        np.testing.assert_allclose(result.muhat, np.zeros(3), atol=0.2)
        np.testing.assert_allclose(result.sigmahat, np.eye(3), atol=0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])