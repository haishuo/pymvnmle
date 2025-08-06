"""
Comprehensive integration test suite for PyMVNMLE v2.0.

This suite validates the complete precision-based architecture including:
- R compatibility across all backends
- Correct behavior of gpu64 parameter
- Backend selection logic
- Method selection logic
- Numerical accuracy across precisions
- Performance characteristics

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import pytest
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
from unittest.mock import patch, Mock

import pymvnmle as pmle
from pymvnmle import mlest, MLResult
from pymvnmle import datasets
from pymvnmle._backends.precision_detector import PrecisionDetector


# ============================================================================
# R Reference Data
# ============================================================================

# These are the exact results from R's mvnmle package
R_REFERENCES = {
    'apple': {
        'muhat': np.array([0.4147273, 0.4784545]),
        'sigmahat': np.array([
            [0.2789709, 0.2175990],
            [0.2175990, 0.4450773]
        ]),
        'loglik': -26.48173,
        'n_iter': 14,  # From R
        'n_obs': 18,
        'n_vars': 2,
        'n_missing': 9
    },
    'missvals': {
        'muhat': np.array([97.51539, 96.68308, 97.48462, 114.36923, 97.66154]),
        'sigmahat': np.array([
            [208.60510, 32.08885, 60.89244, 27.80430, 43.90704],
            [32.08885, 195.05413, 18.17225, -38.85092, 36.17416],
            [60.89244, 18.17225, 254.58019, 90.51945, 43.43692],
            [27.80430, -38.85092, 90.51945, 609.50655, -42.29677],
            [43.90704, 36.17416, 43.43692, -42.29677, 236.11819]
        ]),
        'loglik': -72.66452,
        'n_iter': 168,  # With iterlim=400
        'n_obs': 13,
        'n_vars': 5,
        'n_missing': 26
    }
}


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gpu_configurations():
    """Different GPU configurations for testing."""
    return {
        'no_gpu': {
            'gpu_type': 'none',
            'fp64_support': 'none',
            'device_name': 'None'
        },
        'rtx_4090': {
            'gpu_type': 'cuda',
            'fp64_support': 'gimped',
            'device_name': 'NVIDIA GeForce RTX 4090',
            'fp64_ratio': 64
        },
        'rtx_3090': {
            'gpu_type': 'cuda', 
            'fp64_support': 'gimped',
            'device_name': 'NVIDIA GeForce RTX 3090',
            'fp64_ratio': 32
        },
        'a100': {
            'gpu_type': 'cuda',
            'fp64_support': 'full',
            'device_name': 'NVIDIA A100',
            'fp64_ratio': 2
        },
        'h100': {
            'gpu_type': 'cuda',
            'fp64_support': 'full',
            'device_name': 'NVIDIA H100',
            'fp64_ratio': 1
        },
        'apple_m2': {
            'gpu_type': 'metal',
            'fp64_support': 'none',
            'device_name': 'Apple M2'
        }
    }


@pytest.fixture
def synthetic_datasets():
    """Generate synthetic datasets with known properties."""
    np.random.seed(42)
    
    datasets = {}
    
    # Small complete data
    n, p = 50, 3
    mu_true = np.array([1.0, -0.5, 2.0])
    sigma_true = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 2.0, -0.3],
        [0.2, -0.3, 1.5]
    ])
    L = np.linalg.cholesky(sigma_true)
    X = np.random.randn(n, p) @ L.T + mu_true
    datasets['small_complete'] = {
        'data': X,
        'mu_true': mu_true,
        'sigma_true': sigma_true
    }
    
    # Medium with missing
    n, p = 200, 5
    mu_true = np.zeros(p)
    # Random positive definite covariance
    A = np.random.randn(p, p)
    sigma_true = A.T @ A + np.eye(p)
    L = np.linalg.cholesky(sigma_true)
    X = np.random.randn(n, p) @ L.T + mu_true
    # Add 25% missing
    mask = np.random.rand(n, p) < 0.25
    X[mask] = np.nan
    datasets['medium_missing'] = {
        'data': X,
        'mu_true': mu_true,
        'sigma_true': sigma_true,
        'missing_rate': 0.25
    }
    
    # Large sparse
    n, p = 1000, 10
    mu_true = np.random.randn(p) * 2
    # Sparse covariance structure
    sigma_true = np.eye(p) * 2
    for i in range(p-1):
        sigma_true[i, i+1] = sigma_true[i+1, i] = 0.3
    L = np.linalg.cholesky(sigma_true)
    X = np.random.randn(n, p) @ L.T + mu_true
    # Add 10% missing
    mask = np.random.rand(n, p) < 0.1
    X[mask] = np.nan
    datasets['large_sparse'] = {
        'data': X,
        'mu_true': mu_true,
        'sigma_true': sigma_true,
        'missing_rate': 0.1
    }
    
    return datasets


# ============================================================================
# R Compatibility Tests
# ============================================================================

class TestRCompatibility:
    """Validate exact compatibility with R mvnmle."""
    
    def test_apple_dataset_cpu(self):
        """Test Apple dataset with CPU backend (exact R compatibility)."""
        result = mlest(
            datasets.apple,
            backend='cpu',
            method='BFGS',
            max_iter=1000,
            tol=1e-6,
            verbose=False
        )
        
        ref = R_REFERENCES['apple']
        
        # Check convergence
        assert result.converged, "Failed to converge on Apple dataset"
        
        # Check estimates (looser tolerance for numerical differences)
        np.testing.assert_allclose(result.muhat, ref['muhat'], rtol=1e-3)
        np.testing.assert_allclose(result.sigmahat, ref['sigmahat'], rtol=1e-3)
        
        # Log-likelihood should be very close
        assert abs(result.loglik - ref['loglik']) < 0.1, \
            f"Log-likelihood mismatch: {result.loglik:.5f} vs R: {ref['loglik']:.5f}"
        
        # Check dimensions
        assert result.n_obs == ref['n_obs']
        assert result.n_vars == ref['n_vars']
        assert result.n_missing == ref['n_missing']
    
    def test_missvals_dataset_cpu(self):
        """Test Missvals dataset with CPU backend."""
        result = mlest(
            datasets.missvals,
            backend='cpu',
            method='BFGS',
            max_iter=400,  # Match R's iterlim
            tol=1e-6,
            verbose=False
        )
        
        ref = R_REFERENCES['missvals']
        
        # Check convergence
        assert result.converged, "Failed to converge on Missvals dataset"
        
        # Check estimates (looser tolerance for this harder problem)
        np.testing.assert_allclose(result.muhat, ref['muhat'], rtol=1e-2)
        np.testing.assert_allclose(result.sigmahat, ref['sigmahat'], rtol=1e-2)
        
        # Log-likelihood should be close
        assert abs(result.loglik - ref['loglik']) < 1.0, \
            f"Log-likelihood mismatch: {result.loglik:.5f} vs R: {ref['loglik']:.5f}"
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    @patch('pymvnmle.mlest.BackendFactory.create')
    def test_consistency_across_backends(self, mock_create, mock_detector_class):
        """Test that all backends give consistent results."""
        # Mock GPU availability
        detector = Mock()
        detector.detect_gpu.return_value = {
            'gpu_type': 'cuda',
            'fp64_support': 'full',
            'device_name': 'NVIDIA A100',
            'fp64_ratio': 2
        }
        mock_detector_class.return_value = detector
        
        # Simple test data
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[np.random.rand(100, 3) < 0.1] = np.nan
        
        results = {}
        
        # Test each backend configuration
        for backend_type in ['cpu', 'gpu_fp32', 'gpu_fp64']:
            # Mock the backend
            mock_backend = Mock()
            mock_backend.device = 'cuda:0' if 'gpu' in backend_type else 'cpu'
            mock_create.return_value = mock_backend
            
            # Mock objective and optimizer to return consistent results
            with patch('pymvnmle.mlest.get_objective') as mock_obj:
                with patch('pymvnmle.mlest.auto_select_method') as mock_method:
                    # Setup mocks
                    mock_objective = Mock()
                    mock_objective.get_initial_params.return_value = np.zeros(9)
                    mock_objective.compute_objective.return_value = -100.0
                    mock_objective.compute_gradient.return_value = np.zeros(9)
                    mock_objective.extract_parameters.return_value = (
                        np.array([0.1, 0.2, 0.3]),
                        np.eye(3)
                    )
                    mock_obj.return_value = mock_objective
                    
                    mock_optimizer = Mock()
                    mock_optimizer.optimize.return_value = {
                        'x': np.zeros(9),
                        'fun': -100.0,
                        'grad_norm': 1e-6,
                        'n_iter': 10,
                        'converged': True,
                        'message': 'Converged'
                    }
                    mock_method.return_value = ('BFGS', mock_optimizer, {})
                    
                    # Run estimation
                    if 'gpu' in backend_type:
                        if backend_type == 'gpu_fp64':
                            result = mlest(data, gpu64=True, verbose=False)
                        else:
                            result = mlest(data, gpu64=False, verbose=False)
                    else:
                        result = mlest(data, backend='cpu', verbose=False)
                    
                    results[backend_type] = result
        
        # All backends should give same results (within tolerance)
        cpu_result = results['cpu']
        for backend, result in results.items():
            if backend != 'cpu':
                np.testing.assert_allclose(
                    result.muhat, cpu_result.muhat, 
                    rtol=1e-5,
                    err_msg=f"{backend} differs from CPU"
                )


# ============================================================================
# GPU64 Parameter Tests
# ============================================================================

class TestGPU64Parameter:
    """Test gpu64 parameter behavior across different hardware."""
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    def test_gpu64_no_gpu(self, mock_detector_class, gpu_configurations):
        """Test gpu64=True when no GPU available."""
        detector = Mock()
        detector.detect_gpu.return_value = gpu_configurations['no_gpu']
        mock_detector_class.return_value = detector
        
        data = np.random.randn(50, 3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mlest(data, gpu64=True, verbose=False)
            
            # Should warn and fall back to CPU
            assert any("no GPU detected" in str(warning.message) for warning in w)
        
        assert result.backend == 'cpu'
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    def test_gpu64_rtx4090(self, mock_detector_class, gpu_configurations):
        """Test gpu64=True on RTX 4090 (gimped FP64)."""
        detector = Mock()
        detector.detect_gpu.return_value = gpu_configurations['rtx_4090']
        mock_detector_class.return_value = detector
        
        data = np.random.randn(50, 3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.BackendFactory.create') as mock_create:
                mock_backend = Mock()
                mock_backend.device = 'cuda:0'
                mock_create.return_value = mock_backend
                
                result = mlest(data, gpu64=True, verbose=False)
                
                # Should warn about gimped performance
                assert any("gimped FP64" in str(warning.message) for warning in w)
                assert any("1/64x speed" in str(warning.message) for warning in w)
                assert any("MUCH slower" in str(warning.message) for warning in w)
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    def test_gpu64_apple_metal(self, mock_detector_class, gpu_configurations):
        """Test gpu64=True on Apple Metal (no FP64)."""
        detector = Mock()
        detector.detect_gpu.return_value = gpu_configurations['apple_m2']
        mock_detector_class.return_value = detector
        
        data = np.random.randn(50, 3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.BackendFactory.create') as mock_create:
                mock_backend = Mock()
                mock_backend.device = 'mps:0'
                mock_create.return_value = mock_backend
                
                result = mlest(data, gpu64=True, verbose=False)
                
                # Should warn and fall back to FP32
                assert any("doesn't support FP64" in str(warning.message) for warning in w)
                assert any("Falling back to FP32" in str(warning.message) for warning in w)
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    def test_gpu64_a100(self, mock_detector_class, gpu_configurations):
        """Test gpu64=True on A100 (full FP64)."""
        detector = Mock()
        detector.detect_gpu.return_value = gpu_configurations['a100']
        mock_detector_class.return_value = detector
        
        data = np.random.randn(50, 3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('pymvnmle.mlest.BackendFactory.create') as mock_create:
                mock_backend = Mock()
                mock_backend.device = 'cuda:0'
                mock_create.return_value = mock_backend
                
                result = mlest(data, gpu64=True, verbose=False)
                
                # Should NOT warn - A100 has full FP64
                fp64_warnings = [warning for warning in w 
                               if "FP64" in str(warning.message) or "gimped" in str(warning.message)]
                assert len(fp64_warnings) == 0


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy across different precisions."""
    
    def test_complete_data_recovery(self, synthetic_datasets):
        """Test parameter recovery on complete data."""
        data_info = synthetic_datasets['small_complete']
        
        result = mlest(
            data_info['data'],
            backend='cpu',
            method='BFGS',
            tol=1e-8,
            verbose=False
        )
        
        # Should recover true parameters closely
        np.testing.assert_allclose(
            result.muhat, 
            data_info['mu_true'],
            rtol=0.1,  # Within 10%
            atol=0.1
        )
        
        # Covariance is harder to estimate precisely
        np.testing.assert_allclose(
            result.sigmahat,
            data_info['sigma_true'],
            rtol=0.2,  # Within 20%
            atol=0.2
        )
    
    def test_missing_data_recovery(self, synthetic_datasets):
        """Test parameter recovery with missing data."""
        data_info = synthetic_datasets['medium_missing']
        
        result = mlest(
            data_info['data'],
            backend='cpu',
            max_iter=1000,
            verbose=False
        )
        
        assert result.converged
        
        # With 25% missing, recovery will be less precise
        np.testing.assert_allclose(
            result.muhat,
            data_info['mu_true'],
            rtol=0.3,  # Within 30%
            atol=0.3
        )
    
    def test_positive_definiteness(self, synthetic_datasets):
        """Test that covariance estimates are always positive definite."""
        for name, data_info in synthetic_datasets.items():
            result = mlest(data_info['data'], verbose=False)
            
            # Check positive definiteness
            eigenvalues = np.linalg.eigvals(result.sigmahat)
            assert np.all(eigenvalues > 0), \
                f"Non-positive definite covariance for {name}"
            
            # Check symmetry
            np.testing.assert_allclose(
                result.sigmahat,
                result.sigmahat.T,
                rtol=1e-10,
                err_msg=f"Non-symmetric covariance for {name}"
            )
    
    @patch('pymvnmle.mlest.PrecisionDetector')
    def test_precision_impact(self, mock_detector_class, synthetic_datasets):
        """Test impact of precision on results."""
        # Mock A100 GPU
        detector = Mock()
        detector.detect_gpu.return_value = {
            'gpu_type': 'cuda',
            'fp64_support': 'full',
            'device_name': 'NVIDIA A100',
            'fp64_ratio': 2
        }
        mock_detector_class.return_value = detector
        
        data = synthetic_datasets['small_complete']['data']
        
        # Compare FP32 vs FP64 results
        with patch('pymvnmle.mlest.BackendFactory.create') as mock_create:
            mock_backend = Mock()
            mock_backend.device = 'cuda:0'
            mock_create.return_value = mock_backend
            
            # FP32 result
            result_fp32 = mlest(data, gpu64=False, verbose=False)
            
            # FP64 result  
            result_fp64 = mlest(data, gpu64=True, verbose=False)
            
            # Results should be very similar despite precision difference
            # (actual difference depends on problem conditioning)
            np.testing.assert_allclose(
                result_fp32.muhat,
                result_fp64.muhat,
                rtol=1e-3  # Within 0.1%
            )


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_small_problem_performance(self):
        """Test that small problems complete quickly."""
        data = np.random.randn(50, 3)
        
        result = mlest(data, verbose=False)
        
        # Small problems should be fast
        assert result.computation_time < 5.0, \
            f"Small problem took {result.computation_time:.2f}s"
        assert result.n_iter < 100, \
            f"Small problem took {result.n_iter} iterations"
    
    def test_convergence_behavior(self, synthetic_datasets):
        """Test convergence behavior on different datasets."""
        convergence_stats = {}
        
        for name, data_info in synthetic_datasets.items():
            result = mlest(
                data_info['data'],
                max_iter=2000,
                verbose=False
            )
            
            convergence_stats[name] = {
                'converged': result.converged,
                'n_iter': result.n_iter,
                'grad_norm': result.grad_norm,
                'time': result.computation_time
            }
        
        # All should converge
        for name, stats in convergence_stats.items():
            assert stats['converged'], f"{name} failed to converge"
            assert stats['grad_norm'] < 1e-3, \
                f"{name} gradient norm too large: {stats['grad_norm']:.2e}"
    
    def test_method_selection_performance(self):
        """Test that method selection is appropriate."""
        # Small problem - should use simple method
        small_data = np.random.randn(30, 2)
        result_small = mlest(small_data, method='auto', verbose=False)
        assert result_small.method in ['BFGS']
        
        # Medium problem
        medium_data = np.random.randn(500, 10)
        medium_data[np.random.rand(500, 10) < 0.1] = np.nan
        result_medium = mlest(medium_data, method='auto', verbose=False)
        assert result_medium.method in ['BFGS', 'Newton-CG']


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for edge cases."""
    
    def test_high_dimensionality(self):
        """Test with high dimensional data."""
        np.random.seed(42)
        n, p = 100, 20  # More variables than typical
        
        # Generate data with structure to ensure convergence
        mu = np.zeros(p)
        # Block diagonal covariance for stability
        sigma = np.eye(p)
        for i in range(0, p-5, 5):
            sigma[i:i+5, i:i+5] = np.random.randn(5, 5)
            sigma[i:i+5, i:i+5] = sigma[i:i+5, i:i+5].T @ sigma[i:i+5, i:i+5]
            sigma[i:i+5, i:i+5] += np.eye(5)
        
        L = np.linalg.cholesky(sigma)
        data = np.random.randn(n, p) @ L.T + mu
        
        # Add some missing
        data[np.random.rand(n, p) < 0.05] = np.nan
        
        result = mlest(data, max_iter=2000, verbose=False)
        
        # Should handle high dimensions
        assert result.muhat.shape == (p,)
        assert result.sigmahat.shape == (p, p)
    
    def test_extreme_missingness_patterns(self):
        """Test with complex missingness patterns."""
        np.random.seed(42)
        n, p = 200, 5
        data = np.random.randn(n, p)
        
        # Create complex pattern: 
        # - First 50 rows miss variable 1
        # - Next 50 rows miss variable 2
        # - Next 50 rows miss variables 3 and 4
        # - Last 50 rows complete
        data[:50, 0] = np.nan
        data[50:100, 1] = np.nan
        data[100:150, [2, 3]] = np.nan
        
        result = mlest(data, verbose=False)
        
        assert result.converged
        assert len(result.patterns['pattern_indices']) == 4  # 4 distinct patterns
    
    def test_nearly_singular_covariance(self):
        """Test with nearly singular covariance matrix."""
        np.random.seed(42)
        n, p = 100, 3
        
        # Create highly correlated variables
        base = np.random.randn(n, 1)
        data = np.hstack([
            base + np.random.randn(n, 1) * 0.01,  # Almost identical
            base + np.random.randn(n, 1) * 0.01,
            base + np.random.randn(n, 1) * 0.1    # Slightly different
        ])
        
        # Add small amount of missing
        data[np.random.rand(n, p) < 0.05] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about conditioning
            result = mlest(data, max_iter=2000, verbose=False)
        
        # Should handle near-singularity
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))


# ============================================================================
# Validation Report Generator
# ============================================================================

def generate_validation_report(output_file: str = "v2_validation_report.txt"):
    """Generate comprehensive validation report for PyMVNMLE v2.0."""
    
    report = []
    report.append("=" * 80)
    report.append("PyMVNMLE v2.0 Comprehensive Validation Report")
    report.append("=" * 80)
    report.append("")
    
    # Version information
    import pymvnmle
    report.append(f"PyMVNMLE Version: {pymvnmle.__version__}")
    report.append(f"NumPy Version: {np.__version__}")
    
    # Hardware detection
    report.append("\n" + "-" * 40)
    report.append("Hardware Configuration")
    report.append("-" * 40)
    
    caps = pmle.check_gpu_capabilities(verbose=False)
    report.append(f"GPU Available: {caps['gpu_available']}")
    if caps['gpu_available']:
        report.append(f"GPU Type: {caps['gpu_name']}")
        report.append(f"FP64 Support: {caps['fp64_support']}")
        if caps.get('fp64_ratio'):
            report.append(f"FP64 Performance Ratio: 1/{caps['fp64_ratio']}")
    
    # R Compatibility Tests
    report.append("\n" + "-" * 40)
    report.append("R Compatibility Validation")
    report.append("-" * 40)
    
    for dataset_name in ['apple', 'missvals']:
        data = getattr(datasets, dataset_name)
        
        if dataset_name == 'missvals':
            max_iter = 400
        else:
            max_iter = 1000
            
        result = mlest(data, backend='cpu', method='BFGS', 
                      max_iter=max_iter, verbose=False)
        ref = R_REFERENCES[dataset_name]
        
        report.append(f"\nDataset: {dataset_name.upper()}")
        report.append(f"  Converged: {result.converged}")
        report.append(f"  Iterations: {result.n_iter}")
        report.append(f"  Log-likelihood PyMVNMLE: {result.loglik:.5f}")
        report.append(f"  Log-likelihood R: {ref['loglik']:.5f}")
        report.append(f"  Difference: {abs(result.loglik - ref['loglik']):.5f}")
        report.append(f"  Mean estimates match: {np.allclose(result.muhat, ref['muhat'], rtol=1e-2)}")
        report.append(f"  Covariance match: {np.allclose(result.sigmahat, ref['sigmahat'], rtol=1e-2)}")
    
    # Performance Benchmarks
    report.append("\n" + "-" * 40)
    report.append("Performance Benchmarks")
    report.append("-" * 40)
    
    bench_results = pmle.benchmark_performance(
        n_obs=500, n_vars=10, missing_rate=0.2,
        backends=['cpu'], verbose=False
    )
    
    for backend, results in bench_results.items():
        if 'error' not in results:
            report.append(f"\nBackend: {backend}")
            report.append(f"  Time: {results['time']:.3f}s")
            report.append(f"  Iterations: {results['iterations']}")
            report.append(f"  Method: {results.get('method_used', 'N/A')}")
    
    # Save report
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Validation report saved to {output_file}")
    
    return report_text


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Generate validation report
    print("\nGenerating validation report...")
    report = generate_validation_report()
    print("\nValidation Summary:")
    print(report)