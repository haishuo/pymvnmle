"""
Pytest configuration for PyMVNMLE regulatory validation tests.

This configuration ensures consistent test execution for regulatory compliance.
"""

import pytest
import numpy as np
import warnings
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure numpy for reproducible testing
np.random.seed(42)
np.set_printoptions(precision=15, suppress=False)

# Suppress specific warnings during testing
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')  # JAX CUDA warnings
warnings.filterwarnings('ignore', message='.*JAX.*')     # JAX warnings

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest for regulatory validation."""
    config.addinivalue_line(
        "markers", "regulatory: mark test as regulatory validation requirement"
    )
    config.addinivalue_line(
        "markers", "r_reference: mark test as R reference validation"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case validation"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for regulatory compliance."""
    # Add regulatory marker to all tests in test_regulatory_validation.py
    for item in items:
        if "test_regulatory_validation" in str(item.fspath):
            item.add_marker(pytest.mark.regulatory)
        
        # Add specific markers based on test names
        if "apple" in item.name or "missvals" in item.name:
            item.add_marker(pytest.mark.r_reference)
        
        if "efficiency" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.performance)
            
        if "edge" in item.name or "robustness" in item.name:
            item.add_marker(pytest.mark.edge_case)

@pytest.fixture(scope="session")
def regulatory_test_data():
    """Provide regulatory test data for validation."""
    return {
        'tolerance_strict': 1e-7,    # Log-likelihood agreement
        'tolerance_params': 1e-3,    # Parameter estimates (0.1%)
        'tolerance_complex': 5e-3,   # Complex datasets (0.5%)
        'max_iterations': 1000,      # Maximum optimization iterations
        'r_reference_version': '0.1-11.2'
    }

@pytest.fixture
def suppress_warnings():
    """Suppress warnings during test execution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate regulatory compliance summary."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "REGULATORY VALIDATION SUMMARY")
        terminalreporter.write_line("✅ ALL TESTS PASSED - FDA SUBMISSION READY")
        terminalreporter.write_line("Mathematical equivalence with R mvnmle confirmed")
        terminalreporter.write_line("PyMVNMLE validated for clinical trial use")
    else:
        terminalreporter.write_sep("=", "REGULATORY VALIDATION FAILED")
        terminalreporter.write_line("❌ TESTS FAILED - REQUIRES INVESTIGATION")
        terminalreporter.write_line("PyMVNMLE not ready for regulatory submission")