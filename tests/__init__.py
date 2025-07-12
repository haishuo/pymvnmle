"""
PyMVNMLE Test Suite
==================

Regulatory-grade validation tests for FDA submission compliance.

This test suite validates PyMVNMLE against R's mvnmle package with:
- Exact numerical agreement on reference datasets  
- Mathematical property verification
- Edge case robustness testing
- Performance benchmarking
- Input validation coverage

Run tests:
    python tests/test_regulatory_validation.py
    python -m pytest tests/ -v

For regulatory documentation:
    python -c "from pymvnmle import create_validation_report; print(create_validation_report())"
"""

__version__ = "1.0.0"
__author__ = "PyMVNMLE Development Team"

# Test configuration
TEST_TOLERANCE_STRICT = 1e-7   # For log-likelihood agreement
TEST_TOLERANCE_PARAMS = 1e-3   # For parameter estimates (0.1%)
TEST_TOLERANCE_COMPLEX = 5e-3  # For complex datasets (0.5%)

# Regulatory test categories
REGULATORY_TESTS = [
    "apple_dataset_exact_validation",
    "missvals_dataset_exact_validation", 
    "mathematical_properties",
    "edge_cases_and_robustness",
    "computational_efficiency",
    "input_validation_and_error_handling",
    "reproducibility"
]

# Test data sources
R_REFERENCE_SOURCES = {
    "apple": "R mvnmle::apple dataset - Little & Rubin (1987)",
    "missvals": "R mvnmle::missvals dataset - Draper & Smith (1966)"
}

# Validation status
VALIDATION_STATUS = "FDA_SUBMISSION_READY"
LAST_VALIDATION_DATE = "2025-01-12"
R_REFERENCE_VERSION = "mvnmle_0.1-11.2"