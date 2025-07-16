"""
Test suite for pattern analysis functionality in PyMVNMLE v1.5
=============================================================

This test suite validates the new pattern analysis module and ensures
proper integration with the MCAR test functionality.

Tests cover:
- Basic pattern identification
- Pattern summary statistics
- Integration with MCAR test
- Edge cases and error handling
- Backward compatibility

Author: PyMVNMLE Development Team
Date: January 2025
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle import analyze_patterns, pattern_summary, little_mcar_test
from pymvnmle.patterns import PatternInfo, PatternSummary, describe_patterns
from pymvnmle import datasets


class TestPatternAnalysis:
    """Test suite for pattern analysis functionality."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Simple test data with known patterns
        self.simple_data = np.array([
            [1.0, 2.0, 3.0],      # Complete case
            [4.0, np.nan, 6.0],   # Missing middle
            [7.0, 8.0, np.nan],   # Missing last
            [10.0, 11.0, 12.0],   # Complete case
            [np.nan, 14.0, 15.0], # Missing first
            [16.0, np.nan, 18.0]  # Missing middle
        ])
        
        # Expected patterns for simple_data:
        # Pattern 1: Complete cases (rows 0, 3) - 2 cases
        # Pattern 2: Missing middle (rows 1, 5) - 2 cases  
        # Pattern 3: Missing last (row 2) - 1 case
        # Pattern 4: Missing first (row 4) - 1 case
        
    def test_basic_pattern_identification(self):
        """Test basic pattern identification functionality."""
        patterns = analyze_patterns(self.simple_data)
        
        # Should identify 4 unique patterns
        assert len(patterns) == 4
        
        # Check pattern ordering (most common first)
        assert patterns[0].n_cases >= patterns[1].n_cases
        assert patterns[1].n_cases >= patterns[2].n_cases
        assert patterns[2].n_cases >= patterns[3].n_cases
        
        # Check total cases
        total_cases = sum(p.n_cases for p in patterns)
        assert total_cases == 6  # Total rows in simple_data
        
        # Check pattern IDs are sequential
        pattern_ids = [p.pattern_id for p in patterns]
        assert pattern_ids == [1, 2, 3, 4]
        
    def test_pattern_info_properties(self):
        """Test PatternInfo object properties."""
        patterns = analyze_patterns(self.simple_data)
        
        for pattern in patterns:
            # Basic properties
            assert isinstance(pattern.pattern_id, int)
            assert pattern.pattern_id >= 1
            assert isinstance(pattern.n_cases, int)
            assert pattern.n_cases >= 1
            
            # Array properties
            assert isinstance(pattern.observed_indices, np.ndarray)
            assert isinstance(pattern.missing_indices, np.ndarray)
            assert isinstance(pattern.data, np.ndarray)
            assert isinstance(pattern.pattern_vector, np.ndarray)
            
            # Consistency checks
            assert len(pattern.observed_indices) == pattern.n_observed
            assert len(pattern.missing_indices) == pattern.n_missing
            assert pattern.n_observed + pattern.n_missing == 3  # Total variables
            assert pattern.data.shape[0] == pattern.n_cases
            assert pattern.data.shape[1] == pattern.n_observed
            
            # Percentage should be positive
            assert pattern.percent_cases > 0
            assert pattern.percent_cases <= 100
            
    def test_pattern_summary_statistics(self):
        """Test pattern summary computation."""
        patterns = analyze_patterns(self.simple_data)
        summary = pattern_summary(patterns, data_shape=self.simple_data.shape)
        
        # Basic summary properties
        assert isinstance(summary, PatternSummary)
        assert summary.n_patterns == 4
        assert summary.total_cases == 6
        
        # Most common pattern should be one of the 2-case patterns
        assert summary.most_common_pattern.n_cases == 2
        
        # Missing rate calculation
        # Simple data: 6 total values missing out of 18 total (6×3)
        expected_missing_rate = 6 / 18
        assert abs(summary.overall_missing_rate - expected_missing_rate) < 0.01
        
        # Variable-specific missing rates
        assert len(summary.variable_missing_rates) == 3
        for var_idx, missing_rate in summary.variable_missing_rates.items():
            assert 0 <= missing_rate <= 1
            
    def test_pandas_dataframe_input(self):
        """Test pattern analysis with pandas DataFrame input."""
        df = pd.DataFrame(self.simple_data, columns=['A', 'B', 'C'])
        
        # Should work with DataFrame
        patterns = analyze_patterns(df)
        assert len(patterns) == 4
        
        # Results should be identical to NumPy array
        np_patterns = analyze_patterns(self.simple_data)
        
        for i, (pd_pattern, np_pattern) in enumerate(zip(patterns, np_patterns)):
            assert pd_pattern.n_cases == np_pattern.n_cases
            assert pd_pattern.n_observed == np_pattern.n_observed
            np.testing.assert_array_equal(pd_pattern.observed_indices, np_pattern.observed_indices)
            np.testing.assert_array_equal(pd_pattern.missing_indices, np_pattern.missing_indices)
            
    def test_apple_dataset_patterns(self):
        """Test pattern analysis on Apple dataset."""
        patterns = analyze_patterns(datasets.apple)
        
        # Apple dataset should have 2 patterns
        assert len(patterns) == 2
        
        # Pattern 1: Complete cases (both variables observed)
        # Pattern 2: Missing second variable (only first variable observed)
        complete_pattern = next(p for p in patterns if p.n_missing == 0)
        missing_pattern = next(p for p in patterns if p.n_missing == 1)
        
        assert complete_pattern.n_observed == 2
        assert missing_pattern.n_observed == 1
        
        # Check case counts (from Apple dataset structure)
        assert complete_pattern.n_cases == 12  # Complete cases
        assert missing_pattern.n_cases == 6   # Missing second variable
        
    def test_missvals_dataset_patterns(self):
        """Test pattern analysis on Missvals dataset."""
        patterns = analyze_patterns(datasets.missvals)
        
        # Missvals should have multiple patterns
        assert len(patterns) >= 3
        
        # All patterns should have at least 1 case
        for pattern in patterns:
            assert pattern.n_cases >= 1
            
        # Total cases should equal dataset size
        total_cases = sum(p.n_cases for p in patterns)
        assert total_cases == datasets.missvals.shape[0]
        
    def test_complete_data_edge_case(self):
        """Test pattern analysis with complete data (no missing values)."""
        complete_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        patterns = analyze_patterns(complete_data)
        
        # Should have exactly 1 pattern (complete cases)
        assert len(patterns) == 1
        
        pattern = patterns[0]
        assert pattern.n_cases == 3
        assert pattern.n_observed == 3
        assert pattern.n_missing == 0
        assert len(pattern.missing_indices) == 0
        np.testing.assert_array_equal(pattern.observed_indices, [0, 1, 2])
        
    def test_single_pattern_edge_case(self):
        """Test pattern analysis with single missingness pattern."""
        single_pattern_data = np.array([
            [1.0, np.nan, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, np.nan, 9.0]
        ])
        
        patterns = analyze_patterns(single_pattern_data)
        
        # Should have exactly 1 pattern
        assert len(patterns) == 1
        
        pattern = patterns[0]
        assert pattern.n_cases == 3
        assert pattern.n_observed == 2  # Variables 0 and 2
        assert pattern.n_missing == 1   # Variable 1
        np.testing.assert_array_equal(pattern.observed_indices, [0, 2])
        np.testing.assert_array_equal(pattern.missing_indices, [1])
        
    def test_integration_with_mcar_test(self):
        """Test that pattern analysis integrates properly with MCAR test."""
        # Run MCAR test on simple data
        mcar_result = little_mcar_test(self.simple_data)
        
        # MCAR result should contain pattern information
        assert hasattr(mcar_result, 'patterns')
        assert len(mcar_result.patterns) == 4
        
        # Pattern information should match standalone analysis
        standalone_patterns = analyze_patterns(self.simple_data)
        
        assert len(mcar_result.patterns) == len(standalone_patterns)
        
        for mcar_pattern, standalone_pattern in zip(mcar_result.patterns, standalone_patterns):
            assert mcar_pattern.n_cases == standalone_pattern.n_cases
            assert mcar_pattern.n_observed == standalone_pattern.n_observed
            np.testing.assert_array_equal(mcar_pattern.observed_indices, standalone_pattern.observed_indices)
            
    def test_describe_patterns_function(self):
        """Test pattern description functionality."""
        patterns = analyze_patterns(self.simple_data)
        
        # Test with default variable names
        description = describe_patterns(patterns)
        assert isinstance(description, str)
        assert "Pattern 1:" in description
        assert "Var_0" in description  # Default variable naming
        
        # Test with custom variable names
        var_names = ['Height', 'Weight', 'Age']
        description_custom = describe_patterns(patterns, var_names)
        assert "Height" in description_custom
        assert "Weight" in description_custom
        assert "Age" in description_custom
        assert "Var_0" not in description_custom  # Should not have default names
        
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test 1D array rejection
        with pytest.raises(ValueError, match="2-dimensional"):
            analyze_patterns(np.array([1, 2, 3]))
            
        # Test empty data
        with pytest.raises(ValueError, match="at least one observation"):
            analyze_patterns(np.array([]).reshape(0, 3))
            
        # Test no variables
        with pytest.raises(ValueError, match="at least one variable"):
            analyze_patterns(np.array([]).reshape(3, 0))
            
    def test_pattern_summary_validation(self):
        """Test pattern summary validation."""
        # Test empty pattern list
        with pytest.raises(ValueError, match="No patterns provided"):
            pattern_summary([])
            
        # Test with valid patterns
        patterns = analyze_patterns(self.simple_data)
        
        # Should work without data_shape
        summary_no_shape = pattern_summary(patterns)
        assert isinstance(summary_no_shape, PatternSummary)
        assert np.isnan(summary_no_shape.overall_missing_rate)
        assert len(summary_no_shape.variable_missing_rates) == 0
        
        # Should work with data_shape
        summary_with_shape = pattern_summary(patterns, self.simple_data.shape)
        assert not np.isnan(summary_with_shape.overall_missing_rate)
        assert len(summary_with_shape.variable_missing_rates) == 3
        
    def test_backward_compatibility(self):
        """Test that existing functionality still works after refactoring."""
        # Test that mlest still works (should be unaffected)
        from pymvnmle import mlest
        
        result = mlest(datasets.apple, verbose=False)
        assert result.converged
        assert hasattr(result, 'muhat')
        assert hasattr(result, 'sigmahat')
        assert hasattr(result, 'loglik')
        
        # Test that MCAR test still works
        mcar_result = little_mcar_test(datasets.apple)
        assert hasattr(mcar_result, 'statistic')
        assert hasattr(mcar_result, 'p_value')
        assert hasattr(mcar_result, 'patterns')
        
    def test_pattern_percentage_calculation(self):
        """Test that pattern percentages are calculated correctly."""
        patterns = analyze_patterns(self.simple_data)
        
        # Sum of all percentages should equal 100%
        total_percentage = sum(p.percent_cases for p in patterns)
        assert abs(total_percentage - 100.0) < 0.01
        
        # Each percentage should be reasonable
        for pattern in patterns:
            expected_percentage = (pattern.n_cases / 6) * 100
            assert abs(pattern.percent_cases - expected_percentage) < 0.01
            
    def test_pattern_data_integrity(self):
        """Test that pattern data extraction is correct."""
        patterns = analyze_patterns(self.simple_data)
        
        for pattern in patterns:
            # Pattern data should only contain observed variables
            assert pattern.data.shape[1] == len(pattern.observed_indices)
            
            # Pattern data should not contain any NaN values
            assert not np.any(np.isnan(pattern.data))
            
            # Pattern data should contain the correct number of cases
            assert pattern.data.shape[0] == pattern.n_cases


def test_pattern_analysis_main_functions():
    """Test the main pattern analysis functions work correctly."""
    # Test data
    test_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [np.nan, 8.0, 9.0]
    ])
    
    # Test analyze_patterns
    patterns = analyze_patterns(test_data)
    assert len(patterns) == 3  # Three different patterns
    
    # Test pattern_summary
    summary = pattern_summary(patterns, test_data.shape)
    assert summary.n_patterns == 3
    assert summary.total_cases == 3
    
    # Test describe_patterns
    description = describe_patterns(patterns)
    assert isinstance(description, str)
    assert "Pattern 1:" in description


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])