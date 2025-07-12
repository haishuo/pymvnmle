"""Test Little's MCAR implementation against R references."""

import pytest
import numpy as np
import json
from pathlib import Path

from pymvnmle import datasets
from pymvnmle.mcar_test import little_mcar_test  # Correct import

class TestLittleMCAR:
    """Test suite for Little's MCAR test."""
    
    @pytest.fixture
    def load_reference(self):
        """Load R reference results."""
        def _load(name):
            ref_path = Path('tests/references') / f'little_mcar_{name}.json'
            with open(ref_path) as f:
                return json.load(f)
        return _load
    
    def test_apple_dataset(self, load_reference):
        """Test against R reference for apple dataset."""
        # Run our test
        result = little_mcar_test(datasets.apple, verbose=True)
        
        # Load R reference
        ref = load_reference('apple')
        
        # Validate core results
        assert abs(result.statistic - ref['test_statistic']) < 0.01
        assert abs(result.p_value - ref['p_value']) < 0.001
        assert result.df == ref['df']
        assert result.rejected == True  # Should reject MCAR
        
        print(result.summary())
    
    def test_simple_mcar_dataset(self, load_reference):
        """Test dataset that should NOT reject MCAR."""
        # Need to recreate the simple MCAR data
        np.random.seed(42)
        data = np.random.randn(100, 3)
        missing_indices = np.random.choice(300, 30, replace=False)
        data.flat[missing_indices] = np.nan
        
        result = little_mcar_test(data)
        
        # Should NOT reject MCAR (p > 0.05)
        assert result.p_value > 0.05
        assert result.rejected == False
    
    def test_complete_data(self):
        """Test edge case with no missing data."""
        data = np.random.randn(50, 3)
        result = little_mcar_test(data)
        
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.df == 0
        assert result.n_patterns == 1
        assert result.n_patterns_used == 0
    
    def test_missvals_robustness(self):
        """Test that we handle missvals better than R."""
        # This is where R fails - we should succeed
        result = little_mcar_test(datasets.missvals, verbose=True)
        
        # We should get a result even if R couldn't
        assert np.isfinite(result.statistic) or len(result.numerical_issues) > 0
        
        # If we got a result, it should be reasonable
        if np.isfinite(result.statistic):
            assert result.statistic > 0
            assert 0 <= result.p_value <= 1
            assert result.df > 0
        
        print(f"Handled missvals: {result.summary()}")