"""
Pattern optimization utilities for efficient likelihood computation.

This module provides specialized data structures and functions for grouping
observations by missing data pattern to enable vectorized likelihood computation.
These utilities are used internally by the objective functions to achieve
significant performance improvements, especially for datasets with many
observations but few unique missing patterns.

The optimization works by:
1. Identifying unique missing data patterns
2. Grouping observations with identical patterns
3. Computing expensive operations (matrix inversions, etc.) once per pattern
4. Vectorizing computations within each pattern group

This is intentionally separate from patterns.py which focuses on statistical
analysis and MCAR testing. The two modules serve different purposes and use
different data structures optimized for their specific use cases.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class OptimizedPatternGroup:
    """
    Container for pattern group data optimized for likelihood computation.
    
    This class holds all observations sharing the same missing pattern,
    with data pre-extracted and organized for efficient vectorized operations.
    
    Attributes
    ----------
    pattern_id : int
        Unique identifier for this pattern
    observed_mask : np.ndarray
        Boolean mask of observed variables (shape: n_vars)
    observed_indices : np.ndarray
        Indices of observed variables
    missing_indices : np.ndarray
        Indices of missing variables  
    row_indices : np.ndarray
        Row indices in original data with this pattern
    n_obs : int
        Number of observations with this pattern
    observed_data : np.ndarray
        Observed data values for this pattern (n_obs × n_observed)
        Pre-extracted for efficiency
    """
    pattern_id: int
    observed_mask: np.ndarray
    observed_indices: np.ndarray
    missing_indices: np.ndarray
    row_indices: np.ndarray
    n_obs: int
    observed_data: np.ndarray
    
    @property
    def n_observed(self) -> int:
        """Number of observed variables in this pattern."""
        return len(self.observed_indices)
    
    @property
    def n_missing(self) -> int:
        """Number of missing variables in this pattern."""
        return len(self.missing_indices)
    
    @property
    def is_complete(self) -> bool:
        """True if this pattern has no missing values."""
        return self.n_missing == 0


def prepare_pattern_groups(
    data: np.ndarray, 
    sort_by_size: bool = True
) -> List[OptimizedPatternGroup]:
    """
    Prepare pattern groups optimized for likelihood computation.
    
    Groups observations by their missing data pattern and pre-extracts
    observed values for efficient vectorized computation.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix with missing values as np.nan (n_obs × n_vars)
    sort_by_size : bool, default=True
        If True, sort patterns by number of observations (largest first).
        This improves cache efficiency and vectorization performance.
        
    Returns
    -------
    List[OptimizedPatternGroup]
        List of pattern groups ready for vectorized likelihood computation.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, np.nan], [3, 4, 5], [6, 7, np.nan]])
    >>> groups = prepare_pattern_groups(data)
    >>> print(f"Found {len(groups)} unique patterns")
    Found 2 unique patterns
    >>> # Process complete cases efficiently
    >>> complete_groups = [g for g in groups if g.is_complete]
    """
    n_obs, n_vars = data.shape
    
    # Create observed mask (True = observed, False = missing)
    observed_mask = ~np.isnan(data)
    
    # Find unique patterns and map observations to patterns
    unique_patterns, inverse = np.unique(observed_mask, axis=0, return_inverse=True)
    n_patterns = len(unique_patterns)
    
    # Build optimized pattern groups
    pattern_groups = []
    
    for pattern_id in range(n_patterns):
        # Get the pattern mask
        pattern_mask = unique_patterns[pattern_id]
        
        # Find all rows with this pattern
        row_indices = np.where(inverse == pattern_id)[0]
        n_obs_pattern = len(row_indices)
        
        # Get observed and missing variable indices
        observed_indices = np.where(pattern_mask)[0]
        missing_indices = np.where(~pattern_mask)[0]
        
        # Extract observed data for this pattern
        # This is the key optimization - pre-extract data by pattern
        if len(observed_indices) > 0:
            observed_data = data[np.ix_(row_indices, observed_indices)]
        else:
            # All variables missing (edge case)
            observed_data = np.empty((n_obs_pattern, 0))
        
        # Create pattern group
        group = OptimizedPatternGroup(
            pattern_id=pattern_id,
            observed_mask=pattern_mask,
            observed_indices=observed_indices,
            missing_indices=missing_indices,
            row_indices=row_indices,
            n_obs=n_obs_pattern,
            observed_data=observed_data
        )
        
        pattern_groups.append(group)
    
    # Sort by number of observations if requested
    # Largest groups first improves vectorization efficiency
    if sort_by_size:
        pattern_groups.sort(key=lambda g: g.n_obs, reverse=True)
        # Update pattern IDs after sorting
        for i, group in enumerate(pattern_groups):
            group.pattern_id = i
    
    return pattern_groups


def compute_efficiency_metrics(
    pattern_groups: List[OptimizedPatternGroup],
    n_total_obs: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute metrics about pattern optimization efficiency.
    
    These metrics help determine whether pattern optimization will provide
    significant performance benefits for a given dataset.
    
    Parameters
    ----------
    pattern_groups : List[OptimizedPatternGroup]
        Pattern groups from prepare_pattern_groups
    n_total_obs : Optional[int]
        Total observations (computed from groups if not provided)
        
    Returns
    -------
    Dict[str, float]
        Efficiency metrics including:
        - n_patterns: number of unique patterns
        - compression_ratio: n_patterns / n_observations (lower is better)
        - avg_pattern_size: average observations per pattern
        - expected_speedup: estimated performance improvement factor
        
    Examples
    --------
    >>> metrics = compute_efficiency_metrics(groups)
    >>> if metrics['expected_speedup'] > 2.0:
    ...     print("Pattern optimization recommended")
    """
    n_patterns = len(pattern_groups)
    
    if n_total_obs is None:
        n_total_obs = sum(g.n_obs for g in pattern_groups)
    
    if n_total_obs == 0:
        return {
            'n_patterns': 0,
            'n_total_obs': 0,
            'compression_ratio': 1.0,
            'avg_pattern_size': 0.0,
            'max_pattern_size': 0,
            'min_pattern_size': 0,
            'expected_speedup': 1.0
        }
    
    pattern_sizes = [g.n_obs for g in pattern_groups]
    
    # Compression ratio: how much we reduce the problem size
    compression_ratio = n_patterns / n_total_obs
    
    # Average observations per pattern
    avg_pattern_size = np.mean(pattern_sizes)
    
    # Size extremes
    max_pattern_size = max(pattern_sizes)
    min_pattern_size = min(pattern_sizes)
    
    # Estimate expected speedup
    # Based on: fewer matrix operations + vectorization benefits
    base_speedup = 1.0 / compression_ratio
    
    # Vectorization bonus for large pattern groups
    if avg_pattern_size > 10:
        vectorization_bonus = 1.5
    elif avg_pattern_size > 5:
        vectorization_bonus = 1.2
    else:
        vectorization_bonus = 1.0
    
    expected_speedup = base_speedup * vectorization_bonus
    
    return {
        'n_patterns': n_patterns,
        'n_total_obs': n_total_obs,
        'compression_ratio': compression_ratio,
        'avg_pattern_size': avg_pattern_size,
        'max_pattern_size': max_pattern_size,
        'min_pattern_size': min_pattern_size,
        'expected_speedup': expected_speedup
    }


def validate_pattern_groups(
    data: np.ndarray, 
    pattern_groups: List[OptimizedPatternGroup]
) -> bool:
    """
    Validate that pattern groups correctly represent the original data.
    
    This validation is critical for ensuring the optimization produces
    identical results to the naive implementation.
    
    Parameters
    ----------
    data : np.ndarray
        Original data matrix
    pattern_groups : List[OptimizedPatternGroup]
        Pattern groups to validate
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    AssertionError
        If validation fails with details about the failure
    """
    n_obs, n_vars = data.shape
    
    # Check 1: All observations are accounted for
    all_row_indices = np.concatenate([g.row_indices for g in pattern_groups])
    assert len(all_row_indices) == n_obs, \
        f"Pattern groups contain {len(all_row_indices)} obs, expected {n_obs}"
    assert len(np.unique(all_row_indices)) == n_obs, \
        "Duplicate row indices found in pattern groups"
    assert set(all_row_indices) == set(range(n_obs)), \
        "Not all observations are covered by pattern groups"
    
    # Check 2: Validate each pattern group
    for group in pattern_groups:
        # Verify dimensions
        assert group.observed_data.shape[0] == group.n_obs, \
            f"Pattern {group.pattern_id}: data rows ({group.observed_data.shape[0]}) != n_obs ({group.n_obs})"
        assert group.observed_data.shape[1] == len(group.observed_indices), \
            f"Pattern {group.pattern_id}: data cols ({group.observed_data.shape[1]}) != observed indices ({len(group.observed_indices)})"
        
        # Verify data matches original
        for i, row_idx in enumerate(group.row_indices):
            if len(group.observed_indices) > 0:
                original_obs = data[row_idx, group.observed_indices]
                pattern_obs = group.observed_data[i]
                
                # Check values match (handle NaN properly)
                assert np.allclose(original_obs, pattern_obs, equal_nan=True), \
                    f"Pattern {group.pattern_id}, row {i}: data mismatch"
            
            # Verify missing values are actually missing
            if len(group.missing_indices) > 0:
                original_missing = data[row_idx, group.missing_indices]
                assert np.all(np.isnan(original_missing)), \
                    f"Pattern {group.pattern_id}, row {i}: expected missing values not NaN"
        
        # Verify indices consistency
        assert len(group.observed_indices) + len(group.missing_indices) == n_vars, \
            f"Pattern {group.pattern_id}: observed + missing != n_vars"
        
        # Verify no index appears in both observed and missing
        assert len(set(group.observed_indices) & set(group.missing_indices)) == 0, \
            f"Pattern {group.pattern_id}: indices appear in both observed and missing"
        
        # Verify mask consistency
        assert np.all(group.observed_mask[group.observed_indices]), \
            f"Pattern {group.pattern_id}: observed mask False for observed indices"
        assert np.all(~group.observed_mask[group.missing_indices]), \
            f"Pattern {group.pattern_id}: observed mask True for missing indices"
    
    return True


def should_use_pattern_optimization(
    data: np.ndarray,
    min_speedup: float = 1.5
) -> bool:
    """
    Determine if pattern optimization should be used for given data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix with missing values
    min_speedup : float, default=1.5
        Minimum expected speedup to use optimization
        
    Returns
    -------
    bool
        True if pattern optimization is recommended
        
    Notes
    -----
    Pattern optimization has overhead, so it's only beneficial when:
    - The compression ratio is good (few patterns relative to observations)
    - Pattern groups are large enough for vectorization benefits
    """
    # Quick check: if no missing data, no benefit
    if not np.any(np.isnan(data)):
        return False
    
    # Quick check: very small datasets don't benefit
    n_obs = data.shape[0]
    if n_obs < 50:
        return False
    
    # Prepare patterns and check efficiency
    groups = prepare_pattern_groups(data)
    metrics = compute_efficiency_metrics(groups)
    
    return metrics['expected_speedup'] >= min_speedup


# Optional: Helper for debugging and analysis
def print_pattern_summary(pattern_groups: List[OptimizedPatternGroup]) -> None:
    """
    Print a summary of pattern groups for debugging.
    
    Parameters
    ----------
    pattern_groups : List[OptimizedPatternGroup]
        Pattern groups to summarize
    """
    metrics = compute_efficiency_metrics(pattern_groups)
    
    print("\nPattern Optimization Summary")
    print("=" * 60)
    print(f"Unique patterns: {metrics['n_patterns']}")
    print(f"Total observations: {metrics['n_total_obs']}")
    print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
    print(f"Average pattern size: {metrics['avg_pattern_size']:.1f}")
    print(f"Expected speedup: {metrics['expected_speedup']:.1f}x")
    
    print("\nPattern Details (top 5):")
    for group in pattern_groups[:5]:
        pct = group.n_obs / metrics['n_total_obs'] * 100
        print(f"  Pattern {group.pattern_id}: {group.n_obs:4d} obs ({pct:5.1f}%) "
              f"- {group.n_observed} observed, {group.n_missing} missing")
    
    if len(pattern_groups) > 5:
        print(f"  ... and {len(pattern_groups) - 5} more patterns")