"""
Pattern analysis utilities for PyMVNMLE
=======================================

This module provides tools for analyzing missingness patterns in multivariate data.
Originally part of the MCAR test implementation, these utilities are now exposed
as standalone functionality for comprehensive missing data analysis.

Key Functions:
- analyze_patterns(): Identify all unique missingness patterns
- pattern_summary(): Generate summary statistics for patterns
- identify_missingness_patterns(): Core pattern identification algorithm

Author: PyMVNMLE Development Team
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PatternInfo:
    """
    Information about a single missingness pattern.
    
    This class contains all relevant information about a unique pattern of
    missing values in the dataset, including which variables are observed,
    how many cases follow this pattern, and the actual data for those cases.
    
    Attributes
    ----------
    pattern_id : int
        Unique identifier for this pattern (1-indexed for readability)
    observed_indices : np.ndarray
        Array of indices indicating which variables are observed in this pattern
    missing_indices : np.ndarray
        Array of indices indicating which variables are missing in this pattern
    n_cases : int
        Number of observations that follow this missingness pattern
    data : np.ndarray
        Data matrix containing only the observed variables for cases with this pattern
    pattern_vector : np.ndarray
        Binary vector where 1=observed, 0=missing for each variable
    """
    pattern_id: int
    observed_indices: np.ndarray
    missing_indices: np.ndarray
    n_cases: int
    data: np.ndarray
    pattern_vector: np.ndarray
    
    @property
    def n_observed(self) -> int:
        """Number of observed variables in this pattern."""
        return len(self.observed_indices)
    
    @property
    def n_missing(self) -> int:
        """Number of missing variables in this pattern."""
        return len(self.missing_indices)
    
    @property
    def percent_cases(self) -> float:
        """Percentage of total cases with this pattern (set externally)."""
        return getattr(self, '_percent_cases', 0.0)
    
    @percent_cases.setter
    def percent_cases(self, value: float):
        """Set percentage of cases with this pattern."""
        self._percent_cases = value
    
    def __repr__(self) -> str:
        """String representation of pattern info."""
        return (f"PatternInfo(id={self.pattern_id}, n_cases={self.n_cases}, "
                f"n_observed={self.n_observed}, n_missing={self.n_missing})")


@dataclass
class PatternSummary:
    """
    Summary statistics for all missingness patterns in a dataset.
    
    This class provides aggregate information about the missing data structure,
    including overall missingness rates, pattern frequencies, and variable-specific
    missing rates.
    
    Attributes
    ----------
    n_patterns : int
        Total number of unique missingness patterns
    total_cases : int
        Total number of observations in the dataset
    overall_missing_rate : float
        Overall proportion of missing values (0.0 to 1.0)
    most_common_pattern : PatternInfo
        The pattern with the highest frequency of observations
    complete_cases : int
        Number of observations with no missing values
    complete_cases_percent : float
        Percentage of observations with no missing values
    variable_missing_rates : Dict[int, float]
        Missing rate for each variable (variable index -> missing rate)
    """
    n_patterns: int
    total_cases: int
    overall_missing_rate: float
    most_common_pattern: PatternInfo
    complete_cases: int
    complete_cases_percent: float
    variable_missing_rates: Dict[int, float]
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Missingness Pattern Summary",
            f"=" * 40,
            f"Total patterns: {self.n_patterns}",
            f"Total cases: {self.total_cases}",
            f"Overall missing rate: {self.overall_missing_rate:.1%}",
            f"Complete cases: {self.complete_cases} ({self.complete_cases_percent:.1%})",
            f"Most common pattern: {self.most_common_pattern.n_cases} cases "
            f"({self.most_common_pattern.percent_cases:.1%})"
        ]
        return "\n".join(lines)


def identify_missingness_patterns(data: np.ndarray) -> List[PatternInfo]:
    """
    Identify and extract all unique missingness patterns in the data.
    
    This function implements the core pattern identification algorithm that
    groups observations by their pattern of missing values. It uses the same
    approach as R's pattern identification but with improved efficiency.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix with missing values as np.nan
        Shape: (n_observations, n_variables)
        
    Returns
    -------
    List[PatternInfo]
        List of PatternInfo objects, one for each unique pattern.
        Patterns are sorted by frequency (most common first).
        
    Notes
    -----
    The algorithm works by:
    1. Creating binary pattern matrix (1=observed, 0=missing)
    2. Converting patterns to unique decimal identifiers
    3. Grouping observations by identical patterns
    4. Extracting data subsets for each pattern
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0, 3.0],
    ...                  [4.0, np.nan, 6.0],
    ...                  [7.0, 8.0, np.nan],
    ...                  [10.0, 11.0, 12.0]])
    >>> patterns = identify_missingness_patterns(data)
    >>> len(patterns)  # Number of unique patterns
    3
    >>> patterns[0].n_observed  # Most common pattern
    3
    """
    n_obs, n_vars = data.shape
    
    # Create binary pattern matrix (1 = observed, 0 = missing)
    pattern_matrix = (~np.isnan(data)).astype(int)
    
    # Convert patterns to unique identifiers using powers of 2
    # This creates a unique decimal representation for each pattern
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_ids = pattern_matrix @ powers
    
    # Find unique patterns and their frequencies
    unique_patterns, inverse_indices = np.unique(pattern_ids, return_inverse=True)
    
    # Build PatternInfo objects for each unique pattern
    patterns = []
    for i, pattern_id in enumerate(unique_patterns):
        # Find cases with this pattern
        case_mask = (pattern_ids == pattern_id)
        pattern_data = data[case_mask]
        
        # Get the actual pattern vector (first occurrence)
        pattern_idx = np.where(pattern_ids == pattern_id)[0][0]
        pattern_vector = pattern_matrix[pattern_idx]
        
        # Identify observed and missing variables
        observed_indices = np.where(pattern_vector == 1)[0]
        missing_indices = np.where(pattern_vector == 0)[0]
        
        # Extract only observed columns for this pattern
        pattern_data_observed = pattern_data[:, observed_indices]
        
        # Create PatternInfo object
        pattern_info = PatternInfo(
            pattern_id=i + 1,  # 1-indexed for user friendliness
            observed_indices=observed_indices,
            missing_indices=missing_indices,
            n_cases=int(np.sum(case_mask)),
            data=pattern_data_observed,
            pattern_vector=pattern_vector
        )
        
        patterns.append(pattern_info)
    
    # Sort by frequency (most common first)
    patterns.sort(key=lambda p: p.n_cases, reverse=True)
    
    # Update pattern IDs to reflect sorted order
    for i, pattern in enumerate(patterns):
        pattern.pattern_id = i + 1
    
    # Calculate percentage information
    total_cases = n_obs
    for pattern in patterns:
        pattern.percent_cases = (pattern.n_cases / total_cases) * 100
    
    return patterns


def analyze_patterns(data: Union[np.ndarray, pd.DataFrame]) -> List[PatternInfo]:
    """
    Analyze missingness patterns in the data.
    
    This is the main public interface for pattern analysis. It handles data
    validation and conversion before calling the core pattern identification
    algorithm.
    
    Parameters
    ----------
    data : array-like
        Data matrix with missing values as np.nan.
        Can be NumPy array or pandas DataFrame.
        
    Returns
    -------
    List[PatternInfo]
        List of PatternInfo objects describing each unique missingness pattern.
        Patterns are sorted by frequency (most common first).
        
    Raises
    ------
    ValueError
        If data is not 2-dimensional or contains invalid values
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pymvnmle import analyze_patterns
    >>> 
    >>> # With NumPy array
    >>> data = np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 4.0]])
    >>> patterns = analyze_patterns(data)
    >>> print(f"Found {len(patterns)} patterns")
    >>> 
    >>> # With pandas DataFrame
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    >>> patterns = analyze_patterns(df)
    >>> for pattern in patterns:
    ...     print(f"Pattern {pattern.pattern_id}: {pattern.n_cases} cases")
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        data_array = data.values.astype(float)
    else:
        data_array = np.asarray(data, dtype=float)
    
    # Basic validation
    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    
    if data_array.shape[0] < 1:
        raise ValueError("Data must have at least one observation")
    
    if data_array.shape[1] < 1:
        raise ValueError("Data must have at least one variable")
    
    # Call core pattern identification
    return identify_missingness_patterns(data_array)


def pattern_summary(patterns: List[PatternInfo], 
                   data_shape: Optional[Tuple[int, int]] = None) -> PatternSummary:
    """
    Generate summary statistics for missingness patterns.
    
    This function computes aggregate statistics about the missing data structure,
    including overall missingness rates, pattern frequencies, and variable-specific
    missing rates.
    
    Parameters
    ----------
    patterns : List[PatternInfo]
        List of pattern information objects from analyze_patterns()
    data_shape : Optional[Tuple[int, int]]
        Original data shape (n_observations, n_variables) if available.
        If provided, enables calculation of overall missing rate and
        variable-specific missing rates.
        
    Returns
    -------
    PatternSummary
        Summary statistics for all patterns
        
    Examples
    --------
    >>> patterns = analyze_patterns(data)
    >>> summary = pattern_summary(patterns, data.shape)
    >>> print(summary)
    >>> print(f"Variable 0 missing rate: {summary.variable_missing_rates[0]:.1%}")
    """
    if not patterns:
        raise ValueError("No patterns provided")
    
    n_patterns = len(patterns)
    total_cases = sum(p.n_cases for p in patterns)
    
    # Find complete cases pattern (no missing variables)
    complete_pattern = None
    for pattern in patterns:
        if len(pattern.missing_indices) == 0:
            complete_pattern = pattern
            break
    
    complete_cases = complete_pattern.n_cases if complete_pattern else 0
    complete_cases_percent = (complete_cases / total_cases * 100) if total_cases > 0 else 0
    
    # Most common pattern
    most_common = max(patterns, key=lambda p: p.n_cases)
    
    # Overall missing rate calculation
    if data_shape:
        n_obs, n_vars = data_shape
        total_possible_values = n_obs * n_vars
        
        # Calculate total observed values across all patterns
        total_observed_values = sum(p.n_cases * p.n_observed for p in patterns)
        overall_missing_rate = 1 - (total_observed_values / total_possible_values)
        
        # Variable-specific missing rates
        variable_missing_rates = {}
        for var_idx in range(n_vars):
            var_observed_cases = sum(p.n_cases for p in patterns 
                                   if var_idx in p.observed_indices)
            variable_missing_rates[var_idx] = 1 - (var_observed_cases / total_cases)
    else:
        overall_missing_rate = np.nan
        variable_missing_rates = {}
    
    return PatternSummary(
        n_patterns=n_patterns,
        total_cases=total_cases,
        overall_missing_rate=overall_missing_rate,
        most_common_pattern=most_common,
        complete_cases=complete_cases,
        complete_cases_percent=complete_cases_percent,
        variable_missing_rates=variable_missing_rates
    )


def describe_patterns(patterns: List[PatternInfo], 
                     variable_names: Optional[List[str]] = None) -> str:
    """
    Generate a detailed description of missingness patterns.
    
    This function creates a human-readable report of all missingness patterns,
    including which variables are observed/missing in each pattern and the
    frequency of each pattern.
    
    Parameters
    ----------
    patterns : List[PatternInfo]
        List of pattern information objects
    variable_names : Optional[List[str]]
        Names of variables for more readable output. If None, uses indices.
        
    Returns
    -------
    str
        Detailed description of all patterns
        
    Examples
    --------
    >>> patterns = analyze_patterns(data)
    >>> description = describe_patterns(patterns, ['Age', 'Weight', 'Height'])
    >>> print(description)
    """
    if not patterns:
        return "No patterns found."
    
    lines = ["Missingness Pattern Details", "=" * 40]
    
    for pattern in patterns:
        lines.append(f"\nPattern {pattern.pattern_id}: {pattern.n_cases} cases ({pattern.percent_cases:.1f}%)")
        
        # Format variable names
        if variable_names:
            observed_vars = [variable_names[i] for i in pattern.observed_indices]
            missing_vars = [variable_names[i] for i in pattern.missing_indices]
        else:
            observed_vars = [f"Var_{i}" for i in pattern.observed_indices]
            missing_vars = [f"Var_{i}" for i in pattern.missing_indices]
        
        if observed_vars:
            lines.append(f"  Observed: {', '.join(observed_vars)}")
        if missing_vars:
            lines.append(f"  Missing:  {', '.join(missing_vars)}")
        else:
            lines.append(f"  Missing:  (none - complete cases)")
    
    return "\n".join(lines)