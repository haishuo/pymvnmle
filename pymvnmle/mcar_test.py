"""
Little's MCAR Test for PyMVNMLE v1.5
====================================

Implementation of Little's (1988) test for Missing Completely at Random (MCAR).

This module provides a statistical test to assess whether missing data patterns
are related to the observed values, which would violate the MCAR assumption.

Reference:
    Little, R.J.A. (1988). A test of missing completely at random for 
    multivariate data with missing values. Journal of the American 
    Statistical Association, 83(404), 1198-1202.

Author: PyMVNMLE Development Team
Date: January 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

# Import from other PyMVNMLE modules
from .mlest import mlest


@dataclass
class PatternInfo:
    """Information about a single missingness pattern."""
    pattern_id: int
    observed_indices: np.ndarray  # Which variables are observed
    missing_indices: np.ndarray   # Which variables are missing
    n_cases: int                  # Number of cases with this pattern
    data: np.ndarray             # Data for cases with this pattern
    pattern_vector: np.ndarray   # Binary pattern (1=observed, 0=missing)
    
    @property
    def n_observed(self) -> int:
        """Number of observed variables in this pattern."""
        return len(self.observed_indices)
    
    @property
    def percent_cases(self) -> float:
        """Percentage of total cases with this pattern."""
        # Will be set by the parent function
        return getattr(self, '_percent_cases', 0.0)


@dataclass 
class MCARTestResult:
    """
    Result of Little's MCAR test.
    
    Attributes
    ----------
    statistic : float
        The chi-square test statistic
    df : int
        Degrees of freedom
    p_value : float
        P-value of the test
    rejected : bool
        Whether MCAR hypothesis is rejected at given alpha
    alpha : float
        Significance level used
    patterns : List[PatternInfo]
        Information about each missingness pattern
    n_patterns : int
        Number of unique missingness patterns
    n_patterns_used : int
        Number of patterns actually used in test (excluding patterns with issues)
    ml_mean : np.ndarray
        ML estimate of mean vector
    ml_cov : np.ndarray
        ML estimate of covariance matrix
    convergence_warnings : List[str]
        Any numerical warnings during computation
    """
    statistic: float
    df: int
    p_value: float
    rejected: bool
    alpha: float
    patterns: List[PatternInfo]
    n_patterns: int
    n_patterns_used: int
    ml_mean: np.ndarray
    ml_cov: np.ndarray
    convergence_warnings: List[str]
    
    def summary(self) -> str:
        """Generate human-readable summary of test results."""
        summary_lines = [
            "Little's MCAR Test Results",
            "=" * 40,
            f"Test statistic (χ²): {self.statistic:.4f}",
            f"Degrees of freedom: {self.df}",
            f"P-value: {self.p_value:.4f}",
            f"",
            f"Decision at α={self.alpha}: {'Reject MCAR' if self.rejected else 'Fail to reject MCAR'}",
            f"",
            f"Number of patterns: {self.n_patterns}",
        ]
        
        # Add pattern summary
        summary_lines.append("\nMissingness Patterns:")
        summary_lines.append("-" * 40)
        for pattern in sorted(self.patterns, key=lambda p: p.n_cases, reverse=True):
            pct = (pattern.n_cases / sum(p.n_cases for p in self.patterns)) * 100
            summary_lines.append(
                f"Pattern {pattern.pattern_id}: {pattern.n_cases} cases ({pct:.1f}%), "
                f"{pattern.n_observed} variables observed"
            )
        
        # Add warnings if any
        if self.convergence_warnings:
            summary_lines.append("\nWarnings:")
            for warning in self.convergence_warnings:
                summary_lines.append(f"  - {warning}")
        
        # Add interpretation
        summary_lines.extend([
            "",
            "Interpretation:",
            "-" * 40
        ])
        
        # Special case: no missing data
        if self.df == 0 and len(self.patterns) == 1:
            summary_lines.extend([
                "No missing data detected.",
                "MCAR test is not applicable when all data are complete.",
                "All analyses can proceed without missing data considerations."
            ])
        elif self.rejected:
            summary_lines.extend([
                "The null hypothesis of MCAR is REJECTED.",
                "Evidence suggests the missing data mechanism depends on the observed values.",
                "Standard methods assuming MCAR (listwise deletion, etc.) may be biased.",
                "Consider methods that handle MAR or MNAR data."
            ])
        else:
            summary_lines.extend([
                "The null hypothesis of MCAR is NOT REJECTED.", 
                "No evidence that missingness depends on the observed values.",
                "However, this does not prove MCAR - the test may lack power.",
                "Missing data could still depend on unobserved values (MNAR)."
            ])
        
        return "\n".join(summary_lines)


def identify_missingness_patterns(data: np.ndarray) -> List[PatternInfo]:
    """
    Identify and extract all unique missingness patterns in the data.
    
    This implements the pattern identification similar to R's approach,
    grouping observations by their pattern of missing values.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix with missing values as np.nan
        
    Returns
    -------
    List[PatternInfo]
        Information about each unique pattern
    """
    n_obs, n_vars = data.shape
    
    # Create binary pattern matrix (1 = observed, 0 = missing)
    pattern_matrix = (~np.isnan(data)).astype(int)
    
    # Convert patterns to unique identifiers (like R's approach)
    # Use powers of 2 to create unique decimal representation
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_ids = pattern_matrix @ powers
    
    # Find unique patterns
    unique_patterns, inverse_indices = np.unique(pattern_ids, return_inverse=True)
    
    # Build PatternInfo for each unique pattern
    patterns = []
    for i, pattern_id in enumerate(unique_patterns):
        # Find cases with this pattern
        case_mask = (pattern_ids == pattern_id)
        pattern_data = data[case_mask]
        
        # Get the actual pattern vector
        pattern_idx = np.where(pattern_ids == pattern_id)[0][0]
        pattern_vector = pattern_matrix[pattern_idx]
        
        # Indices of observed/missing variables
        observed_indices = np.where(pattern_vector == 1)[0]
        missing_indices = np.where(pattern_vector == 0)[0]
        
        # Extract only observed columns for this pattern
        pattern_data_obs = pattern_data[:, observed_indices]
        
        pattern_info = PatternInfo(
            pattern_id=i + 1,  # 1-indexed for readability
            observed_indices=observed_indices,
            missing_indices=missing_indices,
            n_cases=int(np.sum(case_mask)),
            data=pattern_data_obs,
            pattern_vector=pattern_vector
        )
        
        patterns.append(pattern_info)
    
    # Sort by number of cases (descending)
    patterns.sort(key=lambda p: p.n_cases, reverse=True)
    
    # Add percentage information
    total_cases = n_obs
    for pattern in patterns:
        pattern._percent_cases = (pattern.n_cases / total_cases) * 100
    
    return patterns


def regularized_inverse(matrix: np.ndarray, 
                       condition_threshold: float = 1e12,
                       regularization: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """
    Compute inverse with regularization for near-singular matrices.
    
    This handles cases where the covariance matrix is near-singular,
    which causes R's implementation to fail (e.g., missvals dataset).
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to invert
    condition_threshold : float
        Maximum acceptable condition number
    regularization : float
        Regularization parameter to add to diagonal
        
    Returns
    -------
    inv_matrix : np.ndarray
        Inverted matrix
    was_regularized : bool
        Whether regularization was applied
    """
    # Check condition number
    cond = np.linalg.cond(matrix)
    
    if cond < condition_threshold:
        # Matrix is well-conditioned
        try:
            return np.linalg.inv(matrix), False
        except np.linalg.LinAlgError:
            pass
    
    # Need regularization
    n = matrix.shape[0]
    reg_matrix = matrix + regularization * np.eye(n)
    
    # Try with regularization
    try:
        return np.linalg.inv(reg_matrix), True
    except np.linalg.LinAlgError:
        # If still fails, use eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Regularize small eigenvalues
        min_eigenval = np.max(eigenvals) * regularization
        eigenvals_reg = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct inverse
        inv_matrix = eigenvecs @ np.diag(1/eigenvals_reg) @ eigenvecs.T
        
        return inv_matrix, True


def little_mcar_test(data: Union[np.ndarray, pd.DataFrame],
                     alpha: float = 0.05,
                     verbose: bool = False) -> MCARTestResult:
    """
    Little's test for Missing Completely at Random (MCAR).
    
    This implements Little (1988) JASA test which assesses whether the
    missingness mechanism depends on the observed data values. The test
    compares the means of observed variables across different patterns
    of missingness.
    
    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with missing values as np.nan.
        Can be NumPy array or pandas DataFrame.
    alpha : float, default=0.05
        Significance level for hypothesis test
    verbose : bool, default=False
        Whether to print detailed progress information
        
    Returns
    -------
    MCARTestResult
        Test results including statistic, p-value, and interpretation
        
    Notes
    -----
    The null hypothesis is that data are MCAR (missingness is independent
    of both observed and unobserved values). The alternative is that 
    missingness depends on the observed values.
    
    The test statistic is:
        d² = Σⱼ mⱼ(ȳ_obs,j - μ̂_obs,j)' Σ̂⁻¹_obs,j (ȳ_obs,j - μ̂_obs,j)
        
    which is asymptotically chi-squared with degrees of freedom:
        df = (sum of observed variables across patterns) - (total variables)
        
    References
    ----------
    Little, R.J.A. (1988). A test of missing completely at random for
    multivariate data with missing values. Journal of the American
    Statistical Association, 83(404), 1198-1202.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pymvnmle import little_mcar_test
    >>> 
    >>> # Create data with MCAR pattern
    >>> np.random.seed(42)
    >>> data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    >>> # Randomly delete some values
    >>> mask = np.random.random(data.shape) < 0.2
    >>> data[mask] = np.nan
    >>> 
    >>> result = little_mcar_test(data)
    >>> print(result.summary())
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        data_array = data.values.astype(float)
    else:
        data_array = np.asarray(data, dtype=float)
    
    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    
    n_obs, n_vars = data_array.shape
    
    if verbose:
        print(f"Little's MCAR Test")
        print(f"Data shape: {n_obs} observations × {n_vars} variables")
        missing_rate = np.sum(np.isnan(data_array)) / (n_obs * n_vars)
        print(f"Overall missing rate: {missing_rate:.1%}")
    
    # Check if we have any complete cases
    complete_cases = np.sum(~np.isnan(data_array).any(axis=1))
    if complete_cases < 2:
        raise ValueError("Need at least 2 complete cases for ML estimation")
    
    # Check each variable is observed at least once
    for j in range(n_vars):
        if np.all(np.isnan(data_array[:, j])):
            raise ValueError(f"Variable {j} is completely missing")
    
    # Step 1: Get ML estimates using pooled data
    if verbose:
        print("\nStep 1: Computing ML estimates...")
    
    try:
        ml_result = mlest(data_array, verbose=False)
        mu_ml = ml_result.muhat
        sigma_ml = ml_result.sigmahat
    except Exception as e:
        raise RuntimeError(f"ML estimation failed: {e}")
    
    if verbose:
        print(f"ML estimation converged: {ml_result.converged}")
        print(f"ML mean: {mu_ml}")
    
    # Step 2: Identify missingness patterns
    if verbose:
        print("\nStep 2: Identifying missingness patterns...")
    
    patterns = identify_missingness_patterns(data_array)
    
    if verbose:
        print(f"Found {len(patterns)} unique patterns")
        for p in patterns[:5]:  # Show first 5
            print(f"  Pattern {p.pattern_id}: {p.n_cases} cases, "
                  f"{p.n_observed} observed variables")
    
    # Step 3: Compute test statistic
    if verbose:
        print("\nStep 3: Computing test statistic...")
    
    test_statistic = 0.0
    total_observed_vars = 0
    convergence_warnings = []
    n_patterns_used = 0  # Track patterns actually used in computation
    
    for pattern in patterns:
        # Skip patterns with no observed variables (shouldn't happen after validation)
        if pattern.n_observed == 0:
            continue
        
        # Get indices of observed variables
        obs_idx = pattern.observed_indices
        n_k = pattern.n_cases
        
        # Compute pattern mean (only for observed variables)
        y_bar_k = np.mean(pattern.data, axis=0)
        
        # Extract ML estimates for observed variables
        mu_obs_k = mu_ml[obs_idx]
        sigma_obs_k = sigma_ml[np.ix_(obs_idx, obs_idx)]
        
        # Compute inverse of covariance submatrix
        try:
            sigma_inv_k, was_regularized = regularized_inverse(sigma_obs_k)
            
            if was_regularized:
                msg = f"Pattern {pattern.pattern_id}: Covariance regularized (near-singular)"
                convergence_warnings.append(msg)
                if verbose:
                    print(f"  Warning: {msg}")
        
        except Exception as e:
            msg = f"Pattern {pattern.pattern_id}: Failed to invert covariance: {e}"
            convergence_warnings.append(msg)
            if verbose:
                print(f"  Error: {msg}")
            continue
        
        # Compute contribution to test statistic
        diff = y_bar_k - mu_obs_k
        contribution = n_k * (diff @ sigma_inv_k @ diff)
        test_statistic += contribution
        n_patterns_used += 1  # Count this pattern as used
        
        # Track total observed variables for df calculation
        total_observed_vars += n_k * pattern.n_observed
        
        if verbose and pattern.n_cases > 5:  # Only show for larger patterns
            print(f"  Pattern {pattern.pattern_id}: contribution = {contribution:.4f}")
    
    # Step 4: Compute degrees of freedom
    # df = (sum of observed variables across patterns) - (total variables)
    # But we need to count by pattern, not by observation
    df = sum(p.n_observed for p in patterns) - n_vars
    
    # Handle edge cases
    if len(patterns) == 1 and patterns[0].n_observed == n_vars:
        # Complete data - no missing values, MCAR test not applicable
        return MCARTestResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejected=False,
            alpha=alpha,
            patterns=patterns,
            n_patterns=1,
            n_patterns_used=0,  # No patterns used since test not applicable
            ml_mean=mu_ml,
            ml_cov=sigma_ml,
            convergence_warnings=["No missing data - MCAR test not applicable"]
        )
    
    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom: {df}. "
                        "This may indicate insufficient variation in missingness patterns.")
    
    # Step 5: Compute p-value
    p_value = 1 - stats.chi2.cdf(test_statistic, df)
    
    # Determine if MCAR is rejected
    rejected = p_value < alpha
    
    if verbose:
        print(f"\nResults:")
        print(f"Test statistic: {test_statistic:.4f}")
        print(f"Degrees of freedom: {df}")
        print(f"P-value: {p_value:.4f}")
        print(f"Decision at α={alpha}: {'Reject MCAR' if rejected else 'Fail to reject MCAR'}")
    
    # Create result object
    result = MCARTestResult(
        statistic=test_statistic,
        df=df,
        p_value=p_value,
        rejected=rejected,
        alpha=alpha,
        patterns=patterns,
        n_patterns=len(patterns),
        n_patterns_used=n_patterns_used,
        ml_mean=mu_ml,
        ml_cov=sigma_ml,
        convergence_warnings=convergence_warnings
    )
    
    return result


def analyze_patterns(data: Union[np.ndarray, pd.DataFrame]) -> List[PatternInfo]:
    """
    Analyze missingness patterns in the data.
    
    This is a convenience function that extracts pattern information
    without running the full MCAR test.
    
    Parameters
    ----------
    data : array-like
        Data matrix with missing values
        
    Returns
    -------
    List[PatternInfo]
        Information about each missingness pattern
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values.astype(float)
    else:
        data_array = np.asarray(data, dtype=float)
    
    return identify_missingness_patterns(data_array)


@dataclass
class PatternSummary:
    """Summary statistics for missingness patterns."""
    n_patterns: int
    total_cases: int
    overall_missing_rate: float
    most_common_pattern: PatternInfo
    complete_cases: int
    complete_cases_percent: float
    variable_missing_rates: Dict[int, float]
    
    def __str__(self) -> str:
        """String representation of pattern summary."""
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


def pattern_summary(patterns: List[PatternInfo], 
                   data_shape: Optional[Tuple[int, int]] = None) -> PatternSummary:
    """
    Generate summary statistics for missingness patterns.
    
    Parameters
    ----------
    patterns : List[PatternInfo]
        Pattern information from analyze_patterns or little_mcar_test
    data_shape : Optional[Tuple[int, int]]
        Original data shape (n_obs, n_vars) if known
        
    Returns
    -------
    PatternSummary
        Summary statistics
    """
    n_patterns = len(patterns)
    total_cases = sum(p.n_cases for p in patterns)
    
    # Find complete cases pattern
    complete_pattern = None
    for p in patterns:
        if len(p.missing_indices) == 0:
            complete_pattern = p
            break
    
    complete_cases = complete_pattern.n_cases if complete_pattern else 0
    complete_cases_percent = complete_cases / total_cases if total_cases > 0 else 0
    
    # Most common pattern
    most_common = max(patterns, key=lambda p: p.n_cases)
    
    # Overall missing rate (need original data or shape)
    if data_shape:
        n_obs, n_vars = data_shape
        total_values = n_obs * n_vars
        observed_values = sum(p.n_cases * p.n_observed for p in patterns)
        overall_missing_rate = 1 - (observed_values / total_values)
    else:
        overall_missing_rate = np.nan
    
    # Variable-specific missing rates
    variable_missing_rates = {}
    if data_shape:
        n_vars = data_shape[1]
        for var_idx in range(n_vars):
            var_observed = sum(p.n_cases for p in patterns 
                             if var_idx in p.observed_indices)
            variable_missing_rates[var_idx] = 1 - (var_observed / total_cases)
    
    return PatternSummary(
        n_patterns=n_patterns,
        total_cases=total_cases,
        overall_missing_rate=overall_missing_rate,
        most_common_pattern=most_common,
        complete_cases=complete_cases,
        complete_cases_percent=complete_cases_percent,
        variable_missing_rates=variable_missing_rates
    )