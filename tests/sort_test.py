#!/usr/bin/env python3
"""
Debug the exact sorting behavior to understand the pattern order issue.
"""

import numpy as np
from pymvnmle import datasets

def debug_sorting():
    """Debug the exact sorting of Apple dataset."""
    
    data = datasets.apple
    print("ORIGINAL APPLE DATA:")
    for i, row in enumerate(data):
        print(f"  Row {i+1:2d}: [{row[0]:4.0f}, {str(row[1]):>5s}]")
    
    print("\nPATTERN ANALYSIS:")
    
    # Create binary representation
    is_observed = (~np.isnan(data)).astype(int)
    print("\nBinary patterns (1=observed, 0=missing):")
    for i, pattern in enumerate(is_observed):
        print(f"  Row {i+1:2d}: {pattern}")
    
    # Calculate pattern codes using R's method
    powers = 2 ** np.arange(2 - 1, -1, -1)  # [2, 1] for 2 variables
    print(f"\nPowers used: {powers}")
    
    pattern_codes = is_observed @ powers
    print("\nPattern codes:")
    for i, code in enumerate(pattern_codes):
        print(f"  Row {i+1:2d}: {code} (binary: {is_observed[i]}, data: [{data[i,0]:4.0f}, {str(data[i,1]):>5s}])")
    
    # Sort by pattern code
    sort_indices = np.argsort(pattern_codes)
    print(f"\nSort indices: {sort_indices + 1}")  # +1 for 1-based indexing
    
    print("\nSORTED DATA:")
    sorted_data = data[sort_indices]
    sorted_codes = pattern_codes[sort_indices]
    for i, (row, code) in enumerate(zip(sorted_data, sorted_codes)):
        print(f"  Position {i+1:2d}: [{row[0]:4.0f}, {str(row[1]):>5s}] (code: {code})")
    
    # Count patterns
    unique_codes, counts = np.unique(sorted_codes, return_counts=True)
    print(f"\nPattern summary:")
    for code, count in zip(unique_codes, counts):
        print(f"  Code {code}: {count} observations")

if __name__ == "__main__":
    debug_sorting()