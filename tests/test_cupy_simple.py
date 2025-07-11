# Create test_cupy_simple.py
import cupy as cp
import numpy as np

# Test basic operations
try:
    print("Testing basic CuPy operations...")
    x = cp.array([1, 2, 3, 4, 5])
    y = cp.sum(x)
    print(f"Basic operation works: {y}")
    
    # Test linear algebra
    print("Testing matrix operations...")
    A = cp.random.randn(100, 100, dtype=cp.float32)  # Start small
    B = cp.matmul(A, A.T)
    print(f"Matrix multiplication works: {B.shape}")
    
    # Test Cholesky on small matrix
    print("Testing Cholesky...")
    small_matrix = cp.eye(10, dtype=cp.float32) + 0.1 * cp.random.randn(10, 10)
    small_matrix = small_matrix @ small_matrix.T  # Make positive definite
    chol = cp.linalg.cholesky(small_matrix)
    print(f"Cholesky works: {chol.shape}")
    
except Exception as e:
    print(f"Error: {e}")