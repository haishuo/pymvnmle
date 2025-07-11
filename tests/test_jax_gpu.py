import jax
import jax.numpy as jnp
import time

print(f"JAX devices: {jax.devices()}")

# Force computation on GPU
with jax.default_device(jax.devices('gpu')[0]):
    # Large matrix test
    print("Testing JAX on GPU...")
    
    key = jax.random.PRNGKey(42)
    large_matrix = jax.random.normal(key, (5000, 5000))
    large_matrix = large_matrix @ large_matrix.T  # Make positive definite
    
    start = time.time()
    chol = jnp.linalg.cholesky(large_matrix)
    chol.block_until_ready()  # Ensure computation completes
    gpu_time = time.time() - start
    
    print(f"JAX GPU time: {gpu_time:.3f}s")