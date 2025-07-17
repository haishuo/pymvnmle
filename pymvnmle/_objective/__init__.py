"""
Objective function factory for PyMVNMLE.
Provides backend-specific implementations of the MLE objective function.
"""

def get_objective(data, backend='numpy', **kwargs):
    """
    Get objective function for specified backend.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix with missing values as np.nan
    backend : str, default='numpy'
        Computational backend: 'numpy', 'pytorch', 'gpu', 'cpu'
    **kwargs
        Backend-specific options (e.g., device for GPU backends)
    
    Returns
    -------
    objective
        Backend-specific objective function instance
    
    Raises
    ------
    ValueError
        If backend is not recognized
    """
    backend = backend.lower()
    
    if backend in ['numpy', 'cpu']:
        from .numpy_objective import NumpyMLEObjective
        return NumpyMLEObjective(data)
        
    elif backend in ['pytorch', 'torch', 'gpu', 'cuda']:
        from .torch_objective import TorchMLEObjective
        device = kwargs.get('device', 'cuda')
        return TorchMLEObjective(data, device=device)
        
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available options: 'numpy', 'pytorch', 'gpu'"
        )


# Convenience alias for backward compatibility
MLEObjective = get_objective