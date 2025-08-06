"""
Hardware precision capability detection for PyMVNMLE.

This module detects GPU hardware and determines FP64 computation capabilities.
No defaults, no assumptions - explicit detection only.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class PrecisionSupport(Enum):
    """FP64 support level for hardware."""
    NO_GPU = "no_gpu"           # No GPU available
    NO_FP64 = "no_fp64"          # GPU exists but no FP64 (Apple Metal)
    GIMPED_FP64 = "gimped_fp64"  # FP64 exists but slow (consumer NVIDIA)
    FULL_FP64 = "full_fp64"      # Full-speed FP64 (A100, H100)


@dataclass
class GPUCapabilities:
    """
    GPU capability information.
    
    Attributes
    ----------
    has_gpu : bool
        Whether any GPU is available
    gpu_name : str
        Human-readable GPU name (e.g., "NVIDIA GeForce RTX 4090")
    gpu_type : str
        Backend type: 'cuda', 'metal', or 'none'
    fp64_support : PrecisionSupport
        Level of FP64 support
    fp64_throughput_ratio : float
        Ratio of FP64 to FP32 throughput (e.g., 1/32 for RTX 4090)
    recommended_fp64 : bool
        Whether FP64 is recommended for this hardware
    """
    has_gpu: bool
    gpu_name: str
    gpu_type: str
    fp64_support: PrecisionSupport
    fp64_throughput_ratio: float
    recommended_fp64: bool


def detect_gpu_capabilities() -> GPUCapabilities:
    """
    Detect GPU hardware and FP64 capabilities.
    
    Returns
    -------
    GPUCapabilities
        Detected hardware capabilities and recommendations
        
    Notes
    -----
    Detection priority: CUDA > Metal > CPU-only
    
    FP64 classifications:
    - Apple Metal: No FP64 support at all
    - Consumer NVIDIA (RTX/GTX): Gimped FP64 (1/32 or 1/64 ratio)
    - Data center NVIDIA (A100/H100): Full FP64 (1/2 ratio)
    - Unknown NVIDIA: Assume gimped unless proven otherwise
    """
    # Try CUDA first
    cuda_caps = _detect_cuda_capabilities()
    if cuda_caps is not None:
        return cuda_caps
    
    # Try Metal second
    metal_caps = _detect_metal_capabilities()
    if metal_caps is not None:
        return metal_caps
    
    # No GPU available
    return GPUCapabilities(
        has_gpu=False,
        gpu_name="CPU only",
        gpu_type="none",
        fp64_support=PrecisionSupport.NO_GPU,
        fp64_throughput_ratio=1.0,
        recommended_fp64=True  # CPU always supports FP64
    )


def _detect_cuda_capabilities() -> Optional[GPUCapabilities]:
    """Detect NVIDIA CUDA GPU capabilities."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
            
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
        
        # Classify GPU based on name
        fp64_support, ratio, recommended = _classify_nvidia_gpu(gpu_name)
        
        return GPUCapabilities(
            has_gpu=True,
            gpu_name=gpu_name,
            gpu_type="cuda",
            fp64_support=fp64_support,
            fp64_throughput_ratio=ratio,
            recommended_fp64=recommended
        )
        
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"CUDA detection failed: {e}")
        return None


def _detect_metal_capabilities() -> Optional[GPUCapabilities]:
    """Detect Apple Metal GPU capabilities."""
    try:
        import torch
        
        if not hasattr(torch.backends, 'mps'):
            return None
            
        if not torch.backends.mps.is_available():
            return None
        
        # Metal detected - but no FP64 support
        return GPUCapabilities(
            has_gpu=True,
            gpu_name="Apple Metal GPU",
            gpu_type="metal",
            fp64_support=PrecisionSupport.NO_FP64,
            fp64_throughput_ratio=0.0,  # No FP64 at all
            recommended_fp64=False
        )
        
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Metal detection failed: {e}")
        return None


def _classify_nvidia_gpu(gpu_name: str) -> Tuple[PrecisionSupport, float, bool]:
    """
    Classify NVIDIA GPU's FP64 capabilities based on model name.
    
    Parameters
    ----------
    gpu_name : str
        GPU name from torch.cuda.get_device_name()
        
    Returns
    -------
    (support_level, throughput_ratio, recommended)
        
    Notes
    -----
    Known classifications:
    - A100, H100: Full FP64 (1/2 ratio)
    - V100: Full FP64 (1/2 ratio)
    - RTX 40xx/30xx/20xx: Gimped (1/64 ratio)
    - GTX 16xx/10xx: Gimped (1/32 ratio)
    - Quadro: Varies (usually 1/32)
    - Tesla: Usually full FP64
    - Titan: Varies by generation
    """
    gpu_upper = gpu_name.upper()
    
    # Data center GPUs with full FP64
    full_fp64_models = [
        'A100', 'A800',  # Ampere data center
        'H100', 'H800',  # Hopper data center
        'V100',          # Volta data center
        'P100',          # Pascal data center
        'TESLA K80',     # Kepler data center
        'TESLA K40',     # Kepler data center
        'TESLA K20',     # Kepler data center
    ]
    
    for model in full_fp64_models:
        if model in gpu_upper:
            return PrecisionSupport.FULL_FP64, 0.5, True
    
    # Consumer GPUs with gimped FP64
    # RTX 40 series (Ada Lovelace) - 1/64 ratio
    if 'RTX 40' in gpu_upper or 'RTX 50' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/64, False
    
    # RTX 30 series (Ampere) - 1/64 ratio
    if 'RTX 30' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/64, False
    
    # RTX 20 series (Turing) - 1/32 ratio
    if 'RTX 20' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # GTX series - generally 1/32 ratio
    if 'GTX' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # Quadro - varies, assume gimped
    if 'QUADRO' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # Titan - varies by generation
    if 'TITAN' in gpu_upper:
        if 'TITAN V' in gpu_upper:  # Volta Titan has good FP64
            return PrecisionSupport.FULL_FP64, 0.5, True
        else:  # Most Titans are gimped
            return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # Unknown NVIDIA GPU - assume gimped to be safe
    warnings.warn(
        f"Unknown NVIDIA GPU '{gpu_name}'. Assuming gimped FP64. "
        f"Please report this model for proper classification."
    )
    return PrecisionSupport.GIMPED_FP64, 1/32, False


def validate_fp64_request(capabilities: GPUCapabilities, 
                         force_fp64: bool) -> None:
    """
    Validate user's FP64 request against hardware capabilities.
    
    Parameters
    ----------
    capabilities : GPUCapabilities
        Detected hardware capabilities
    force_fp64 : bool
        User's request to use FP64
        
    Raises
    ------
    RuntimeError
        If FP64 requested on hardware without support (Apple Metal)
        
    Warns
    -----
    UserWarning
        If FP64 requested on gimped hardware (will be slow)
    """
    if not force_fp64:
        return  # FP32 always works
    
    if capabilities.fp64_support == PrecisionSupport.NO_FP64:
        raise RuntimeError(
            f"FP64 requested but not supported on {capabilities.gpu_name}. "
            f"Apple Metal does not support FP64 computation. "
            f"Please use FP32 (use_fp64=False) or switch to CPU."
        )
    
    if capabilities.fp64_support == PrecisionSupport.GIMPED_FP64:
        warnings.warn(
            f"FP64 requested on {capabilities.gpu_name} with gimped FP64 "
            f"(throughput ratio: {capabilities.fp64_throughput_ratio:.3f}). "
            f"This will be {int(1/capabilities.fp64_throughput_ratio)}x slower than FP32. "
            f"Consider using FP32 (use_fp64=False) for better performance.",
            UserWarning
        )
    
    if capabilities.fp64_support == PrecisionSupport.NO_GPU:
        # CPU - FP64 works but might want to inform user
        pass  # No warning needed, CPU FP64 is fine


def recommend_precision(capabilities: GPUCapabilities,
                        user_preference: Optional[bool]) -> bool:
    """
    Recommend FP64 vs FP32 based on hardware and user preference.
    
    Parameters
    ----------
    capabilities : GPUCapabilities
        Detected hardware capabilities
    user_preference : Optional[bool]
        User's FP64 preference (None for auto)
        
    Returns
    -------
    bool
        True for FP64, False for FP32
        
    Notes
    -----
    Decision logic:
    1. User explicit preference (if valid for hardware)
    2. Hardware recommendation (full FP64 → True, otherwise → False)
    """
    # User explicitly requested
    if user_preference is not None:
        validate_fp64_request(capabilities, user_preference)
        return user_preference
    
    # Auto-select based on hardware
    return capabilities.recommended_fp64


# Convenience function for quick testing
def print_capabilities() -> None:
    """Print detected GPU capabilities (for debugging)."""
    caps = detect_gpu_capabilities()
    
    print("GPU Capability Detection")
    print("=" * 50)
    print(f"GPU Available: {caps.has_gpu}")
    print(f"GPU Name: {caps.gpu_name}")
    print(f"GPU Type: {caps.gpu_type}")
    print(f"FP64 Support: {caps.fp64_support.value}")
    print(f"FP64/FP32 Ratio: {caps.fp64_throughput_ratio:.4f}")
    print(f"Recommended FP64: {caps.recommended_fp64}")
    
    if caps.fp64_support == PrecisionSupport.GIMPED_FP64:
        print(f"⚠️  FP64 is {int(1/caps.fp64_throughput_ratio)}x slower than FP32")
    elif caps.fp64_support == PrecisionSupport.NO_FP64:
        print("❌ No FP64 support on this GPU")
    elif caps.fp64_support == PrecisionSupport.FULL_FP64:
        print("✅ Full-speed FP64 available")


if __name__ == "__main__":
    # Test detection when run directly
    print_capabilities()