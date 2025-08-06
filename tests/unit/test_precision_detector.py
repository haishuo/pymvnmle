#!/usr/bin/env python3
"""
Test suite for precision_detector.py

Run this to verify hardware detection works correctly on your system.
Also includes mock tests for hardware you don't have access to.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvnmle._backends.precision_detector import (
    detect_gpu_capabilities,
    PrecisionSupport,
    GPUCapabilities,
    validate_fp64_request,
    recommend_precision,
    _classify_nvidia_gpu,
    print_capabilities
)


class TestPrecisionDetection(unittest.TestCase):
    """Test hardware detection on actual system."""
    
    def test_actual_hardware_detection(self):
        """Test detection on whatever hardware this is running on."""
        print("\n" + "="*60)
        print("ACTUAL HARDWARE TEST")
        print("="*60)
        
        caps = detect_gpu_capabilities()
        
        # Should always return something
        self.assertIsNotNone(caps)
        self.assertIsInstance(caps, GPUCapabilities)
        
        # Print what we found
        print(f"Detected GPU: {caps.gpu_name}")
        print(f"GPU Type: {caps.gpu_type}")
        print(f"FP64 Support: {caps.fp64_support.value}")
        print(f"FP64/FP32 Ratio: {caps.fp64_throughput_ratio:.4f}")
        print(f"Recommended FP64: {caps.recommended_fp64}")
        
        # Basic sanity checks
        self.assertIn(caps.gpu_type, ['cuda', 'metal', 'none'])
        self.assertIsInstance(caps.has_gpu, bool)
        self.assertIsInstance(caps.recommended_fp64, bool)
        
        # If no GPU, should recommend FP64 (CPU always supports it)
        if not caps.has_gpu:
            self.assertEqual(caps.fp64_support, PrecisionSupport.NO_GPU)
            self.assertTrue(caps.recommended_fp64)
        
        # If Metal, should not support FP64
        if caps.gpu_type == 'metal':
            self.assertEqual(caps.fp64_support, PrecisionSupport.NO_FP64)
            self.assertFalse(caps.recommended_fp64)
            self.assertEqual(caps.fp64_throughput_ratio, 0.0)
        
        # If CUDA, should have some classification
        if caps.gpu_type == 'cuda':
            self.assertIn(caps.fp64_support, [
                PrecisionSupport.FULL_FP64,
                PrecisionSupport.GIMPED_FP64
            ])
            self.assertGreater(caps.fp64_throughput_ratio, 0)
    
    def test_print_capabilities(self):
        """Test the debug print function."""
        print("\n" + "="*60)
        print("PRINT CAPABILITIES TEST")
        print("="*60)
        
        # Should not raise any errors
        try:
            print_capabilities()
        except Exception as e:
            self.fail(f"print_capabilities() raised {e}")


class TestNvidiaClassification(unittest.TestCase):
    """Test classification of various NVIDIA GPUs."""
    
    def test_rtx_4090_classification(self):
        """Test RTX 4090 is correctly classified as gimped."""
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA GeForce RTX 4090")
        self.assertEqual(support, PrecisionSupport.GIMPED_FP64)
        self.assertEqual(ratio, 1/64)
        self.assertFalse(recommended)
    
    def test_rtx_5070ti_classification(self):
        """Test RTX 5070 Ti (your Forge GPU) is correctly classified."""
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA GeForce RTX 5070 Ti")
        self.assertEqual(support, PrecisionSupport.GIMPED_FP64)
        self.assertEqual(ratio, 1/64)  # Assuming same as 40 series
        self.assertFalse(recommended)
    
    def test_a100_classification(self):
        """Test A100 is correctly classified as full FP64."""
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA A100-SXM4-40GB")
        self.assertEqual(support, PrecisionSupport.FULL_FP64)
        self.assertEqual(ratio, 0.5)
        self.assertTrue(recommended)
    
    def test_h100_classification(self):
        """Test H100 is correctly classified as full FP64."""
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA H100 80GB HBM3")
        self.assertEqual(support, PrecisionSupport.FULL_FP64)
        self.assertEqual(ratio, 0.5)
        self.assertTrue(recommended)
    
    def test_old_titan_classification(self):
        """Test old Titan cards (like at UMass Dartmouth)."""
        # Most Titans are gimped
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA TITAN X")
        self.assertEqual(support, PrecisionSupport.GIMPED_FP64)
        self.assertEqual(ratio, 1/32)
        self.assertFalse(recommended)
        
        # Except Titan V
        support, ratio, recommended = _classify_nvidia_gpu("NVIDIA TITAN V")
        self.assertEqual(support, PrecisionSupport.FULL_FP64)
        self.assertEqual(ratio, 0.5)
        self.assertTrue(recommended)
    
    def test_unknown_gpu_classification(self):
        """Test unknown GPU defaults to gimped (safe assumption)."""
        with self.assertWarns(UserWarning):
            support, ratio, recommended = _classify_nvidia_gpu("NVIDIA Future GPU 9999")
        self.assertEqual(support, PrecisionSupport.GIMPED_FP64)
        self.assertEqual(ratio, 1/32)
        self.assertFalse(recommended)


class TestMockedHardware(unittest.TestCase):
    """Test detection with mocked hardware to simulate different GPUs."""
    
    def test_mock_rtx_4090(self):
        """Simulate RTX 4090 detection."""
        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        
        # Patch the import to return our mock
        with patch.dict('sys.modules', {'torch': mock_torch}):
            caps = detect_gpu_capabilities()
        
        self.assertTrue(caps.has_gpu)
        self.assertEqual(caps.gpu_type, "cuda")
        self.assertEqual(caps.fp64_support, PrecisionSupport.GIMPED_FP64)
        self.assertFalse(caps.recommended_fp64)
    
    def test_mock_apple_metal(self):
        """Simulate Apple Metal detection."""
        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        
        # Patch the import to return our mock
        with patch.dict('sys.modules', {'torch': mock_torch}):
            caps = detect_gpu_capabilities()
        
        self.assertTrue(caps.has_gpu)
        self.assertEqual(caps.gpu_type, "metal")
        self.assertEqual(caps.fp64_support, PrecisionSupport.NO_FP64)
        self.assertFalse(caps.recommended_fp64)
    
    def test_mock_no_gpu(self):
        """Simulate CPU-only system."""
        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        
        # Patch the import to return our mock
        with patch.dict('sys.modules', {'torch': mock_torch}):
            caps = detect_gpu_capabilities()
        
        self.assertFalse(caps.has_gpu)
        self.assertEqual(caps.gpu_type, "none")
        self.assertEqual(caps.fp64_support, PrecisionSupport.NO_GPU)
        self.assertTrue(caps.recommended_fp64)  # CPU always supports FP64
    
    def test_no_torch_installed(self):
        """Test behavior when PyTorch is not installed."""
        # Mock ImportError when trying to import torch
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            caps = detect_gpu_capabilities()
        
        # Should fall back to CPU
        self.assertFalse(caps.has_gpu)
        self.assertEqual(caps.gpu_type, "none")
        self.assertEqual(caps.fp64_support, PrecisionSupport.NO_GPU)
        self.assertTrue(caps.recommended_fp64)


class TestValidation(unittest.TestCase):
    """Test FP64 request validation."""
    
    def test_fp64_request_on_metal_raises(self):
        """Test that FP64 request on Metal raises error."""
        metal_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="Apple M2 Max",
            gpu_type="metal",
            fp64_support=PrecisionSupport.NO_FP64,
            fp64_throughput_ratio=0.0,
            recommended_fp64=False
        )
        
        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            validate_fp64_request(metal_caps, force_fp64=True)
        
        self.assertIn("not supported", str(context.exception))
        self.assertIn("Apple Metal", str(context.exception))
    
    def test_fp64_request_on_gimped_warns(self):
        """Test that FP64 request on gimped GPU warns."""
        gimped_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_type="cuda",
            fp64_support=PrecisionSupport.GIMPED_FP64,
            fp64_throughput_ratio=1/64,
            recommended_fp64=False
        )
        
        # Should warn but not raise
        with self.assertWarns(UserWarning) as warning:
            validate_fp64_request(gimped_caps, force_fp64=True)
        
        self.assertIn("64x slower", str(warning.warning))
    
    def test_fp32_request_always_works(self):
        """Test that FP32 request works on any hardware."""
        # Test on Metal
        metal_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="Apple M2 Max",
            gpu_type="metal",
            fp64_support=PrecisionSupport.NO_FP64,
            fp64_throughput_ratio=0.0,
            recommended_fp64=False
        )
        
        # Should not raise or warn
        validate_fp64_request(metal_caps, force_fp64=False)


class TestRecommendation(unittest.TestCase):
    """Test precision recommendation logic."""
    
    def test_auto_recommendation_follows_hardware(self):
        """Test auto mode follows hardware recommendation."""
        # Full FP64 hardware
        full_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="NVIDIA A100",
            gpu_type="cuda",
            fp64_support=PrecisionSupport.FULL_FP64,
            fp64_throughput_ratio=0.5,
            recommended_fp64=True
        )
        
        # Auto should recommend FP64
        self.assertTrue(recommend_precision(full_caps, user_preference=None))
        
        # Gimped hardware
        gimped_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_type="cuda",
            fp64_support=PrecisionSupport.GIMPED_FP64,
            fp64_throughput_ratio=1/64,
            recommended_fp64=False
        )
        
        # Auto should recommend FP32
        self.assertFalse(recommend_precision(gimped_caps, user_preference=None))
    
    def test_explicit_preference_overrides(self):
        """Test explicit user preference overrides recommendation."""
        gimped_caps = GPUCapabilities(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_type="cuda",
            fp64_support=PrecisionSupport.GIMPED_FP64,
            fp64_throughput_ratio=1/64,
            recommended_fp64=False
        )
        
        # User wants FP64 despite gimped hardware
        with self.assertWarns(UserWarning):
            result = recommend_precision(gimped_caps, user_preference=True)
        self.assertTrue(result)
        
        # User wants FP32 explicitly
        result = recommend_precision(gimped_caps, user_preference=False)
        self.assertFalse(result)


def run_hardware_test():
    """Quick function to just test actual hardware detection."""
    print("\n" + "="*70)
    print("QUICK HARDWARE DETECTION TEST")
    print("="*70)
    
    caps = detect_gpu_capabilities()
    
    print(f"\n📊 Detection Results:")
    print(f"  GPU Available: {caps.has_gpu}")
    print(f"  GPU Name: {caps.gpu_name}")
    print(f"  GPU Type: {caps.gpu_type}")
    print(f"  FP64 Support: {caps.fp64_support.value}")
    
    if caps.fp64_support == PrecisionSupport.GIMPED_FP64:
        slowdown = int(1/caps.fp64_throughput_ratio)
        print(f"  ⚠️  FP64 Performance: {slowdown}x slower than FP32")
    elif caps.fp64_support == PrecisionSupport.NO_FP64:
        print(f"  ❌ FP64 Performance: Not supported")
    elif caps.fp64_support == PrecisionSupport.FULL_FP64:
        print(f"  ✅ FP64 Performance: Full speed")
    
    print(f"\n💡 Recommendation: Use {'FP64' if caps.recommended_fp64 else 'FP32'}")
    
    # Test validation
    print(f"\n🧪 Testing FP64 request validation...")
    try:
        validate_fp64_request(caps, force_fp64=True)
        print("  ✅ FP64 request would succeed (possibly with warning)")
    except RuntimeError as e:
        print(f"  ❌ FP64 request would fail: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test precision detection")
    parser.add_argument("--quick", action="store_true", 
                       help="Just run hardware detection test")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_hardware_test()
    else:
        # Run full test suite
        unittest.main(argv=[''], verbosity=2 if args.verbose else 1)