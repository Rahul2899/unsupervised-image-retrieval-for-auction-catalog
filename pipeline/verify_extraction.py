#!/usr/bin/env python3
"""
QUICK VERIFICATION SCRIPT
Tests the memory-critical fusion step with a simulated 660K-sized dataset
Runtime: ~2-5 minutes instead of 5 hours
"""

import torch
import torch.nn.functional as F
import psutil
import gc

def log_memory(tag=""):
    """Log both GPU and RAM"""
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] GPU: {gpu_alloc:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")

    mem = psutil.virtual_memory()
    ram_used = mem.used / 1024**3
    ram_avail = mem.available / 1024**3
    print(f"[{tag}] RAM: {ram_used:.2f}GB used, {ram_avail:.2f}GB available")

def l2_normalize(x):
    """L2 normalization"""
    return F.normalize(x, p=2, dim=1, eps=1e-8)

def test_fusion_cpu_based():
    """Test the CPU-based fusion approach"""
    print("="*60)
    print("TESTING CPU-BASED FUSION (NEW APPROACH)")
    print("="*60)

    # Simulate 660K dataset dimensions
    n_samples = 660302
    c4_dim = 1024
    c5_dim = 2048

    print(f"\nSimulating 660K dataset:")
    print(f"  C4: ({n_samples}, {c4_dim}) = {n_samples * c4_dim * 4 / 1024**3:.2f}GB")
    print(f"  C5: ({n_samples}, {c5_dim}) = {n_samples * c5_dim * 4 / 1024**3:.2f}GB")

    log_memory("Initial")

    # Create simulated features on CPU (like after concatenation)
    print("\n1. Creating simulated C4 and C5 features on CPU...")
    c4 = torch.randn(n_samples, c4_dim, dtype=torch.float32)  # On CPU
    c5 = torch.randn(n_samples, c5_dim, dtype=torch.float32)  # On CPU
    log_memory("After creating features")

    # Test L2 normalization on CPU
    print("\n2. L2 normalizing on CPU...")
    c4_norm = l2_normalize(c4)
    log_memory("After C4 normalize")

    c5_norm = l2_normalize(c5)
    log_memory("After C5 normalize")

    # Clear original features
    del c4, c5
    gc.collect()
    log_memory("After deleting originals")

    # Test projection on CPU
    print("\n3. Projecting C4 to C5 dimensions on CPU...")
    W = torch.randn(c5_dim, c4_dim) * 0.01  # On CPU
    c4_proj = F.linear(c4_norm, W)
    del W, c4_norm
    gc.collect()
    log_memory("After projection")

    # Test fusion on CPU
    print("\n4. Applying max fusion on CPU...")
    fused = torch.max(c4_proj, c5_norm)
    del c4_proj, c5_norm
    gc.collect()
    log_memory("After fusion")

    print(f"\nFused tensor: shape={fused.shape}, device={fused.device}")

    # Test moving to GPU (this is what happens in PCA)
    if torch.cuda.is_available():
        print("\n5. Testing GPU transfer (simulating PCA input)...")
        try:
            fused_gpu = fused.cuda()
            log_memory("After GPU transfer")
            print("✅ GPU transfer successful!")
            del fused_gpu
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"❌ GPU transfer failed: {e}")
            return False

    del fused
    gc.collect()
    log_memory("Final cleanup")

    print("\n✅ CPU-BASED FUSION TEST PASSED!")
    return True

def test_fusion_old_gpu_based():
    """Test the old GPU-based approach (should fail)"""
    print("\n" + "="*60)
    print("TESTING OLD GPU-BASED FUSION (SHOULD FAIL)")
    print("="*60)

    # Simulate 660K dataset dimensions
    n_samples = 660302
    c4_dim = 1024
    c5_dim = 2048

    log_memory("Initial")

    # Create simulated features on CPU
    print("\n1. Creating simulated C4 and C5 features on CPU...")
    c4 = torch.randn(n_samples, c4_dim, dtype=torch.float32)
    c5 = torch.randn(n_samples, c5_dim, dtype=torch.float32)
    log_memory("After creating features")

    if torch.cuda.is_available():
        try:
            print("\n2. Trying to move C4 to GPU (OLD APPROACH)...")
            c4_gpu = c4.cuda()
            log_memory("After C4 to GPU")

            print("\n3. Trying to move C5 to GPU (OLD APPROACH)...")
            c5_gpu = c5.cuda()
            log_memory("After C5 to GPU")

            print("❌ Unexpectedly succeeded - you have more GPU memory than expected")
            return True

        except RuntimeError as e:
            print(f"\n✅ Expected failure: {e}")
            print("This is why we need CPU-based fusion!")
            torch.cuda.empty_cache()
            return True
    else:
        print("No CUDA available, skipping GPU test")
        return True

def main():
    print("MEMORY FIX VERIFICATION")
    print("Testing with simulated 660K dataset dimensions")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_total:.1f}GB)")

    print()

    # Test old approach (should show the problem)
    test_fusion_old_gpu_based()

    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*60)
    input("Press ENTER to test the NEW CPU-based approach...")
    print()

    # Test new approach (should work)
    success = test_fusion_cpu_based()

    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    if success:
        print("✅ The CPU-based fusion approach works correctly!")
        print("✅ Your extractfull.py should now complete without OOM errors")
        print("\nYou can safely run the full extraction now.")
    else:
        print("❌ The test failed - there may still be issues")

    print("="*60)

if __name__ == "__main__":
    main()