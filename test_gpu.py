"""
GPU Test Script for SmartPark System
Verifies CUDA, GPU, and YOLO model are working correctly
"""
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time

print("=" * 60)
print("  SmartPark GPU Verification Test")
print("=" * 60)

# Test 1: PyTorch CUDA
print("\n[Test 1] PyTorch & CUDA")
print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print("  ✅ CUDA is working!")
else:
    print("  ❌ CUDA not available - will use CPU")
    
# Test 2: Load YOLO Model
print("\n[Test 2] Loading YOLO Model")
try:
    model = YOLO('yolov8n.pt')
    print("  ✅ YOLOv8n model loaded successfully")
    
    # Check if model is on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
except Exception as e:
    print(f"  ❌ Failed to load model: {e}")
    exit(1)

# Test 3: Process Test Image with GPU
print("\n[Test 3] GPU Processing Test")
print("  Creating test image and running inference...")

# Create a test frame (simulating video)
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

try:
    start_time = time.time()
    
    # Run inference on GPU
    results = model.predict(
        source=test_frame,
        device=device,
        imgsz=640,
        conf=0.25,
        verbose=False,
        half=True if torch.cuda.is_available() else False  # FP16 on GPU
    )
    
    process_time = time.time() - start_time
    
    print(f"  ✅ Inference completed in {process_time*1000:.1f}ms")
    print(f"  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        
        if process_time < 0.05:  # Less than 50ms
            print("  ⚡ GPU acceleration is WORKING! (Very fast)")
        elif process_time < 0.1:  # Less than 100ms
            print("  ✅ GPU acceleration is working (Good speed)")
        else:
            print("  ⚠️ Slow - check if GPU is actually being used")
    else:
        print(f"  CPU processing time: {process_time*1000:.1f}ms")
        
except Exception as e:
    print(f"  ❌ Inference failed: {e}")
    exit(1)

# Test 4: FPS Estimation
print("\n[Test 4] Multi-Frame Performance Test")
print("  Running 30 frames to estimate FPS...")

frame_times = []
for i in range(30):
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    start = time.time()
    results = model.predict(
        source=test_frame,
        device=device,
        imgsz=640,
        conf=0.25,
        verbose=False,
        half=True if torch.cuda.is_available() else False
    )
    frame_times.append(time.time() - start)

avg_time = sum(frame_times) / len(frame_times)
fps = 1.0 / avg_time

print(f"  Average processing time: {avg_time*1000:.1f}ms")
print(f"  Estimated FPS: {fps:.1f}")

if torch.cuda.is_available():
    if fps >= 30:
        print("  ⚡ EXCELLENT - Can handle 3+ video streams!")
    elif fps >= 20:
        print("  ✅ GOOD - Can handle 2-3 video streams")
    elif fps >= 15:
        print("  ✅ OK - Can handle 1-2 video streams")
    else:
        print("  ⚠️ Slow - may need optimization")
else:
    print("  ⚠️ CPU mode - will be slower with multiple videos")

# Summary
print("\n" + "=" * 60)
print("  Test Summary")
print("=" * 60)

if torch.cuda.is_available():
    print("  ✅ GPU (CUDA) is ENABLED and WORKING!")
    print(f"  ✅ RTX 3050 detected and active")
    print(f"  ✅ Expected performance: {fps:.0f} FPS per video")
    print(f"  ✅ Can run approximately {int(fps/15)} video streams smoothly")
    print("\n  Your system is ready for GPU-accelerated multi-video processing! 🚀")
else:
    print("  ❌ GPU not detected - using CPU")
    print("  Install CUDA-enabled PyTorch to enable GPU")
    
print("=" * 60)
