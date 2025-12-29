"""
Quick FPS Test - Compare Original vs Optimized
"""
import cv2
import time
from ultralytics import YOLO
import torch

def test_model_speed(model_path, imgsz, num_frames=30):
    """Test YOLO model speed"""
    print(f"\nTesting: {model_path} @ {imgsz}px")
    print("-" * 50)
    
    # Load model
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Open video
    cap = cv2.VideoCapture('parking_evening_vedio.mp4')
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            model.track(source=frame, device=device, imgsz=imgsz, 
                       verbose=False, persist=True)
    
    # Test
    print(f"Testing {num_frames} frames...")
    times = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        results = model.track(
            source=frame,
            device=device,
            imgsz=imgsz,
            conf=0.15,
            iou=0.4,
            classes=[2, 3, 5, 7],
            verbose=False,
            persist=True,
            tracker='bytetrack.yaml'
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            avg_fps = len(times) / sum(times)
            print(f"  Frame {i+1}/{num_frames} - Current FPS: {avg_fps:.1f}")
    
    cap.release()
    
    # Results
    avg_time = sum(times) / len(times)
    avg_fps = 1.0 / avg_time
    min_time = min(times)
    max_time = max(times)
    max_fps = 1.0 / min_time
    min_fps = 1.0 / max_time
    
    print(f"\n✅ Results for {model_path} @ {imgsz}px:")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Best FPS: {max_fps:.1f}")
    print(f"   Worst FPS: {min_fps:.1f}")
    print(f"   Avg time per frame: {avg_time*1000:.1f}ms")
    
    return avg_fps


def test_ocr_speed(num_tests=10):
    """Test EasyOCR speed"""
    try:
        import easyocr
        import numpy as np
        
        print(f"\nTesting: EasyOCR")
        print("-" * 50)
        
        print("Loading EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Create dummy vehicle image
        test_img = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        
        # Warm up
        print("Warming up...")
        reader.readtext(test_img, detail=0)
        
        # Test
        print(f"Testing {num_tests} OCR operations...")
        times = []
        
        for i in range(num_tests):
            start = time.time()
            results = reader.readtext(test_img, detail=0, paragraph=False)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Test {i+1}/{num_tests}: {elapsed*1000:.1f}ms")
        
        # Results
        avg_time = sum(times) / len(times)
        
        print(f"\n✅ OCR Results:")
        print(f"   Average time per vehicle: {avg_time*1000:.1f}ms")
        print(f"   Time for 5 vehicles: {avg_time*5*1000:.1f}ms")
        print(f"   FPS with 5 vehicles: {1.0/(avg_time*5):.2f}")
        
        return avg_time
        
    except ImportError:
        print("⚠️  EasyOCR not installed, skipping test")
        return None


def main():
    print("=" * 70)
    print("SMART PARKING FPS PERFORMANCE TEST")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  Running on CPU")
    
    # Test different configurations
    configs = [
        ('yolov8n.pt', 640, 'OPTIMIZED (Recommended)'),
        ('yolov8n.pt', 960, 'Nano + High Res'),
        ('yolov8s.pt', 640, 'Small + Low Res'),
        ('yolov8s.pt', 960, 'ORIGINAL (Slow)'),
    ]
    
    results = {}
    
    print("\n" + "=" * 70)
    print("YOLO DETECTION TESTS")
    print("=" * 70)
    
    for model, imgsz, name in configs:
        try:
            fps = test_model_speed(model, imgsz, num_frames=30)
            results[name] = {'fps': fps, 'ocr': False}
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
    
    # Test OCR
    print("\n" + "=" * 70)
    print("VEHICLE ANALYSIS (OCR) TEST")
    print("=" * 70)
    
    ocr_time = test_ocr_speed(num_tests=10)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    print("\n📊 YOLO-Only Performance (No Vehicle Analysis):")
    print(f"{'Configuration':<30} {'FPS':<10} {'Notes'}")
    print("-" * 70)
    
    for name, data in results.items():
        fps = data['fps']
        if fps > 20:
            emoji = "🟢"
        elif fps > 10:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        print(f"{emoji} {name:<28} {fps:>6.1f} FPS")
    
    if ocr_time:
        print("\n📊 With Vehicle Analysis (License Plate OCR):")
        print(f"{'Configuration':<30} {'FPS':<10} {'Notes'}")
        print("-" * 70)
        
        for name, data in results.items():
            fps_base = data['fps']
            # Assume 5 vehicles per frame
            ocr_overhead = ocr_time * 5
            total_time = (1.0 / fps_base) + ocr_overhead
            fps_with_ocr = 1.0 / total_time if total_time > 0 else 0
            
            if fps_with_ocr > 5:
                emoji = "🟡"
            elif fps_with_ocr > 1:
                emoji = "🔴"
            else:
                emoji = "⛔"
            
            note = ""
            if fps_with_ocr < 1:
                note = " ← THIS IS YOUR PROBLEM!"
            
            print(f"{emoji} {name:<28} {fps_with_ocr:>6.1f} FPS{note}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n✅ Best Configuration:")
    print("   Model: yolov8n.pt")
    print("   Image Size: 640px")
    print("   Vehicle Analysis: DISABLED or every 60 frames")
    print("   Expected FPS: 15-25 FPS")
    
    print("\n❌ Worst Configuration (Your Current Setup):")
    print("   Model: yolov8s.pt")
    print("   Image Size: 960px")
    print("   Vehicle Analysis: EVERY FRAME")
    print("   Expected FPS: 0.1-0.5 FPS ← Unusable!")
    
    print("\n🚀 To fix your FPS issue:")
    print("   cd smart_parking_mvp")
    print("   python smart_parking_mvp_optimized.py")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
