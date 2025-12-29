"""
Smart Parking MVP - Tracking Test
Simple test to verify YOLO tracking is working correctly
"""
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time


def test_tracking(video_source='parking_video.mp4.mp4', model_path='yolov8m.pt'):
    """
    Test YOLO tracking functionality
    
    This is a minimal implementation to verify tracking works
    """
    print("="*60)
    print("VEHICLE TRACKING TEST")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Video: {video_source}")
    print("\nPress 'q' to quit\n")
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("[OK] Model loaded")
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return
    print("[OK] Video opened")
    
    # Get video properties
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Video: {width}x{height} @ {fps_video} FPS\n")
    
    # Tracking data
    track_history = defaultdict(lambda: [])
    vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    # Statistics
    frame_count = 0
    start_time = time.time()
    tracked_ids = set()
    
    print("Starting tracking...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[END] Video finished or cannot read frame")
            break
        
        frame_count += 1
        
        # CRITICAL: Use track() with persist=True
        results = model.track(
            source=frame,
            persist=True,              # MUST be True for tracking
            tracker='bytetrack.yaml',  # Use ByteTrack
            conf=0.4,                  # Confidence threshold
            iou=0.6,                   # IoU for NMS (higher = fewer duplicates)
            classes=[2, 3, 5, 7],      # Vehicle classes only
            verbose=False,             # Disable verbose output
            device='cuda'              # Use 'cpu' if no GPU
        )
        
        # Process detections
        tracked_count = 0
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Check if tracking ID exists
                if box.id is None:
                    continue
                
                # Get tracking ID
                track_id = int(box.id.item())
                tracked_ids.add(track_id)
                tracked_count += 1
                
                # Get box info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = vehicle_classes.get(cls, 'vehicle')
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Update track history
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                
                # Draw bounding box (GREEN)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Label background (GREEN)
                cv2.rectangle(frame, (x1, y1 - label_h - 10),
                             (x1 + label_w, y1), (0, 255, 0), -1)
                
                # Label text (BLACK)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw trajectory
                points = track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw statistics
        stats = [
            "TRACKING TEST",
            f"FPS: {fps:.1f}",
            f"Frame: {frame_count}",
            f"Tracked Now: {tracked_count}",
            f"Total IDs: {len(tracked_ids)}"
        ]
        
        # Stats background
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        
        # Stats text
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Display
        cv2.imshow('Tracking Test', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("TRACKING TEST RESULTS")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique vehicles tracked: {len(tracked_ids)}")
    print(f"Average FPS: {fps:.2f}")
    
    if len(tracked_ids) > 0:
        print("\n[SUCCESS] Tracking is working! ✓")
        print(f"Track IDs: {sorted(list(tracked_ids))[:20]}" + 
              ("..." if len(tracked_ids) > 20 else ""))
    else:
        print("\n[FAILED] No vehicles tracked ✗")
        print("Check:")
        print("  - Video has visible vehicles")
        print("  - Model confidence threshold (try lowering)")
        print("  - ByteTrack tracker file exists")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Get video source from command line or use default
    video = sys.argv[1] if len(sys.argv) > 1 else 'parking_video.mp4.mp4'
    
    test_tracking(video_source=video)
