"""
OPTIMIZED SMART PARKING SYSTEM - HIGH FPS VERSION
Key optimizations:
1. Smaller/faster YOLO model (yolov8n.pt)
2. Reduced image size (640px instead of 960px)
3. Skip vehicle analysis (license plate OCR) or do it less frequently
4. Async API calls (non-blocking)
5. Frame skipping option
6. Reduced processing per frame
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from shapely.geometry import Polygon, box as shapely_box
import torch
from parking_duration_tracker import ParkingDurationTracker
from violation_detector import ViolationDetector
from vehicle_analyzer import get_analyzer
import requests
from concurrent.futures import ThreadPoolExecutor
import threading


# ============================================================================
# PARKING SLOT CLASS (Same as original)
# ============================================================================

class ParkingSlot:
    """Represents one parking slot with occupancy tracking"""
    
    def __init__(self, slot_id, points):
        self.id = slot_id
        self.points = points
        self.polygon = Polygon(points)
        
        # Vehicle ID locking
        self.locked_vehicle_id = None
        self.locked_entry_time = None
        self.locked_bbox = None
        
        # Vehicle details
        self.license_plate = None
        self.vehicle_type = None
        self.vehicle_color = None
        
        # Stability counters
        self.stability_frames = 8
        self.unlock_frames = 30
        
        # Pending detection
        self.pending_vehicle_id = None
        self.pending_bbox = None
        self.pending_stable_count = 0
        
        # Empty verification
        self.empty_frame_count = 0
        
        # Occupancy state
        self.is_occupied = False
        self.occupied_since = None
        self.last_vacant = time.time()
        
        # Statistics
        self.total_occupancies = 0
        self.total_duration = 0.0
    
    def check_overlap(self, bbox):
        """Check if vehicle bounding box overlaps with this slot"""
        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)
        
        if not self.polygon.intersects(vehicle_box):
            return 0.0
        
        intersection = self.polygon.intersection(vehicle_box).area
        slot_area = self.polygon.area
        
        return intersection / slot_area if slot_area > 0 else 0.0
    
    def check_strict_occupancy(self, bbox):
        """Check if vehicle occupies this slot"""
        overlap = self.check_overlap(bbox)
        if self.id == 5:
            threshold = 0.70
        elif self.id in [6, 7]:
            threshold = 0.70
        else:
            threshold = 0.50
        return overlap >= threshold
    
    def lock_vehicle_id(self, vehicle_id, bbox, vehicle_details=None):
        """Lock a vehicle ID to this slot"""
        self.locked_vehicle_id = vehicle_id
        self.locked_entry_time = time.time()
        self.locked_bbox = bbox
        self.is_occupied = True
        self.occupied_since = self.locked_entry_time
        self.total_occupancies += 1
        self.empty_frame_count = 0
        self.pending_vehicle_id = None
        self.pending_stable_count = 0
        
        if vehicle_details:
            self.license_plate = vehicle_details.get('license_plate', 'N/A')
            self.vehicle_type = vehicle_details.get('vehicle_type', 'car')
            self.vehicle_color = vehicle_details.get('color', 'unknown')
        
        print(f"[SLOT #{self.id}] 🔒 LOCKED - Vehicle ID:{vehicle_id}")
    
    def unlock_vehicle_id(self):
        """Unlock and remove vehicle ID"""
        if self.locked_vehicle_id is not None:
            duration = 0
            if self.locked_entry_time:
                duration = time.time() - self.locked_entry_time
                self.total_duration += duration
            print(f"[SLOT #{self.id}] 🔓 UNLOCKED - Vehicle ID:{self.locked_vehicle_id}, Duration: {duration:.1f}s")
            
            self.locked_vehicle_id = None
            self.locked_entry_time = None
            self.locked_bbox = None
            self.is_occupied = False
            self.occupied_since = None
            self.last_vacant = time.time()
            self.empty_frame_count = 0
            
            self.license_plate = None
            self.vehicle_type = None
            self.vehicle_color = None
    
    def get_duration(self):
        """Get current parking duration in seconds"""
        if not self.is_occupied or not self.locked_entry_time:
            return 0
        return time.time() - self.locked_entry_time
    
    def get_locked_vehicle_id(self):
        """Get the locked vehicle ID"""
        return self.locked_vehicle_id


# ============================================================================
# OPTIMIZED SMART PARKING SYSTEM
# ============================================================================

class SmartParkingMVP:
    """
    OPTIMIZED parking system for high FPS
    """
    
    def __init__(self, slots_json, model_path='yolov8n.pt', 
                 enable_vehicle_analysis=False, 
                 analysis_interval=30,
                 skip_frames=0):
        """
        Initialize optimized parking system
        
        Args:
            slots_json: Path to parking slots JSON
            model_path: YOLO model (yolov8n.pt for speed, yolov8s.pt for accuracy)
            enable_vehicle_analysis: Enable license plate/color detection (SLOW!)
            analysis_interval: Frames between vehicle analysis (higher = faster)
            skip_frames: Skip N frames between processing (0 = process all frames)
        """
        print("="*70)
        print("OPTIMIZED SMART PARKING SYSTEM - HIGH FPS MODE")
        print("="*70)
        
        # GPU check
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("⚠️  Running on CPU (slower)")
        
        # Load YOLO (use nano model for speed)
        print(f"\nLoading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
        print("✅ Model loaded")
        
        # Load parking slots
        print(f"\nLoading slots: {slots_json}")
        with open(slots_json, 'r') as f:
            data = json.load(f)
        
        self.slots = [ParkingSlot(s['id'], s['points']) for s in data['slots']]
        print(f"✅ Loaded {len(self.slots)} parking slots")
        
        # Duration Tracker (with async API calls)
        print("\n🕒 Initializing Duration Tracker...")
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5000')
        self.duration_tracker = ParkingDurationTracker(
            backend_url=backend_url,
            stability_frames=5,
            min_overlap=0.20
        )
        print(f"✅ Duration Tracker initialized")
        
        # Vehicle Analyzer (OPTIONAL - SLOW!)
        self.enable_vehicle_analysis = enable_vehicle_analysis
        self.analysis_interval = analysis_interval
        self.analysis_frame_counter = 0
        
        if self.enable_vehicle_analysis:
            print("\n🚗 Initializing Vehicle Analyzer (WARNING: SLOW!)...")
            self.vehicle_analyzer = get_analyzer()
            print("✅ Vehicle Analyzer initialized")
        else:
            self.vehicle_analyzer = None
            print("\n⚡ Vehicle Analyzer DISABLED for maximum speed")
        
        # Frame skipping
        self.skip_frames = skip_frames
        self.frame_skip_counter = 0
        
        # Configuration - OPTIMIZED FOR SPEED
        self.overlap_threshold = 0.20
        self.conf_threshold = 0.15
        self.min_vehicle_area = 5000
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Vehicle ID generation
        self.next_vehicle_id = 1
        self.tracker_to_vehicle_id = {}
        
        # Violation detection
        self.violation_detector = ViolationDetector()
        self.violation_check_interval = 30
        self.last_violation_check = 0
        self.backend_url = backend_url
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Statistics
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = time.time()
        self.fps_list = []
        
        print("\nOptimization Settings:")
        print(f"  Model: {model_path} (smaller = faster)")
        print(f"  Image Size: 640px (reduced from 960px)")
        print(f"  Vehicle Analysis: {'ENABLED' if enable_vehicle_analysis else 'DISABLED'}")
        if enable_vehicle_analysis:
            print(f"  Analysis Interval: Every {analysis_interval} frames")
        print(f"  Frame Skipping: {skip_frames} frames (0 = no skipping)")
        print(f"  Confidence: {self.conf_threshold}")
        print(f"  API Calls: Async (non-blocking)")
        print("="*70 + "\n")
    
    def process_frame(self, frame):
        """Process one frame - OPTIMIZED VERSION"""
        self.frame_count += 1
        
        # Frame skipping
        if self.skip_frames > 0:
            self.frame_skip_counter += 1
            if self.frame_skip_counter <= self.skip_frames:
                return []  # Skip this frame
            self.frame_skip_counter = 0
        
        self.processed_frames += 1
        
        # Optimized downscale
        h, w = frame.shape[:2]
        if w > 1280:  # Downscale less aggressively
            scale = 1280 / w
            process_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            scale_back = 1.0 / scale
        else:
            process_frame = frame
            scale_back = 1.0
        
        # YOLO tracking - OPTIMIZED SETTINGS
        results = self.model.track(
            source=process_frame,
            device=self.device,
            half=False,
            imgsz=640,   # REDUCED from 960 for speed
            conf=0.15,
            iou=0.4,
            classes=self.vehicle_classes,
            verbose=False,
            max_det=100,  # Reduced from 200
            persist=True,
            tracker='bytetrack.yaml'
        )
        
        # Extract detected vehicles
        all_vehicles = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get tracker ID
                tracker_id = None
                if hasattr(box, 'id') and box.id is not None:
                    tracker_id = int(box.id.item())
                
                # Convert tracker ID to stable vehicle ID
                if tracker_id is not None:
                    if tracker_id not in self.tracker_to_vehicle_id:
                        self.tracker_to_vehicle_id[tracker_id] = self.next_vehicle_id
                        self.next_vehicle_id += 1
                    vehicle_id = self.tracker_to_vehicle_id[tracker_id]
                else:
                    vehicle_id = int((x1 + x2 + y1 + y2) / 4) % 100000
                
                # Scale back
                if scale_back != 1.0:
                    x1, y1, x2, y2 = int(x1 * scale_back), int(y1 * scale_back), int(x2 * scale_back), int(y2 * scale_back)
                
                # Filter small detections
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.min_vehicle_area:
                    continue
                
                vehicle_class = int(box.cls[0].item())
                
                all_vehicles.append({
                    'id': vehicle_id,
                    'tracker_id': tracker_id,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf_score,
                    'area': bbox_area,
                    'class': vehicle_class
                })
        
        # SLOT-BASED FILTERING
        slot_vehicles = []
        for vehicle in all_vehicles:
            for slot in self.slots:
                overlap = slot.check_overlap(vehicle['bbox'])
                if overlap >= 0.20:
                    slot_vehicles.append(vehicle)
                    break
        
        # VEHICLE ANALYSIS (OPTIONAL - only every N frames)
        if self.enable_vehicle_analysis and slot_vehicles and self.vehicle_analyzer:
            self.analysis_frame_counter += 1
            if self.analysis_frame_counter >= self.analysis_interval:
                self.analysis_frame_counter = 0
                # Analyze in background thread to avoid blocking
                self.executor.submit(self._analyze_vehicles_async, frame.copy(), slot_vehicles)
        
        # Update duration tracker
        self.duration_tracker.update(self.slots)
        
        # Update slot occupancy
        self.update_occupancy_simple(slot_vehicles)
        
        # Check violations (less frequently)
        if self.frame_count - self.last_violation_check >= self.violation_check_interval:
            self.check_violations(slot_vehicles)
            self.last_violation_check = self.frame_count
        
        return slot_vehicles
    
    def _analyze_vehicles_async(self, frame, vehicles):
        """Analyze vehicles in background thread (non-blocking)"""
        for vehicle in vehicles:
            try:
                analysis = self.vehicle_analyzer.analyze_vehicle(
                    frame, 
                    vehicle['bbox'], 
                    vehicle.get('class', 2)
                )
                vehicle.update(analysis)
            except Exception as e:
                pass  # Ignore errors in async analysis
    
    def update_occupancy_simple(self, vehicles):
        """SLOT-CENTRIC VEHICLE ID LOCKING SYSTEM (same as original)"""
        for slot in self.slots:
            overlapping_vehicles = []
            for vehicle in vehicles:
                overlap_ratio = slot.check_overlap(vehicle['bbox'])
                if slot.id == 5:
                    threshold = 0.70
                elif slot.id == 6:
                    threshold = 0.90
                else:
                    threshold = 0.50
                if overlap_ratio > threshold:
                    overlapping_vehicles.append((vehicle, overlap_ratio))
            
            overlapping_vehicles.sort(key=lambda x: x[1], reverse=True)
            vehicle_in_slot = overlapping_vehicles[0][0] if overlapping_vehicles else None
            
            # CASE 1: Slot has locked ID
            if slot.locked_vehicle_id is not None:
                if vehicle_in_slot is not None:
                    slot.empty_frame_count = 0
                    slot.locked_bbox = vehicle_in_slot['bbox']
                else:
                    slot.empty_frame_count += 1
                    if slot.empty_frame_count >= slot.unlock_frames:
                        slot.unlock_vehicle_id()
            
            # CASE 2: Slot is vacant
            else:
                if vehicle_in_slot is not None:
                    if slot.pending_vehicle_id == vehicle_in_slot['id']:
                        slot.pending_stable_count += 1
                        slot.pending_bbox = vehicle_in_slot['bbox']
                        
                        if slot.pending_stable_count >= slot.stability_frames:
                            vehicle_details = {
                                'license_plate': vehicle_in_slot.get('license_plate', 'N/A'),
                                'vehicle_type': vehicle_in_slot.get('vehicle_type', 'car'),
                                'color': vehicle_in_slot.get('color', 'unknown')
                            }
                            slot.lock_vehicle_id(vehicle_in_slot['id'], vehicle_in_slot['bbox'], vehicle_details)
                    else:
                        # Different vehicle or first detection
                        slot.pending_vehicle_id = vehicle_in_slot['id']
                        slot.pending_bbox = vehicle_in_slot['bbox']
                        slot.pending_stable_count = 1
                else:
                    # No vehicle
                    slot.pending_vehicle_id = None
                    slot.pending_stable_count = 0
    
    def check_violations(self, vehicles):
        """Check for parking violations (same as original)"""
        try:
            for slot in self.slots:
                if slot.is_occupied:
                    duration = slot.get_duration()
                    violations = self.violation_detector.check_violations(
                        slot.id, duration, slot.locked_vehicle_id
                    )
                    
                    if violations:
                        payload = {
                            'slot_id': slot.id,
                            'vehicle_id': slot.locked_vehicle_id,
                            'violations': violations,
                            'timestamp': datetime.now().isoformat()
                        }
                        # Send async (non-blocking)
                        self.executor.submit(self._send_violation_async, payload)
        except Exception as e:
            pass  # Don't let violations crash the system
    
    def _send_violation_async(self, payload):
        """Send violation to backend in background thread"""
        try:
            requests.post(
                f"{self.backend_url}/api/violations",
                json=payload,
                timeout=2
            )
        except:
            pass  # Ignore network errors
    
    def draw_visualization(self, frame, vehicles):
        """Draw visualization (same as original but simplified)"""
        output = frame.copy()
        
        # Draw slots
        for slot in self.slots:
            points = np.array(slot.points, dtype=np.int32)
            color = (0, 0, 255) if slot.is_occupied else (0, 255, 0)
            cv2.polylines(output, [points], True, color, 2)
            
            # Slot label
            center_x = int(sum([p[0] for p in slot.points]) / 4)
            center_y = int(sum([p[1] for p in slot.points]) / 4)
            
            label = f"#{slot.id}"
            if slot.is_occupied:
                duration = slot.get_duration()
                label += f" {int(duration)}s"
            
            cv2.putText(output, label, (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS
        fps = len(self.fps_list) / sum(self.fps_list) if self.fps_list else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def format_duration(self, seconds):
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def run_video(self, video_path, output_video=None, display=True):
        """Run parking system on video - OPTIMIZED"""
        print(f"\n▶️  Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Video properties
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps_video} FPS, {total_frames} frames")
        
        # Output video
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps_video, (width, height))
        
        # Display window
        if display:
            display_height = 720
            display_scale = display_height / height
            display_width = int(width * display_scale)
            window_name = 'Smart Parking MVP - OPTIMIZED'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display_width, display_height)
        
        frame_times = []
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("\n[END] Video finished")
                break
            
            # Process frame
            vehicles = self.process_frame(frame)
            
            # Visualize
            annotated = self.draw_visualization(frame, vehicles)
            
            # Write output
            if writer:
                writer.write(annotated)
            
            # Display
            if display:
                display_frame = cv2.resize(annotated, (display_width, display_height)) if display_scale < 1.0 else annotated
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[QUIT] User stopped")
                    break
            
            # Calculate FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_time = sum(frame_times) / len(frame_times)
            self.fps_list = frame_times
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Progress
            if self.processed_frames % 30 == 0:
                print(f"Frame {self.frame_count}/{total_frames} | FPS: {current_fps:.1f} | "
                      f"Occupied: {sum(1 for s in self.slots if s.is_occupied)}/{len(self.slots)}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        print(f"\n✅ Processing complete!")
        print(f"Total frames: {self.frame_count}")
        print(f"Processed frames: {self.processed_frames}")
        print(f"Average FPS: {self.processed_frames / (time.time() - self.start_time):.1f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    DEFAULT_VIDEO = 'parking_evening_vedio.mp4'
    DEFAULT_SLOTS = 'configs/parking_slots.json'
    
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Smart Parking System')
    parser.add_argument('video', nargs='?', default=DEFAULT_VIDEO, help='Video file path')
    parser.add_argument('--slots', default=DEFAULT_SLOTS, help='Slots JSON file')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLO model (yolov8n.pt=fastest, yolov8s.pt=balanced, yolov8m.pt=accurate)')
    parser.add_argument('--analyze', action='store_true', 
                       help='Enable vehicle analysis (license plate/color) - SLOW!')
    parser.add_argument('--analysis-interval', type=int, default=30,
                       help='Frames between vehicle analysis (default: 30)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between processing (default: 0 = no skipping)')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Create system
    system = SmartParkingMVP(
        args.slots,
        model_path=args.model,
        enable_vehicle_analysis=args.analyze,
        analysis_interval=args.analysis_interval,
        skip_frames=args.skip_frames
    )
    
    # Run
    system.run_video(
        args.video,
        output_video=args.output,
        display=not args.no_display
    )
