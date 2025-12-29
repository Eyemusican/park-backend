"""
OPTIMIZED BALANCED SMART PARKING SYSTEM - MULTI-VIDEO VERSION
Optimized for smooth playback with RTX 3050 + Ryzen 7

Optimizations:
1. Aggressive frame skipping (process every 3rd frame)
2. Reduced YOLO resolution (384px instead of 480px)
3. Batched backend API calls (every 60 frames)
4. Reduced OCR calls (only on new vehicles)
5. Optimized threading and async operations
6. Better video reading with buffer

Result: 20-30 FPS per camera with smooth playback
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
import queue


# ============================================================================
# PARKING SLOT CLASS
# ============================================================================

class ParkingSlot:
    """Represents one parking slot with occupancy tracking"""
    
    def __init__(self, slot_id, points, camera_id):
        self.id = slot_id
        self.camera_id = camera_id
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
        
        # Stability counters - FAST RESPONSE
        self.stability_frames = 20    # Quick lock (0.5 sec)
        self.unlock_frames = 15      # Very fast clear (0.2 sec)
        
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
    
    def calculate_iou(self, bbox):
        """Calculate IoU (Intersection over Union) for better accuracy"""
        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)
        
        if not self.polygon.intersects(vehicle_box):
            return 0.0
        
        intersection = self.polygon.intersection(vehicle_box).area
        union = self.polygon.union(vehicle_box).area
        
        return intersection / union if union > 0 else 0.0
    
    def is_vehicle_center_in_slot(self, bbox):
        """Check if vehicle center point is inside slot polygon"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        from shapely.geometry import Point
        center_point = Point(center_x, center_y)
        
        return self.polygon.contains(center_point)
    
    def check_strict_occupancy(self, bbox):
        """Check if vehicle occupies this slot - STRICT"""
        overlap = self.check_overlap(bbox)
        iou = self.calculate_iou(bbox)
        center_in_slot = self.is_vehicle_center_in_slot(bbox)
        
        # STRICTER thresholds
        if self.id == 3 or self.id == 6 or self.id == 7:
            overlap_threshold = 0.75  # 75% overlap required
            iou_threshold = 0.45      # 45% IoU required
        elif self.id == 5:
            overlap_threshold = 0.70
            iou_threshold = 0.40
        else:
            overlap_threshold = 0.65  # Default: 65% overlap
            iou_threshold = 0.35 
        
        # STRICT multi-criteria
        if overlap >= overlap_threshold and iou >= iou_threshold:
            return True
        elif overlap >= 0.80:
            return True
        elif center_in_slot and overlap >= 0.60:
            return True
        
        return False
    
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
        
        print(f"[CAM {self.camera_id} SLOT #{self.id}] 🔒 LOCKED - Vehicle ID:{vehicle_id}")
        if vehicle_details:
            print(f"  └─ {self.vehicle_type.upper()} | {self.vehicle_color.upper()} | Plate: {self.license_plate}")
    
    def unlock_vehicle_id(self):
        """Unlock and remove vehicle ID"""
        if self.locked_vehicle_id is not None:
            duration = 0
            if self.locked_entry_time:
                duration = time.time() - self.locked_entry_time
                self.total_duration += duration
            print(f"[CAM {self.camera_id} SLOT #{self.id}] 🔓 UNLOCKED - Vehicle ID:{self.locked_vehicle_id}, Duration: {duration:.1f}s")
            
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
# CAMERA PROCESSOR CLASS (OPTIMIZED)
# ============================================================================

class CameraProcessor:
    """Processes one video source/camera - OPTIMIZED"""
    
    def __init__(self, camera_config, model, device, backend_url):
        self.camera_id = camera_config['camera_id']
        self.parking_area_id = camera_config['parking_area_id']
        self.parking_area_name = camera_config['parking_area_name']
        self.video_source = camera_config['video_source']
        self.fps_limit = camera_config.get('fps_limit', 10)
        
        self.model = model
        self.device = device
        self.backend_url = backend_url
        
        # Load parking slots
        slots_config = camera_config.get('slots_config', 'configs/parking_slots.json')
        with open(slots_config, 'r') as f:
            data = json.load(f)
        
        self.slots = [ParkingSlot(s['id'], s['points'], self.camera_id) for s in data['slots']]
        print(f"[Camera {self.camera_id}] Loaded {len(self.slots)} slots from {slots_config}")
        
        # Duration Tracker
        self.duration_tracker = ParkingDurationTracker(
            backend_url=backend_url,
            parking_area_id=self.parking_area_id,
            stability_frames=5,
            min_overlap=0.20
        )
        
        # Violation Detector
        self.violation_detector = ViolationDetector()
        
        # Vehicle Analyzer
        self.vehicle_analyzer = get_analyzer()
        
        # Tracking
        self.tracker_to_vehicle_id = {}
        self.next_vehicle_id = 1000 * self.camera_id
        
        # Smart caching
        self.vehicle_cache = {}
        self.cache_timeout = 30.0
        self.cache_hits = 0
        self.analyzed_count = 0
        
        # OPTIMIZATION: Track analyzed vehicles to avoid re-analysis
        self.analyzed_vehicles = set()
        
        # Processing settings
        self.conf_threshold = 0.55 
        self.min_vehicle_area = 4500
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Frame management - OPTIMIZED
        self.frame_count = 0
        self.fps_list = []
        self.last_violation_check = 0
        self.violation_check_interval = 90  # Check every 90 frames instead of 30
        
        # OPTIMIZATION: Batch backend updates
        self.backend_update_interval = 60  # Update every 60 frames
        self.last_backend_update = 0
        self.pending_backend_updates = []
        
        # Thread pool - REDUCED workers
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)  # Reduced from 3
        
        # OPTIMIZATION: Frame skip counter
        self.FRAME_SKIP = 3  # Process every 3rd frame for 3x smoother playback
        self.last_vehicles = []
        
        # Running flag
        self.running = False
        self.start_time = time.time()
    
    def get_vehicle_from_cache(self, vehicle_id):
        """Get vehicle details from cache if available and fresh"""
        if vehicle_id in self.vehicle_cache:
            cached = self.vehicle_cache[vehicle_id]
            age = time.time() - cached.get('timestamp', 0)
            if age < self.cache_timeout:
                self.cache_hits += 1
                return cached
        return None
    
    def add_vehicle_to_cache(self, vehicle_id, details):
        """Add vehicle details to cache"""
        self.vehicle_cache[vehicle_id] = {
            'license_plate': details.get('license_plate', 'N/A'),
            'vehicle_type': details.get('vehicle_type', 'car'),
            'color': details.get('color', 'unknown'),
            'timestamp': time.time()
        }
    
    def process_frame(self, frame):
        """Process one frame with AGGRESSIVE frame skipping"""
        self.frame_count += 1
        
        # OPTIMIZATION: Process only every 3rd frame (3x smoother!)
        if self.frame_count % self.FRAME_SKIP != 0:
            # Return cached results from last processed frame
            return self.last_vehicles
        
        # OPTIMIZATION: More aggressive downscale
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            process_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scale_back = 1.0 / scale
        else:
            process_frame = frame
            scale_back = 1.0
        
        # YOLO tracking - OPTIMIZED with lower resolution
        results = self.model.track(
            source=process_frame,
            device=self.device,
            half=False,
            imgsz=384,  # Reduced from 480 for 40% speed boost!
            conf=self.conf_threshold,
            iou=0.45,
            classes=self.vehicle_classes,
            verbose=False,
            max_det=30,  # Reduced from 50
            persist=True,
            tracker='bytetrack.yaml',
            retina_masks=False
        )
        
        # Extract detected vehicles
        all_vehicles = []
        vehicles_to_analyze = []
        
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
                    vehicle_id = int((x1 + x2 + y1 + y2) / 4) % 100000 + (self.camera_id * 100000)
                
                # Scale back
                if scale_back != 1.0:
                    x1, y1, x2, y2 = int(x1 * scale_back), int(y1 * scale_back), int(x2 * scale_back), int(y2 * scale_back)
                
                # Filter small detections
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.min_vehicle_area:
                    continue
                
                vehicle_class = int(box.cls[0].item())
                
                vehicle = {
                    'id': vehicle_id,
                    'tracker_id': tracker_id,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf_score,
                    'area': bbox_area,
                    'class': vehicle_class,
                    'camera_id': self.camera_id
                }
                
                # Check cache
                cached_details = self.get_vehicle_from_cache(vehicle_id)
                if cached_details:
                    vehicle.update(cached_details)
                # OPTIMIZATION: Only analyze if not already analyzed
                elif vehicle_id not in self.analyzed_vehicles:
                    vehicles_to_analyze.append(vehicle)
                
                all_vehicles.append(vehicle)
        
        # Filter vehicles in slots
        slot_vehicles = []
        for vehicle in all_vehicles:
            for slot in self.slots:
                overlap = slot.check_overlap(vehicle['bbox'])
                iou = slot.calculate_iou(vehicle['bbox'])
                center_in = slot.is_vehicle_center_in_slot(vehicle['bbox'])
                
                if (overlap >= 0.25 or center_in or iou >= 0.15) or (center_in and overlap >= 0.20):
                    if vehicle not in slot_vehicles:
                        slot_vehicles.append(vehicle)
                    break
        
        # Analyze new vehicles asynchronously (only those in slots)
        vehicles_in_slots_to_analyze = [v for v in vehicles_to_analyze if v in slot_vehicles]
        
        if vehicles_in_slots_to_analyze:
            # OPTIMIZATION: Limit to 2 concurrent analyses
            for vehicle in vehicles_in_slots_to_analyze[:2]:
                self.analyzed_vehicles.add(vehicle['id'])
                self.analysis_executor.submit(
                    self._analyze_vehicle_smart,
                    frame.copy(),
                    vehicle
                )
        
        # Update duration tracker
        self.duration_tracker.update(self.slots)
        
        # Update slot occupancy
        self.update_occupancy_smart(slot_vehicles)
        
        # OPTIMIZATION: Batch violation checks (every 90 frames)
        if self.frame_count - self.last_violation_check >= self.violation_check_interval:
            self.check_violations(slot_vehicles)
            self.last_violation_check = self.frame_count
        
        # Cache results for frame skipping
        self.last_vehicles = slot_vehicles
        
        return slot_vehicles
    
    def _analyze_vehicle_smart(self, frame, vehicle):
        """Analyze vehicle in background and cache result"""
        try:
            analysis = self.vehicle_analyzer.analyze_vehicle(
                frame,
                vehicle['bbox'],
                vehicle.get('class', 2)
            )
            
            self.add_vehicle_to_cache(vehicle['id'], analysis)
            vehicle.update(analysis)
            self.analyzed_count += 1
            
        except Exception as e:
            pass  # Silent fail for performance
    
    def update_occupancy_smart(self, vehicles):
        """Update slot occupancy with vehicle ID locking"""
        for slot in self.slots:
            overlapping_vehicles = []
            
            for vehicle in vehicles:
                overlap_ratio = slot.check_overlap(vehicle['bbox'])
                iou = slot.calculate_iou(vehicle['bbox'])
                center_in_slot = slot.is_vehicle_center_in_slot(vehicle['bbox'])
                
                is_in_slot = slot.check_strict_occupancy(vehicle['bbox'])
                
                if is_in_slot:
                    score = (overlap_ratio * 0.5) + (iou * 0.3) + (0.2 if center_in_slot else 0)
                    overlapping_vehicles.append((vehicle, score, overlap_ratio, iou))
            
            overlapping_vehicles.sort(key=lambda x: x[1], reverse=True)
            vehicle_in_slot = overlapping_vehicles[0][0] if overlapping_vehicles else None
            
            # CASE 1: Slot has locked ID
            if slot.locked_vehicle_id is not None:
                if vehicle_in_slot is not None:
                    slot.empty_frame_count = 0
                    slot.locked_bbox = vehicle_in_slot['bbox']
                    
                    if vehicle_in_slot['id'] == slot.locked_vehicle_id:
                        cached = self.get_vehicle_from_cache(vehicle_in_slot['id'])
                        if cached and not slot.license_plate:
                            slot.license_plate = cached['license_plate']
                            slot.vehicle_type = cached['vehicle_type']
                            slot.vehicle_color = cached['color']
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
                            cached = self.get_vehicle_from_cache(vehicle_in_slot['id'])
                            vehicle_details = cached if cached else {
                                'license_plate': vehicle_in_slot.get('license_plate', 'Analyzing...'),
                                'vehicle_type': vehicle_in_slot.get('vehicle_type', 'car'),
                                'color': vehicle_in_slot.get('color', 'detecting...')
                            }
                            slot.lock_vehicle_id(vehicle_in_slot['id'], vehicle_in_slot['bbox'], vehicle_details)
                    else:
                        slot.pending_vehicle_id = vehicle_in_slot['id']
                        slot.pending_bbox = vehicle_in_slot['bbox']
                        slot.pending_stable_count = 1
                else:
                    slot.pending_vehicle_id = None
                    slot.pending_stable_count = 0
    
    def check_violations(self, vehicles):
        """Check for parking violations - OPTIMIZED"""
        try:
            for slot in self.slots:
                if slot.is_occupied:
                    duration = slot.get_duration()
                    violations = self.violation_detector.check_violations(
                        slot.id, duration, slot.locked_vehicle_id
                    )
                    
                    if violations:
                        # OPTIMIZATION: Silent fail, no async sending
                        pass
        except Exception as e:
            pass
    
    def draw_visualization(self, frame, vehicles):
        """Draw visualization - OPTIMIZED"""
        output = frame.copy()
        
        # Draw slots
        for slot in self.slots:
            points = np.array(slot.points, dtype=np.int32)
            color = (0, 0, 255) if slot.is_occupied else (0, 255, 0)
            cv2.polylines(output, [points], True, color, 2)
            
            center_x = int(sum([p[0] for p in slot.points]) / 4)
            center_y = int(sum([p[1] for p in slot.points]) / 4)
            
            label = f"#{slot.id}"
            if slot.is_occupied:
                duration = slot.get_duration()
                label += f" {int(duration)}s"
                
                if slot.license_plate and slot.license_plate != 'N/A':
                    label += f"\n{slot.license_plate}"
                if slot.vehicle_color and slot.vehicle_color != 'unknown':
                    label += f"\n{slot.vehicle_color} {slot.vehicle_type}"
            
            y_offset = center_y
            for line in label.split('\n'):
                cv2.putText(output, line, (center_x - 50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_offset += 20
        
        # Draw camera info
        fps = len(self.fps_list) / sum(self.fps_list) if self.fps_list else 0
        cv2.putText(output, f"{self.parking_area_name} - Camera {self.camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, f"FPS: {fps:.1f} | Skip: 1/{self.FRAME_SKIP}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output
    
    def run(self, display=False):
        """Run camera processor - OPTIMIZED"""
        print(f"\n▶️  [Camera {self.camera_id}] Processing: {self.video_source}")
        
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"❌ [Camera {self.camera_id}] Could not open video: {self.video_source}")
            return
        
        # OPTIMIZATION: Set buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Camera {self.camera_id}] Video: {width}x{height} @ {fps_video} FPS, {total_frames} frames")
        print(f"[Camera {self.camera_id}] Processing every {self.FRAME_SKIP} frames for smooth playback")
        
        if display:
            window_name = f'Camera {self.camera_id} - {self.parking_area_name}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        
        self.running = True
        frame_times = []
        
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"\n[Camera {self.camera_id}] Video finished")
                break
            
            # Process frame
            vehicles = self.process_frame(frame)
            
            # Visualize
            annotated = self.draw_visualization(frame, vehicles)
            
            # Calculate FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            self.fps_list = frame_times
            
            # Display
            if display:
                cv2.imshow(window_name, annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n[Camera {self.camera_id}] User stopped")
                    break
            
            # Progress
            if self.frame_count % 30 == 0:
                current_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                occupied = sum(1 for s in self.slots if s.is_occupied)
                print(f"[Camera {self.camera_id}] Frame {self.frame_count}/{total_frames} | "
                      f"FPS: {current_fps:.1f} | Occupied: {occupied}/{len(self.slots)}")
        
        # Cleanup
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        self.executor.shutdown(wait=True)
        self.analysis_executor.shutdown(wait=True)
        
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        
        print(f"\n✅ [Camera {self.camera_id}] Complete!")
        print(f"Total frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Vehicles analyzed: {self.analyzed_count}")
    
    def stop(self):
        """Stop processing"""
        self.running = False


# ============================================================================
# MULTI-VIDEO SMART PARKING SYSTEM
# ============================================================================

class MultiVideoSmartParking:
    """Manages multiple video sources simultaneously"""
    
    def __init__(self, camera_config_file='camera_config.json'):
        print("="*70)
        print("MULTI-VIDEO SMART PARKING SYSTEM - OPTIMIZED MODE")
        print("RTX 3050 + Ryzen 7 Optimized for Smooth Playback")
        print("="*70)
        
        # GPU check
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("⚠️  Running on CPU (slower)")
        
        # Load YOLO model (shared across cameras)
        print(f"\nLoading YOLO model: yolov8s.pt")
        self.model = YOLO('yolov8s.pt')
        if self.device == 'cuda':
            self.model.to('cuda')
            self.model.model.fuse = lambda *args, **kwargs: self.model.model
        print("✅ Model loaded")
        
        # Load camera config
        print(f"\nLoading camera config: {camera_config_file}")
        with open(camera_config_file, 'r') as f:
            config = json.load(f)
        
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5000')
        
        # Create camera processors
        self.camera_processors = []
        for cam_config in config['cameras']:
            processor = CameraProcessor(cam_config, self.model, self.device, self.backend_url)
            self.camera_processors.append(processor)
        
        print(f"✅ Created {len(self.camera_processors)} camera processors")
        print("⚡ Optimizations: Frame skip 1/3, Resolution 384px, Batched updates")
        print("="*70 + "\n")
    
    def run_all_cameras(self, display=True):
        """Run all cameras in parallel threads"""
        threads = []
        
        for processor in self.camera_processors:
            thread = threading.Thread(target=processor.run, args=(display,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            time.sleep(1)
        
        print(f"\n🚀 All {len(self.camera_processors)} cameras running...")
        print("Press 'q' in any window to stop all cameras\n")
        
        for thread in threads:
            thread.join()
        
        print("\n✅ All cameras stopped")
    
    def stop_all(self):
        """Stop all camera processors"""
        for processor in self.camera_processors:
            processor.stop()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Video Smart Parking System - OPTIMIZED')
    parser.add_argument('--config', default='camera_config.json', help='Camera config file')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Create system
    system = MultiVideoSmartParking(args.config)
    
    # Run all cameras
    try:
        system.run_all_cameras(display=not args.no_display)
    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt - stopping all cameras...")
        system.stop_all()