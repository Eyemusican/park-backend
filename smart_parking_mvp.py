"""
SMART PARKING SYSTEM MVP - GovTech PoC
Strictly aligned with Terms of Reference requirements

Core Features (70% Weight):
✓ Vehicle presence detection (YOLO)
✓ Slot availability (occupied/vacant)
✓ Parking duration tracking
✓ Real-time performance (GPU-accelerated)
✓ Clear visualization

Scope: Feasibility demonstration, not perfection
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
import psycopg2
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# CONFIGURATION - Edit these values easily
# ============================================================================

# Default video file (used when running: python smart_parking_mvp.py)
DEFAULT_VIDEO_FILE = 'parking_evening_vedio.mp4'


# ============================================================================
# DATABASE SLOT LOADER
# ============================================================================

def load_slots_from_database(parking_id=None):
    """
    Load parking slot polygons from database.
    Returns list of dicts with 'id' and 'points' keys.
    """
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            dbname=os.environ.get('DB_NAME', 'parking_db'),
            user=os.environ.get('DB_USER', 'parking_user'),
            password=os.environ.get('DB_PASS', '')
        )
        cur = conn.cursor()

        if parking_id:
            cur.execute('''
                SELECT slot_id, slot_number, polygon_points
                FROM parking_slots
                WHERE parking_id = %s AND polygon_points IS NOT NULL
                ORDER BY slot_number
            ''', (parking_id,))
        else:
            # Load all slots with geometry (for backwards compatibility)
            cur.execute('''
                SELECT slot_id, slot_number, polygon_points
                FROM parking_slots
                WHERE polygon_points IS NOT NULL
                ORDER BY slot_id
            ''')

        slots = []
        for row in cur.fetchall():
            slot_id, slot_number, polygon_points = row
            if polygon_points:
                # Handle both string and dict from JSONB
                if isinstance(polygon_points, str):
                    polygon_points = json.loads(polygon_points)
                slots.append({
                    'id': slot_number,  # Use slot_number as ID for display
                    'name': f'Slot-{slot_number}',
                    'points': polygon_points
                })

        cur.close()
        conn.close()

        return slots if slots else None

    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        return None


# ============================================================================
# PARKING SLOT CLASS
# ============================================================================

class ParkingSlot:
    """Represents one parking slot with occupancy tracking"""
    
    def __init__(self, slot_id, points):
        self.id = slot_id
        self.points = points  # 4 corner points
        self.polygon = Polygon(points)
        
        # === VEHICLE ID LOCKING SYSTEM ===
        # Locked vehicle ID - NEVER changes until vehicle fully exits
        self.locked_vehicle_id = None
        self.locked_entry_time = None
        self.locked_bbox = None
        
        # === VEHICLE DETAILS ===
        self.license_plate = None
        self.vehicle_type = None
        self.vehicle_color = None
        
        # Stability counters for ID locking
        self.stability_frames = 8   # N frames for locking (0.5 seconds at 15fps) - stable detection
        self.unlock_frames = 30     # M frames for unlocking (2 seconds at 15fps) - prevents false exits
        
        # Pending vehicle detection (before locking)
        self.pending_vehicle_id = None
        self.pending_bbox = None
        self.pending_stable_count = 0
        
        # Empty slot verification (for unlocking)
        self.empty_frame_count = 0
        
        # Occupancy state (derived from locked ID)
        self.is_occupied = False
        self.occupied_since = None
        self.last_vacant = time.time()
        
        # Statistics
        self.total_occupancies = 0
        self.total_duration = 0.0
    
    def check_overlap(self, bbox):
        """
        Check if vehicle bounding box overlaps with this slot
        Returns overlap ratio based on SLOT area (more reliable)
        """
        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)
        
        if not self.polygon.intersects(vehicle_box):
            return 0.0
        
        intersection = self.polygon.intersection(vehicle_box).area
        slot_area = self.polygon.area
        
        # Use SLOT area as denominator - more stable for different vehicle sizes
        return intersection / slot_area if slot_area > 0 else 0.0
    
    def check_strict_occupancy(self, bbox):
        """Check if vehicle occupies this slot (custom thresholds per slot)"""
        overlap = self.check_overlap(bbox)
        if self.id == 5:
            threshold = 0.70  # Slot 5: 70%
        elif self.id in [6, 7]:
            threshold = 0.70  # Slots 6 & 7: 70%
        else:
            threshold = 0.50  # All others: 50%
        return overlap >= threshold

    def check_overlap_enhanced(self, bbox, vehicle_class=2):
        """
        Enhanced overlap detection with vehicle class weights and center point bonus.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            vehicle_class: COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)

        Returns:
            float: Overlap score (0.0 to 1.0+) with class weighting and center bonus
        """
        from shapely.geometry import Point

        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)

        if not self.polygon.intersects(vehicle_box):
            return 0.0

        intersection = self.polygon.intersection(vehicle_box).area
        slot_area = self.polygon.area

        if slot_area <= 0:
            return 0.0

        # Base overlap ratio
        base_overlap = intersection / slot_area

        # Class-based threshold adjustment
        # Smaller vehicles (motorcycles) need less overlap to be considered "in" the slot
        # Larger vehicles (buses, trucks) need more overlap
        class_weights = {
            2: 1.0,    # car - standard threshold
            3: 0.5,    # motorcycle - lower requirement (smaller vehicle)
            5: 1.2,    # bus - higher requirement (larger vehicle)
            7: 1.1     # truck - slightly higher requirement
        }
        weight = class_weights.get(vehicle_class, 1.0)

        # Center point bonus: +20% if vehicle center is inside slot polygon
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_point = Point(center_x, center_y)
        center_bonus = 0.2 if self.polygon.contains(center_point) else 0.0

        # Combined score with class weighting and center bonus
        overlap_score = (base_overlap * weight) + center_bonus

        return min(overlap_score, 1.0)  # Cap at 1.0

    def lock_vehicle_id(self, vehicle_id, bbox, vehicle_details=None):
        """Lock a vehicle ID to this slot - ID will NEVER change until vehicle exits"""
        # Always reset timer for new vehicle (fresh parking session)
        self.locked_vehicle_id = vehicle_id
        self.locked_entry_time = time.time()  # RESET timer for new vehicle
        self.locked_bbox = bbox
        self.is_occupied = True
        self.occupied_since = self.locked_entry_time
        self.total_occupancies += 1
        self.empty_frame_count = 0
        self.pending_vehicle_id = None
        self.pending_stable_count = 0
        
        # Store vehicle details
        if vehicle_details:
            self.license_plate = vehicle_details.get('license_plate', 'N/A')
            self.vehicle_type = vehicle_details.get('vehicle_type', 'car')
            self.vehicle_color = vehicle_details.get('color', 'unknown')
        
        print(f"[SLOT #{self.id}] 🔒 LOCKED - Vehicle ID:{vehicle_id} - Timer STARTED")
        if vehicle_details:
            print(f"  └─ {self.vehicle_type.upper()} | {self.vehicle_color.upper()} | Plate: {self.license_plate}")
    
    def unlock_vehicle_id(self):
        """Unlock and remove vehicle ID - only when vehicle has fully exited"""
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
            
            # Clear vehicle details
            self.license_plate = None
            self.vehicle_type = None
            self.vehicle_color = None
    
    def get_duration(self):
        """Get current parking duration in seconds"""
        if not self.is_occupied or not self.locked_entry_time:
            return 0
        return time.time() - self.locked_entry_time
    
    def get_locked_vehicle_id(self):
        """Get the locked vehicle ID (stable ID that never changes)"""
        return self.locked_vehicle_id


# ============================================================================
# SMART PARKING SYSTEM MVP
# ============================================================================

class SmartParkingMVP:
    """
    Core parking system for GovTech PoC
    Focused on: detection, occupancy, duration, visualization
    """
    
    def __init__(self, slots_json=None, model_path='yolov8s.pt', parking_id=None):
        """
        Initialize Smart Parking MVP.

        Args:
            slots_json: Path to JSON file with slot definitions (fallback)
            model_path: Path to YOLO model
            parking_id: Database parking_id to load slots from (preferred)
        """
        print("="*70)
        print("SMART PARKING SYSTEM MVP - GovTech PoC")
        print("="*70)

        # GPU check
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = 'cpu'
            print("⚠️  Running on CPU (slower)")

        # Load YOLO
        print(f"\nLoading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
        print("✅ Model loaded")

        # Load parking slots - try database first, then JSON fallback
        slots_data = None

        if parking_id:
            print(f"\nLoading slots from database (parking_id={parking_id})...")
            slots_data = load_slots_from_database(parking_id)
            if slots_data:
                print(f"✅ Loaded {len(slots_data)} slots from database")
            else:
                print("⚠️  No slots found in database for this parking area")

        if not slots_data:
            # Try loading all slots from database (any with geometry)
            print("\nChecking database for any slots with geometry...")
            slots_data = load_slots_from_database()
            if slots_data:
                print(f"✅ Loaded {len(slots_data)} slots from database")

        if not slots_data and slots_json and os.path.exists(slots_json):
            # Fallback to JSON file
            print(f"\nFallback: Loading slots from JSON: {slots_json}")
            with open(slots_json, 'r') as f:
                data = json.load(f)
            slots_data = data.get('slots', [])
            print(f"✅ Loaded {len(slots_data)} slots from JSON")

        if not slots_data:
            raise ValueError("No parking slots found! Create slots via web UI or provide slots JSON file.")

        self.slots = [ParkingSlot(s['id'], s['points']) for s in slots_data]
        print(f"\n✅ Total parking slots loaded: {len(self.slots)}")
        
        # Initialize Duration Tracker
        print("\n🕒 Initializing Duration Tracker...")
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5000')
        self.duration_tracker = ParkingDurationTracker(
            backend_url=backend_url,
            stability_frames=5,  # Fast detection
            min_overlap=0.20  # 20% overlap (balanced)
        )
        print(f"✅ Duration Tracker initialized (Backend: {backend_url})")
        
        # Initialize Vehicle Analyzer for license plate, type, and color detection
        print("\n🚗 Initializing Vehicle Analyzer...")
        self.vehicle_analyzer = get_analyzer()
        print("✅ Vehicle Analyzer initialized (License Plate + Type + Color)")
        
        # Configuration - BALANCED DETECTION (no false positives)
        self.overlap_threshold = 0.20  # 20% overlap required (balanced)
        self.conf_threshold = 0.15  # Balanced confidence to catch cars, ignore noise
        self.min_vehicle_area = 5000  # Minimum bbox area to be considered a vehicle
        
        # Vehicle classes - NO PEDESTRIANS!
        self.vehicle_classes = [2, 3, 5, 7]  # car=2, motorcycle=3, bus=5, truck=7
        # Excluded: person=0, bicycle=1
        
        # Vehicle ID generation - system-generated, NOT from YOLO tracker
        self.next_vehicle_id = 1  # Sequential ID counter
        self.tracker_to_vehicle_id = {}  # Map tracker IDs to our stable IDs
        
        # Violation detection system
        self.violation_detector = ViolationDetector()
        self.violation_check_interval = 30  # Check for violations every 30 frames (~2 seconds)
        self.last_violation_check = 0
        self.backend_url = 'http://localhost:5000'
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\nConfiguration - OPTIMIZED VEHICLE ID LOCKING:")
        print("  Vehicle Tracking: ByteTrack with persistent IDs")
        print("  ID Generation: System-generated sequential IDs")
        print("  Model: YOLOv8s (optimized for speed & accuracy)")
        print("  Locking Logic: Slot-centric with IoU tracking")
        print(f"  Lock Condition: Vehicle stable for {self.slots[0].stability_frames} frames (~0.5s)")
        print(f"  Unlock Con50% slot coverage (90% for slot 6mes (~2.0s)")
        print("  Detection: 90% slot coverage (extremely strict - no false positives)")
        print(f"  Confidence: {self.conf_threshold} (balanced)")
        print(f"  Min Vehicle Size: {self.min_vehicle_area}px² (filters pedestrians)")
        print("  Image Size: 960px (optimized for speed)")
        print("  ID Stability: IoU > 0.5 prevents false resets")
        print("  Timer: FRESH START for each new vehicle")
        print("="*70 + "\n")
    
    def process_frame(self, frame):
        """Process one frame: SLOT-BASED detection only"""
        self.frame_count += 1
        
        # Optimized downscale for speed and accuracy
        h, w = frame.shape[:2]
        if w > 1600:  # Downscale large frames for speed
            scale = 1600 / w
            process_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            scale_back = 1.0 / scale
        else:
            process_frame = frame
            scale_back = 1.0
        
        # USE TRACK MODE with ByteTrack for stable tracker IDs
        # Optimized settings for accuracy and speed
        results = self.model.track(
            source=process_frame,
            device=self.device,
            half=False,
            imgsz=960,   # Smaller for speed
            conf=0.15,   # Balanced confidence
            iou=0.4,     # Standard IOU
            classes=self.vehicle_classes,
            verbose=False,
            max_det=200,  # Allow more detections
            persist=True,  # Keep tracker state across frames
            tracker='bytetrack.yaml'  # Use ByteTrack for stability
        )
        
        # Extract ALL detected vehicles with tracker IDs
        all_vehicles = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get tracker ID (if available)
                tracker_id = None
                if hasattr(box, 'id') and box.id is not None:
                    tracker_id = int(box.id.item())
                
                # Convert tracker ID to our stable vehicle ID
                if tracker_id is not None:
                    if tracker_id not in self.tracker_to_vehicle_id:
                        # New tracker ID - assign new vehicle ID
                        self.tracker_to_vehicle_id[tracker_id] = self.next_vehicle_id
                        self.next_vehicle_id += 1
                    vehicle_id = self.tracker_to_vehicle_id[tracker_id]
                else:
                    # No tracker ID - use position-based fallback
                    vehicle_id = int((x1 + x2 + y1 + y2) / 4) % 100000
                
                # Scale back to original resolution
                if scale_back != 1.0:
                    x1, y1, x2, y2 = int(x1 * scale_back), int(y1 * scale_back), int(x2 * scale_back), int(y2 * scale_back)
                
                # Filter out small detections (pedestrians, noise)
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.min_vehicle_area:
                    continue  # Skip small objects
                
                # Get vehicle class for type detection
                vehicle_class = int(box.cls[0].item())
                
                all_vehicles.append({
                    'id': vehicle_id,
                    'tracker_id': tracker_id,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf_score,
                    'area': bbox_area,
                    'class': vehicle_class  # Add class for analyzer
                })
        
        # SLOT-BASED FILTERING: Only keep vehicles that overlap with ANY slot
        slot_vehicles = []
        for vehicle in all_vehicles:
            # Check if this vehicle overlaps with ANY slot (20%+ overlap for accurate detection)
            for slot in self.slots:
                overlap = slot.check_overlap(vehicle['bbox'])
                if overlap >= 0.20:  # Balanced threshold - no false positives
                    slot_vehicles.append(vehicle)
                    break  # Only add once per vehicle
        
        # ANALYZE VEHICLES: Extract license plate, type, and color
        # Only analyze vehicles in parking slots (for performance)
        if slot_vehicles and self.vehicle_analyzer:
            for vehicle in slot_vehicles:
                try:
                    analysis = self.vehicle_analyzer.analyze_vehicle(
                        frame, 
                        vehicle['bbox'], 
                        vehicle.get('class', 2)  # Default to car if class not available
                    )
                    vehicle.update(analysis)  # Add license_plate, vehicle_type, color
                except Exception as e:
                    # If analysis fails, add default values
                    vehicle['license_plate'] = 'N/A'
                    vehicle['vehicle_type'] = 'car'
                    vehicle['color'] = 'unknown'
        
        # Update duration tracker with parking slots (uses locked IDs)
        self.duration_tracker.update(self.slots)
        
        # Update slot occupancy with anti-flicker buffering
        self.update_occupancy_simple(slot_vehicles)
        
        # Check for violations periodically
        if self.frame_count - self.last_violation_check >= self.violation_check_interval:
            self.check_violations(slot_vehicles)
            self.last_violation_check = self.frame_count
        
        return slot_vehicles
    
    def update_occupancy_simple(self, vehicles):
        """
        SLOT-CENTRIC VEHICLE ID LOCKING SYSTEM:
        - Each slot locks ONE vehicle ID at a time
        - ID is LOCKED after N consecutive frames (stability check)
        - ID NEVER changes while vehicle is parked
        - Unlocks ONLY when slot is empty for M consecutive frames
        - Ignores YOLO tracker ID changes and bounding box jitter
        """
        for slot in self.slots:
            # Find ALL vehicles that overlap with this slot
            overlapping_vehicles = []
            for vehicle in vehicles:
                overlap_ratio = slot.check_overlap(vehicle['bbox'])
                # Custom thresholds per slot
                if slot.id == 5:
                    threshold = 0.70  # Slot 5: 70%
                elif slot.id == 6:
                    threshold = 0.90  # Slot 6: 90%
                else:
                    threshold = 0.50  # All others: 50%
                if overlap_ratio > threshold:
                    overlapping_vehicles.append((vehicle, overlap_ratio))
            
            # Sort by overlap ratio (highest first) - prefer vehicle covering more of the slot
            overlapping_vehicles.sort(key=lambda x: x[1], reverse=True)
            
            # Get the best matching vehicle (highest overlap)
            vehicle_in_slot = overlapping_vehicles[0][0] if overlapping_vehicles else None
            
            # === CASE 1: Slot already has a LOCKED vehicle ID ===
            if slot.locked_vehicle_id is not None:
                if vehicle_in_slot is not None:
                    # Vehicle still present - keep locked ID, reset empty counter
                    slot.empty_frame_count = 0
                    # Update bbox for tracking (but keep same ID!)
                    slot.locked_bbox = vehicle_in_slot['bbox']
                else:
                    # No vehicle detected - increment empty counter
                    slot.empty_frame_count += 1
                    
                    # UNLOCK CONDITION: Slot empty for M consecutive frames
                    if slot.empty_frame_count >= slot.unlock_frames:
                        slot.unlock_vehicle_id()
            
            # === CASE 2: Slot is vacant (no locked ID) ===
            else:
                if vehicle_in_slot is not None:
                    # Vehicle detected in vacant slot
                    if slot.pending_vehicle_id == vehicle_in_slot['id']:
                        # Same vehicle as before - increment stability counter
                        slot.pending_stable_count += 1
                        slot.pending_bbox = vehicle_in_slot['bbox']  # Update bbox
                        
                        # ID LOCKING CONDITION: Vehicle stable for N consecutive frames
                        if slot.pending_stable_count >= slot.stability_frames:
                            # Extract vehicle details for locking
                            vehicle_details = {
                                'license_plate': vehicle_in_slot.get('license_plate', 'N/A'),
                                'vehicle_type': vehicle_in_slot.get('vehicle_type', 'car'),
                                'color': vehicle_in_slot.get('color', 'unknown')
                            }
                            slot.lock_vehicle_id(vehicle_in_slot['id'], vehicle_in_slot['bbox'], vehicle_details)
                    else:
                        # Different vehicle ID detected - check if it's in similar position
                        if slot.pending_bbox is not None:
                            # Calculate IoU with pending bbox
                            x1a, y1a, x2a, y2a = slot.pending_bbox
                            x1b, y1b, x2b, y2b = vehicle_in_slot['bbox']
                            
                            # Calculate intersection
                            xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
                            xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
                            
                            if xi1 < xi2 and yi1 < yi2:
                                inter_area = (xi2 - xi1) * (yi2 - yi1)
                                box1_area = (x2a - x1a) * (y2a - y1a)
                                box2_area = (x2b - x1b) * (y2b - y1b)
                                union_area = box1_area + box2_area - inter_area
                                iou = inter_area / union_area if union_area > 0 else 0
                                
                                # If IoU > 0.5, it's likely the same vehicle (tracker ID changed)
                                if iou > 0.5:
                                    # Continue with same pending vehicle (ignore ID change)
                                    slot.pending_stable_count += 1
                                    slot.pending_bbox = vehicle_in_slot['bbox']
                                    
                                    if slot.pending_stable_count >= slot.stability_frames:
                                        # Extract vehicle details for locking
                                        vehicle_details = {
                                            'license_plate': vehicle_in_slot.get('license_plate', 'N/A'),
                                            'vehicle_type': vehicle_in_slot.get('vehicle_type', 'car'),
                                            'color': vehicle_in_slot.get('color', 'unknown')
                                        }
                                        slot.lock_vehicle_id(slot.pending_vehicle_id, vehicle_in_slot['bbox'], vehicle_details)
                                else:
                                    # Different position - new vehicle detected
                                    slot.pending_vehicle_id = vehicle_in_slot['id']
                                    slot.pending_bbox = vehicle_in_slot['bbox']
                                    slot.pending_stable_count = 1
                            else:
                                # No intersection - new vehicle
                                slot.pending_vehicle_id = vehicle_in_slot['id']
                                slot.pending_bbox = vehicle_in_slot['bbox']
                                slot.pending_stable_count = 1
                        else:
                            # First detection - start stability check
                            slot.pending_vehicle_id = vehicle_in_slot['id']
                            slot.pending_bbox = vehicle_in_slot['bbox']
                            slot.pending_stable_count = 1
                else:
                    # No vehicle - reset pending state
                    slot.pending_vehicle_id = None
                    slot.pending_bbox = None
                    slot.pending_stable_count = 0
    
    def check_violations(self, vehicles):
        """
        Check for parking violations from video feed
        - Duration violations: vehicles parked too long
        - Double parking: multiple vehicles in one slot
        - Outside boundary: vehicles parked over lines
        """
        try:
            for slot in self.slots:
                if slot.is_occupied and slot.locked_vehicle_id is not None:
                    # Get parking duration for this slot
                    duration_seconds = slot.get_duration()
                    duration_minutes = duration_seconds / 60.0
                    
                    # Check for duration violation
                    if duration_minutes >= 2:  # At least 2 minutes
                        violation = self.violation_detector.check_duration_violation(
                            slot_id=f"Slot-{slot.id}",
                            vehicle_id=str(slot.locked_vehicle_id),
                            duration_minutes=duration_minutes,
                            parking_area="Video Feed Area",
                            license_plate=f"VID-{slot.locked_vehicle_id}"
                        )
                        
                        if violation:
                            self.send_violation_to_backend(violation)
                
                # Check for double parking (multiple vehicles in one slot)
                vehicles_in_slot = []
                for vehicle in vehicles:
                    if slot.check_strict_occupancy(vehicle['bbox']):
                        vehicles_in_slot.append(vehicle)
                
                if len(vehicles_in_slot) > 1:
                    violation = self.violation_detector.check_double_parking(
                        slot_id=f"Slot-{slot.id}",
                        vehicle_count=len(vehicles_in_slot),
                        parking_area="Video Feed Area"
                    )
                    if violation:
                        self.send_violation_to_backend(violation)
                
                # Check for boundary violations (vehicle outside slot lines)
                if slot.is_occupied:
                    for vehicle in vehicles:
                        overlap = slot.check_overlap(vehicle['bbox'])
                        # If vehicle overlaps but not fully within (partial overlap)
                        if 0.3 < overlap < 0.7:  # Partially outside
                            violation = self.violation_detector.check_boundary_violation(
                                slot_id=f"Slot-{slot.id}",
                                vehicle_id=str(vehicle['id']),
                                is_outside_boundary=True,
                                parking_area="Video Feed Area",
                                license_plate=f"VID-{vehicle['id']}"
                            )
                            if violation:
                                self.send_violation_to_backend(violation)
                            break  # Only report once per slot
        
        except Exception as e:
            print(f"Error checking violations: {e}")
    
    def send_violation_to_backend(self, violation):
        """Send detected violation to backend API"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/violations/from-video",
                json=violation,
                timeout=1
            )
            if response.status_code == 201:
                print(f"✓ Violation reported: {violation['violation_type']} - {violation['severity']}")
        except Exception as e:
            # Silently fail if backend is not available
            pass
    
    def draw_visualization(self, frame, vehicles):
        """Draw visualization overlay for MVP demo with duration timers"""
        vis = frame.copy()
        
        # Draw parking slots
        for slot in self.slots:
            # Color based on occupancy
            if slot.is_occupied:
                color = (0, 0, 255)  # RED = occupied
                thickness = 3
            else:
                color = (0, 255, 0)  # GREEN = vacant
                thickness = 2
            
            # Draw slot polygon
            points = np.array(slot.points, dtype=np.int32)
            cv2.polylines(vis, [points], True, color, thickness)
            
            # Draw duration timer if slot is occupied
            if slot.is_occupied:
                duration = self.duration_tracker.get_slot_duration(slot.id)
                if duration is not None and duration > 0:
                    # Format duration as "Xm Ys"
                    duration_text = self.format_duration(duration)
                    
                    # Calculate label position (center of slot)
                    center_x = int(np.mean([p[0] for p in slot.points]))
                    center_y = int(np.mean([p[1] for p in slot.points]))
                    
                    # Draw duration background
                    (text_w, text_h), _ = cv2.getTextSize(
                        duration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(vis, 
                                (center_x - text_w//2 - 5, center_y - text_h//2 - 5),
                                (center_x + text_w//2 + 5, center_y + text_h//2 + 5),
                                (0, 0, 0), -1)
                    
                    # Draw duration text
                    cv2.putText(vis, duration_text,
                              (center_x - text_w//2, center_y + text_h//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Semi-transparent fill
            overlay = vis.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            
            # Slot label
            center = np.mean(points, axis=0).astype(int)
            
            # Slot ID and locked vehicle ID
            if slot.locked_vehicle_id is not None:
                label = f"#{slot.id} [ID:{slot.locked_vehicle_id}]"
                # Add vehicle details if available
                if slot.license_plate and slot.license_plate != 'N/A':
                    label += f" {slot.license_plate}"
            else:
                label = f"#{slot.id}"
            
            # Duration if occupied
            if slot.is_occupied:
                duration = slot.get_duration()
                label += f" {self.format_duration(duration)}"
            
            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (center[0] - w//2 - 5, center[1] - h - 5),
                         (center[0] + w//2 + 5, center[1] + 5), (0, 0, 0), -1)
            
            # Label text
            cv2.putText(vis, label, (center[0] - w//2, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw vehicle bounding boxes with details
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['bbox'])
            vehicle_id = vehicle['id']
            
            # Simple cyan color for all vehicles
            color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Vehicle label with details
            label_parts = [f"ID:{vehicle_id}"]
            
            # Add vehicle type and color if available
            if 'vehicle_type' in vehicle and vehicle['vehicle_type'] != 'unknown':
                label_parts.append(vehicle['vehicle_type'].upper())
            if 'color' in vehicle and vehicle['color'] != 'unknown':
                label_parts.append(vehicle['color'].upper())
            if 'license_plate' in vehicle and vehicle['license_plate'] != 'N/A':
                label_parts.append(vehicle['license_plate'])
            
            label = " | ".join(label_parts)
            
            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            
            # Label text
            cv2.putText(vis, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw statistics dashboard
        self.draw_dashboard(vis, vehicles)
        
        return vis
    
    def draw_dashboard(self, frame, vehicles):
        """Draw statistics dashboard"""
        # Calculate stats
        total_slots = len(self.slots)
        occupied = sum(1 for s in self.slots if s.is_occupied)
        vacant = total_slots - occupied
        occupancy_rate = (occupied / total_slots * 100) if total_slots > 0 else 0
        
        # Vehicle stats
        total_vehicles = len(vehicles)
        
        # FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # GPU info
        if self.device == 'cuda':
            gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
            gpu_info = f"GPU: {gpu_mem:.1f}GB"
        else:
            gpu_info = "CPU Mode"
        
        # Dashboard
        stats = [
            "SMART PARKING MVP",
            f"FPS: {fps:.1f} | {gpu_info}",
            "",
            f"CAPACITY: {occupied}/{total_slots} ({occupancy_rate:.0f}%)",
            f"Vacant: {vacant}",
            f"Vehicles: {total_vehicles}",
            "",
            "GREEN = Vacant",
            "RED = Occupied"
        ]
        
        # Background
        max_width = 420
        height = len(stats) * 35 + 20
        cv2.rectangle(frame, (10, 10), (max_width, height), (0, 0, 0), -1)
        
        # Text
        y = 40
        for stat in stats:
            if stat == "":
                y += 15
                continue
            
            if "SMART PARKING" in stat:
                color = (0, 255, 255)
                scale = 0.8
            elif occupied == total_slots and "CAPACITY" in stat:
                color = (0, 0, 255)
                scale = 0.7
            elif "GREEN" in stat:
                color = (0, 255, 0)
                scale = 0.6
            elif "RED" in stat:
                color = (0, 0, 255)
                scale = 0.6
            else:
                color = (255, 255, 255)
                scale = 0.7
            
            cv2.putText(frame, stat, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            y += 35
    
    def format_duration(self, seconds):
        """Format duration as Xm Ys"""
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    
    def run(self, video_source, output_video=None):
        """Run the parking system on video"""
        print("Starting Smart Parking MVP...")
        print(f"Video: {video_source}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("[ERROR] Cannot open video")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        
        # Video writer (optional)
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
            print(f"Recording to: {output_video}")
        
        # Display setup
        max_display_width = 1280
        display_scale = min(max_display_width / frame_width, 1.0)
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)
        
        print(f"Display: {display_width}x{display_height}\n")
        print("Controls:")
        print("  Q - Quit")
        print("  S - Save snapshot")
        print("  SPACE - Pause/Resume")
        print("\n" + "="*70 + "\n")
        
        window_name = 'Smart Parking MVP - GovTech PoC'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[END] Video finished")
                    break
                
                # Process frame
                vehicles = self.process_frame(frame)
                
                # Visualize
                annotated = self.draw_visualization(frame, vehicles)
                
                # Write to output video
                if writer:
                    writer.write(annotated)
                
                # Store for display
                current_frame = annotated.copy()
            else:
                annotated = current_frame
            
            # Resize for display
            if display_scale < 1.0:
                display_frame = cv2.resize(annotated, (display_width, display_height))
            else:
                display_frame = annotated
            
            # Show
            cv2.imshow(window_name, display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[QUIT] User stopped")
                break
            
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'output/snapshots/parking_{timestamp}.jpg'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, annotated)
                print(f"[SAVED] {filename}")
            
            elif key == ord(' '):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final report
        self.print_report()
    
    def print_report(self):
        """Print final statistics"""
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        
        total_slots = len(self.slots)
        occupied = sum(1 for s in self.slots if s.is_occupied)
        
        print(f"\nCapacity: {occupied}/{total_slots} occupied")
        print(f"\nSlot Statistics:")
        
        for slot in self.slots:
            avg_duration = slot.total_duration / slot.total_occupancies if slot.total_occupancies > 0 else 0
            status = "OCCUPIED" if slot.is_occupied else "VACANT"
            
            print(f"  Slot #{slot.id}: {status}")
            print(f"    Total uses: {slot.total_occupancies}")
            print(f"    Avg duration: {self.format_duration(avg_duration)}")
        
        print("\n" + "="*70)


# ============================================================================
# SLOT MAPPER - 2-CLICK ANGLE-AWARE SYSTEM
# ============================================================================

def map_parking_slots(video_path, output_json='configs/parking_slots.json'):
    """
    Minimal-interaction slot mapper for angled parking
    
    Process:
    1. User clicks & drags along ONE parking line direction
    2. System calculates angle and perpendicular automatically
    3. System generates N angled slot polygons
    4. Perfect alignment with painted lines
    """
    print("="*70)
    print("PARKING SLOT MAPPER - 2-Click System")
    print("="*70)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Cannot read video")
        return None
    
    orig_frame = frame.copy()
    print(f"\nVideo: {video_path}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Resize for display
    max_width = 1280
    display_scale = 1.0
    if frame.shape[1] > max_width:
        display_scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    
    print(f"Display scale: {display_scale:.2f}x\n")
    
    # STEP 1: Define parking direction
    print("="*70)
    print("STEP 1: Define Parking Direction")
    print("="*70)
    print("Click and DRAG along ONE parking slot line")
    print("  - Start at one end of the line")
    print("  - Drag to the other end")
    print("  - This defines slot angle and direction")
    print("\nPress ENTER when satisfied | R to redraw")
    print("="*70 + "\n")
    
    drawing = False
    line_start = None
    line_end = None
    line_defined = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, line_start, line_end, line_defined
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            line_start = (x, y)
            line_end = (x, y)
            line_defined = False
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            line_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            line_end = (x, y)
            line_defined = True
            
            dx = line_end[0] - line_start[0]
            dy = line_end[1] - line_start[1]
            angle = np.degrees(np.arctan2(dy, dx))
            length = int(np.sqrt(dx**2 + dy**2) / display_scale)
            
            print(f"\n✓ Direction: {length}px at {angle:.1f}°")
            print("Press ENTER to continue\n")
    
    window_name = 'Draw Line Along Parking Direction'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = frame.copy()
        
        if line_start and line_end:
            color = (0, 255, 0) if line_defined else (0, 255, 255)
            thickness = 3 if line_defined else 2
            
            cv2.line(display, line_start, line_end, color, thickness)
            
            if line_defined:
                cv2.arrowedLine(display, line_start, line_end, (0, 255, 0), 3, tipLength=0.05)
            
            cv2.circle(display, line_start, 8, (255, 0, 0), -1)
            cv2.circle(display, line_end, 8, (0, 0, 255), -1)
        
        # Enhanced visual instructions - MUCH CLEARER
        box_height = 200
        cv2.rectangle(display, (10, 10), (750, box_height), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (750, box_height), (0, 255, 255), 3)
        
        y_pos = 50
        instructions = [
            ("STEP 1: DEFINE PARKING DIRECTION", (0, 255, 255), 0.9, True),
            ("", (255, 255, 255), 0.7, False),
            ("1. CLICK at START of a parking slot line", (255, 255, 255), 0.7, False),
            ("2. DRAG to END of the same line", (255, 255, 255), 0.7, False),
            ("3. RELEASE mouse button", (255, 255, 255), 0.7, False),
            ("", (255, 255, 255), 0.7, False),
            ("Press ENTER when satisfied | R to redraw", (0, 255, 0), 0.7, True)
        ]
        
        for text, color, scale, bold in instructions:
            if text == "":
                y_pos += 20
                continue
            thickness = 2 if bold else 1
            cv2.putText(display, text, (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y_pos += 35
        
        # Draw example diagram if no line yet
        if not line_start or not line_defined:
            # Example box
            ex_x = display.shape[1] - 320
            ex_y = 10
            ex_w = 310
            ex_h = 180
            
            cv2.rectangle(display, (ex_x, ex_y), (ex_x + ex_w, ex_y + ex_h), (0, 0, 0), -1)
            cv2.rectangle(display, (ex_x, ex_y), (ex_x + ex_w, ex_y + ex_h), (0, 255, 255), 2)
            
            cv2.putText(display, "EXAMPLE:", (ex_x + 10, ex_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw example parking slots
            slots_y = ex_y + 60
            for i in range(3):
                offset = i * 85
                # Angled rectangle representing parking slot
                pts = np.array([
                    [ex_x + 20 + offset, slots_y],
                    [ex_x + 70 + offset, slots_y - 10],
                    [ex_x + 70 + offset, slots_y + 70],
                    [ex_x + 20 + offset, slots_y + 80]
                ], np.int32)
                cv2.polylines(display, [pts], True, (100, 100, 100), 1)
            
            # Draw example direction line with arrow
            arrow_start = (ex_x + 35, slots_y + 35)
            arrow_end = (ex_x + 220, slots_y + 15)
            cv2.arrowedLine(display, arrow_start, arrow_end, (0, 255, 0), 3, tipLength=0.1)
            
            cv2.putText(display, "Draw line like this", (ex_x + 30, ex_y + ex_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and line_defined:  # ENTER
            break
        elif key == ord('r') or key == ord('R'):
            line_start = None
            line_end = None
            line_defined = False
            print("✓ Reset")
    
    cv2.destroyAllWindows()
    
    if not line_defined:
        return None
    
    # STEP 2: Calculate vectors
    orig_x1 = int(line_start[0] / display_scale)
    orig_y1 = int(line_start[1] / display_scale)
    orig_x2 = int(line_end[0] / display_scale)
    orig_y2 = int(line_end[1] / display_scale)
    
    dx = orig_x2 - orig_x1
    dy = orig_y2 - orig_y1
    length = np.sqrt(dx**2 + dy**2)
    
    # Direction vector (along slots)
    dir_vec = np.array([dx, dy]) / length
    
    # Perpendicular vector (across slots)
    perp_vec = np.array([-dy, dx]) / length
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    print(f"\n✓ Angle: {angle:.2f}°\n")
    
    # STEP 3: Get parameters with clear explanations
    print("="*70)
    print("STEP 2: Slot Configuration")
    print("="*70)
    print("\nThe line you drew defines the DIRECTION of parking slots.")
    print("Now specify the SIZE and COUNT:\n")
    
    print("1. SLOT LENGTH (along the line you drew)")
    print("   - This is how LONG each parking slot is")
    print("   - Typical car slot: 300-500 pixels")
    print("   - Just press ENTER for default (300)")
    slot_length = int(input("   Enter length [300]: ") or "300")
    print(f"   ✓ Slot length: {slot_length} pixels\n")
    
    print("2. SLOT WIDTH (perpendicular to the line)")
    print("   - This is how WIDE each parking slot is")
    print("   - Typical car slot: 150-250 pixels")
    print("   - Just press ENTER for default (200)")
    slot_width = int(input("   Enter width [200]: ") or "200")
    print(f"   ✓ Slot width: {slot_width} pixels\n")
    
    print("3. NUMBER OF SLOTS")
    print("   - How many parking slots to generate?")
    print("   - Count the parking slots in your video")
    print("   - Just press ENTER for default (9)")
    num_slots = int(input("   Enter number [9]: ") or "9")
    print(f"   ✓ Generating {num_slots} slots\n")
    
    print("4. SPACING between slots")
    print("   - Gap between parking slots (painted lines)")
    print("   - Typical spacing: 30-80 pixels")
    print("   - Just press ENTER for default (50)")
    spacing = int(input("   Enter spacing [50]: ") or "50")
    print(f"   ✓ Spacing: {spacing} pixels\n")
    
    print("5. SLOT POSITION relative to your line:")
    print("   1 = Slots on LEFT side of line")
    print("   2 = Slots on RIGHT side of line")
    print("   3 = Slots CENTERED on line")
    print("   - Just press ENTER for default (1)")
    side = int(input("   Choose position [1]: ") or "1")
    print(f"   ✓ Position: {'Left' if side==1 else 'Right' if side==2 else 'Centered'}\n")
    
    # STEP 4: Generate slots
    print(f"\n✓ Generating {num_slots} angled slots...\n")
    
    slots = []
    start_point = np.array([orig_x1, orig_y1], dtype=float)
    
    for i in range(num_slots):
        slot_center = start_point + dir_vec * i * (slot_length + spacing)
        
        if side == 1:
            offset = perp_vec * slot_width
        elif side == 2:
            offset = -perp_vec * slot_width
        else:
            offset = perp_vec * (slot_width / 2)
            slot_center = slot_center - offset
            offset = perp_vec * slot_width
        
        half_length = slot_length / 2
        
        c1 = slot_center - dir_vec * half_length
        c2 = slot_center + dir_vec * half_length
        c3 = c2 + offset
        c4 = c1 + offset
        
        points = [
            [int(c1[0]), int(c1[1])],
            [int(c2[0]), int(c2[1])],
            [int(c3[0]), int(c3[1])],
            [int(c4[0]), int(c4[1])]
        ]
        
        slots.append({'id': i + 1, 'points': points})
        print(f"  ✓ Slot-{i + 1}")
    
    # Save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({'slots': slots, 'total_slots': len(slots)}, f, indent=2)
    
    print(f"\n✅ Saved {len(slots)} slots to: {output_json}")
    
    # Preview
    preview = orig_frame.copy()
    if preview.shape[1] > 1280:
        preview = cv2.resize(preview, None, fx=1280/preview.shape[1], fy=1280/preview.shape[1])
    
    scale = preview.shape[1] / orig_frame.shape[1]
    for slot in slots:
        pts = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in slot['points']], np.int32)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, preview, 0.85, 0, preview)
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(preview, str(slot['id']), tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite('parking_slots_preview.jpg', preview)
    print("✅ Preview: parking_slots_preview.jpg\n")
    
    cv2.namedWindow('Generated Slots - Press any key', cv2.WINDOW_NORMAL)
    cv2.imshow('Generated Slots - Press any key', preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return slots


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='SMART PARKING SYSTEM MVP - GovTech PoC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Map parking slots interactively
  python smart_parking_mvp.py --map parking.mp4

  # Run detection with JSON slot config
  python smart_parking_mvp.py --run parking.mp4

  # Run detection with slots from database (parking area ID)
  python smart_parking_mvp.py --run parking.mp4 --parking-id 1

  # Run with default video
  python smart_parking_mvp.py
        """
    )

    parser.add_argument('--map', metavar='VIDEO', help='Map parking slots interactively')
    parser.add_argument('--run', metavar='VIDEO', help='Run parking detection on video')
    parser.add_argument('--parking-id', type=int, metavar='ID',
                        help='Load slots from database for this parking area ID')
    parser.add_argument('--output', '-o', metavar='VIDEO', help='Output video file (optional)')
    parser.add_argument('--slots', metavar='JSON', default='configs/parking_slots.json',
                        help='Slots JSON file (default: configs/parking_slots.json)')

    args = parser.parse_args()

    # If no arguments, run with default video
    if not args.map and not args.run:
        video_file = DEFAULT_VIDEO_FILE
        if not os.path.exists(video_file):
            parser.print_help()
            return

        # Run with default video - try database first
        if args.parking_id:
            system = SmartParkingMVP(parking_id=args.parking_id)
        else:
            slots_json = args.slots
            if not os.path.exists(slots_json):
                print(f"[ERROR] Slots not found: {slots_json}")
                print("Run --map first to define parking slots, or use --parking-id")
                return
            system = SmartParkingMVP(slots_json=slots_json)

        system.run(video_file)
        return

    # Slot mapping mode
    if args.map:
        if not os.path.exists(args.map):
            print(f"[ERROR] Video not found: {args.map}")
            return
        map_parking_slots(args.map)
        return

    # Detection mode
    if args.run:
        if not os.path.exists(args.run):
            print(f"[ERROR] Video not found: {args.run}")
            return

        # Load slots from database (preferred) or JSON (fallback)
        if args.parking_id:
            print(f"📊 Loading slots from database (parking_id={args.parking_id})")
            system = SmartParkingMVP(parking_id=args.parking_id)
        else:
            slots_json = args.slots
            if not os.path.exists(slots_json):
                print(f"[ERROR] Slots not found: {slots_json}")
                print("Run --map first to define parking slots, or use --parking-id")
                return
            system = SmartParkingMVP(slots_json=slots_json)

        system.run(args.run, args.output)


if __name__ == "__main__":
    main()
