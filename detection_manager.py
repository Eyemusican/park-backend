"""
Detection Manager - Integrated parking detection for server.py
Manages detection threads for multiple parking areas with database-driven slots.
"""

import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
from shapely.geometry import Polygon, box as shapely_box, Point
from ultralytics import YOLO
import torch
import psycopg2
import json

# Vehicle class names for display
VEHICLE_CLASS_NAMES = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# Motion detection thresholds (FRAME-BASED for reliability)
MOTION_THRESHOLD = 5.0  # Pixels - movement below this is "stopped"

# Frame-based thresholds (detection runs every 3 frames at ~30fps = ~10 updates/sec)
# So 30 frames ≈ 3 seconds, 50 frames ≈ 5 seconds
FRAMES_TO_PARKED = 30  # Frames of low motion before considered "parked" (~3 seconds)
FRAMES_TO_MOVING = 5   # Frames of high motion before considered "moving" (quick response)
POSITION_SMOOTHING_ALPHA = 0.4  # Exponential smoothing for position (0-1, lower = more smoothing)


class VehicleTracker:
    """Track vehicle motion using FRAME-BASED detection with hysteresis"""

    def __init__(self):
        self.vehicles = {}
        self.frame_count = 0

    def update(self, vehicle_id, bbox, vehicle_class, conf):
        """Update vehicle tracking state using frame counters (not time)"""
        self.frame_count += 1
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        raw_pos = (center_x, center_y)

        if vehicle_id not in self.vehicles:
            # New vehicle - initialize with frame-based tracking
            self.vehicles[vehicle_id] = {
                'bbox': bbox,
                'raw_pos': raw_pos,
                'smoothed_pos': raw_pos,
                'last_smoothed_pos': raw_pos,
                'is_stopped': False,
                'stationary_frames': 0,  # Count of consecutive stationary frames
                'moving_frames': 0,       # Count of consecutive moving frames
                'parked_since_frame': None,
                'stopped_duration': 0,    # In frames (for display: frames / 10 ≈ seconds)
                'vehicle_class': vehicle_class,
                'vehicle_type': VEHICLE_CLASS_NAMES.get(vehicle_class, 'Vehicle'),
                'conf': conf,
                'plate': None,
                'last_seen_frame': self.frame_count
            }
        else:
            v = self.vehicles[vehicle_id]

            # Apply exponential smoothing to position (reduces jitter)
            smoothed_x = POSITION_SMOOTHING_ALPHA * raw_pos[0] + (1 - POSITION_SMOOTHING_ALPHA) * v['smoothed_pos'][0]
            smoothed_y = POSITION_SMOOTHING_ALPHA * raw_pos[1] + (1 - POSITION_SMOOTHING_ALPHA) * v['smoothed_pos'][1]
            smoothed_pos = (smoothed_x, smoothed_y)

            # Calculate distance using smoothed positions
            last_smoothed = v['last_smoothed_pos']
            distance = np.sqrt((smoothed_pos[0] - last_smoothed[0])**2 +
                             (smoothed_pos[1] - last_smoothed[1])**2)

            # Frame-based hysteresis state machine
            if distance < MOTION_THRESHOLD:
                # Low motion - increment stationary counter
                v['stationary_frames'] += 1
                v['moving_frames'] = 0

                # Check if parked threshold reached
                if v['stationary_frames'] >= FRAMES_TO_PARKED:
                    if v['parked_since_frame'] is None:
                        v['parked_since_frame'] = self.frame_count
                    v['is_stopped'] = True
                    v['stopped_duration'] = self.frame_count - v['parked_since_frame']
            else:
                # High motion - increment moving counter
                v['moving_frames'] += 1

                # Only reset stopped state after enough moving frames (hysteresis)
                if v['moving_frames'] >= FRAMES_TO_MOVING:
                    v['stationary_frames'] = 0
                    v['parked_since_frame'] = None
                    v['is_stopped'] = False
                    v['stopped_duration'] = 0

            # Update tracking state
            v['last_smoothed_pos'] = smoothed_pos
            v['smoothed_pos'] = smoothed_pos
            v['raw_pos'] = raw_pos
            v['bbox'] = bbox
            v['vehicle_class'] = vehicle_class
            v['vehicle_type'] = VEHICLE_CLASS_NAMES.get(vehicle_class, 'Vehicle')
            v['conf'] = conf
            v['last_seen_frame'] = self.frame_count

        return self.vehicles[vehicle_id]

    def get_vehicle_info(self, vehicle_id):
        """Get vehicle tracking info"""
        return self.vehicles.get(vehicle_id)

    def set_plate(self, vehicle_id, plate):
        """Set license plate for a vehicle"""
        if vehicle_id in self.vehicles:
            self.vehicles[vehicle_id]['plate'] = plate

    def cleanup_old(self, max_frames=50):
        """Remove vehicles not seen for many frames"""
        to_remove = [vid for vid, v in self.vehicles.items()
                     if self.frame_count - v.get('last_seen_frame', 0) > max_frames]
        for vid in to_remove:
            del self.vehicles[vid]


class ParkingSlot:
    """Parking slot with vehicle ID locking for stable occupancy tracking"""

    def __init__(self, slot_id, slot_number, points):
        self.id = slot_id  # Database slot_id
        self.slot_number = slot_number
        self.points = points
        self.polygon = Polygon(points)

        # Vehicle ID locking system
        self.locked_vehicle_id = None
        self.locked_entry_time = None
        self.locked_bbox = None

        # Stability counters
        self.stability_frames = 8    # Frames to lock
        self.unlock_frames = 30      # Frames to unlock

        # Pending detection
        self.pending_vehicle_id = None
        self.pending_bbox = None
        self.pending_stable_count = 0

        # Empty counter
        self.empty_frame_count = 0

        # State
        self.is_occupied = False
        self.occupied_since = None

    def check_overlap(self, bbox):
        """Check overlap ratio between vehicle bbox and slot polygon"""
        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)

        if not self.polygon.intersects(vehicle_box):
            return 0.0

        intersection = self.polygon.intersection(vehicle_box).area
        slot_area = self.polygon.area

        return intersection / slot_area if slot_area > 0 else 0.0

    def check_overlap_enhanced(self, bbox, vehicle_class=2):
        """Enhanced overlap with class weights and center bonus"""
        x1, y1, x2, y2 = bbox
        vehicle_box = shapely_box(x1, y1, x2, y2)

        if not self.polygon.intersects(vehicle_box):
            return 0.0

        intersection = self.polygon.intersection(vehicle_box).area
        slot_area = self.polygon.area

        if slot_area <= 0:
            return 0.0

        base_overlap = intersection / slot_area

        # Class weights
        class_weights = {2: 1.0, 3: 0.5, 5: 1.2, 7: 1.1}
        weight = class_weights.get(vehicle_class, 1.0)

        # Center bonus
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_bonus = 0.2 if self.polygon.contains(Point(center_x, center_y)) else 0.0

        return min((base_overlap * weight) + center_bonus, 1.0)

    def lock_vehicle(self, vehicle_id, bbox):
        """Lock vehicle ID to this slot"""
        self.locked_vehicle_id = vehicle_id
        self.locked_entry_time = time.time()
        self.locked_bbox = bbox
        self.is_occupied = True
        self.occupied_since = self.locked_entry_time
        self.empty_frame_count = 0
        self.pending_vehicle_id = None
        self.pending_stable_count = 0

    def unlock_vehicle(self):
        """Unlock and clear vehicle from slot"""
        duration = 0
        if self.locked_entry_time:
            duration = time.time() - self.locked_entry_time

        self.locked_vehicle_id = None
        self.locked_entry_time = None
        self.locked_bbox = None
        self.is_occupied = False
        self.occupied_since = None
        self.empty_frame_count = 0

        return duration

    def get_duration(self):
        """Get parking duration in seconds"""
        if not self.is_occupied or not self.locked_entry_time:
            return 0
        return time.time() - self.locked_entry_time


class ParkingAreaDetector:
    """Detection handler for a single parking area"""

    def __init__(self, parking_id, model, db_config):
        self.parking_id = parking_id
        self.model = model
        self.db_config = db_config
        self.slots = []
        self.running = False
        self.thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.video_source = None
        self.parking_name = ""
        self.overlap_threshold = 0.50

        # Vehicle motion tracker
        self.vehicle_tracker = VehicleTracker()

        # Load slots from database
        self._load_slots()
        self._load_parking_info()

    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', '5432'),
            dbname=self.db_config.get('dbname', 'parking_db'),
            user=self.db_config.get('user', 'parking_user'),
            password=self.db_config.get('password', '')
        )

    def _load_parking_info(self):
        """Load parking area info from database"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute('''
                SELECT parking_name, video_source, video_source_type
                FROM parking_area
                WHERE parking_id = %s
            ''', (self.parking_id,))
            row = cur.fetchone()
            if row:
                self.parking_name = row[0]
                self.video_source = row[1]
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error loading parking info: {e}")

    def _load_slots(self):
        """Load slot polygons from database"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute('''
                SELECT slot_id, slot_number, polygon_points
                FROM parking_slots
                WHERE parking_id = %s AND polygon_points IS NOT NULL
                ORDER BY slot_number
            ''', (self.parking_id,))

            self.slots = []
            for row in cur.fetchall():
                slot_id, slot_number, polygon_points = row
                if polygon_points:
                    if isinstance(polygon_points, str):
                        polygon_points = json.loads(polygon_points)
                    self.slots.append(ParkingSlot(slot_id, slot_number, polygon_points))

            cur.close()
            conn.close()
            print(f"[Area {self.parking_id}] Loaded {len(self.slots)} slots from database")
        except Exception as e:
            print(f"Error loading slots: {e}")

    def _update_slot_in_db(self, slot, is_occupied, vehicle_id=None):
        """Update slot occupancy in database"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()

            if is_occupied and vehicle_id:
                # Check if there's already an active event for this slot
                cur.execute('''
                    SELECT event_id FROM parking_events
                    WHERE slot_id = %s AND departure_time IS NULL
                ''', (slot.id,))
                existing = cur.fetchone()

                if not existing:
                    # Create new parking event
                    cur.execute('''
                        INSERT INTO parking_events (slot_id, vehicle_id, arrival_time)
                        VALUES (%s, %s, NOW())
                    ''', (slot.id, str(vehicle_id)))
            else:
                # Close any active parking event
                cur.execute('''
                    UPDATE parking_events
                    SET departure_time = NOW()
                    WHERE slot_id = %s AND departure_time IS NULL
                ''', (slot.id,))

            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error updating slot in DB: {e}")

    def _detect_vehicles(self, frame):
        """Run YOLO detection on frame with motion tracking"""
        results = self.model(frame, conf=0.25, verbose=False)

        vehicles = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0].cpu().numpy())
                # Vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
                if cls in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id[0]) if box.id is not None else i
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Update motion tracker
                    tracker_info = self.vehicle_tracker.update(track_id, bbox, cls, conf)

                    vehicles.append({
                        'id': track_id,
                        'bbox': bbox,
                        'class': cls,
                        'conf': conf,
                        'vehicle_type': tracker_info['vehicle_type'],
                        'is_stopped': tracker_info['is_stopped'],
                        'stopped_duration': tracker_info['stopped_duration'],
                        'plate': tracker_info.get('plate')
                    })

        return vehicles

    def _update_occupancy(self, vehicles):
        """Update slot occupancy using overlap + stability frames (motion is for display only)"""
        for slot in self.slots:
            # Find ALL vehicles overlapping this slot (no motion requirement)
            overlapping = []
            for vehicle in vehicles:
                overlap = slot.check_overlap(vehicle['bbox'])
                if overlap > self.overlap_threshold:
                    overlapping.append((vehicle, overlap))

            overlapping.sort(key=lambda x: x[1], reverse=True)
            best_vehicle = overlapping[0][0] if overlapping else None

            # Case 1: Slot has locked vehicle
            if slot.locked_vehicle_id is not None:
                if best_vehicle is not None:
                    slot.empty_frame_count = 0
                    slot.locked_bbox = best_vehicle['bbox']
                    # Store vehicle info for display
                    slot.vehicle_type = best_vehicle.get('vehicle_type', 'Vehicle')
                    slot.vehicle_plate = best_vehicle.get('plate')
                else:
                    slot.empty_frame_count += 1
                    if slot.empty_frame_count >= slot.unlock_frames:
                        duration = slot.unlock_vehicle()
                        self._update_slot_in_db(slot, False)
                        print(f"[{self.parking_name}] Slot {slot.slot_number}: VACANT (was {duration:.1f}s)")

            # Case 2: Slot is vacant - use overlap + stability frames (original reliable approach)
            else:
                if best_vehicle is not None:
                    # Track pending vehicle using stability frames (no motion requirement)
                    if slot.pending_vehicle_id == best_vehicle['id']:
                        slot.pending_stable_count += 1
                        slot.pending_bbox = best_vehicle['bbox']

                        # Lock after enough stable frames
                        if slot.pending_stable_count >= slot.stability_frames:
                            slot.lock_vehicle(best_vehicle['id'], best_vehicle['bbox'])
                            slot.vehicle_type = best_vehicle.get('vehicle_type', 'Vehicle')
                            slot.vehicle_plate = best_vehicle.get('plate')
                            self._update_slot_in_db(slot, True, best_vehicle['id'])
                            print(f"[{self.parking_name}] Slot {slot.slot_number}: OCCUPIED ({best_vehicle.get('vehicle_type', 'Vehicle')} ID:{best_vehicle['id']})")
                    else:
                        # New vehicle or ID changed - check IoU for tracker ID changes
                        if slot.pending_bbox is not None:
                            iou = self._calc_iou(slot.pending_bbox, best_vehicle['bbox'])
                            if iou > 0.5:
                                # Same vehicle, tracker ID changed - continue counting
                                slot.pending_stable_count += 1
                                slot.pending_bbox = best_vehicle['bbox']
                                if slot.pending_stable_count >= slot.stability_frames:
                                    slot.lock_vehicle(slot.pending_vehicle_id, best_vehicle['bbox'])
                                    slot.vehicle_type = best_vehicle.get('vehicle_type', 'Vehicle')
                                    slot.vehicle_plate = best_vehicle.get('plate')
                                    self._update_slot_in_db(slot, True, slot.pending_vehicle_id)
                                    print(f"[{self.parking_name}] Slot {slot.slot_number}: OCCUPIED ({best_vehicle.get('vehicle_type', 'Vehicle')})")
                            else:
                                # Different vehicle - reset
                                slot.pending_vehicle_id = best_vehicle['id']
                                slot.pending_bbox = best_vehicle['bbox']
                                slot.pending_stable_count = 1
                        else:
                            # First detection
                            slot.pending_vehicle_id = best_vehicle['id']
                            slot.pending_bbox = best_vehicle['bbox']
                            slot.pending_stable_count = 1
                else:
                    # No vehicle overlapping - reset pending
                    slot.pending_vehicle_id = None
                    slot.pending_bbox = None
                    slot.pending_stable_count = 0

    def _calc_iou(self, box1, box2):
        """Calculate IoU between two bboxes"""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2

        xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
        xi2, yi2 = min(x2a, x2b), min(y2a, y2b)

        if xi1 >= xi2 or yi1 >= yi2:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2a - x1a) * (y2a - y1a)
        box2_area = (x2b - x1b) * (y2b - y1b)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _draw_frame(self, frame, vehicles):
        """Draw visualization on frame with translucent boxes and vehicle info"""
        vis = frame.copy()
        overlay = frame.copy()

        # Draw slots with translucent fill
        for slot in self.slots:
            points = np.array(slot.points, dtype=np.int32)

            if slot.is_occupied:
                # Translucent red fill for occupied
                cv2.fillPoly(overlay, [points], (0, 0, 180))
                border_color = (0, 0, 255)  # Red border
            else:
                # Translucent green fill for vacant
                cv2.fillPoly(overlay, [points], (0, 180, 0))
                border_color = (0, 255, 0)  # Green border

            cv2.polylines(vis, [points], True, border_color, 2)

            # Slot label with vehicle info (LARGER FONT)
            centroid = np.mean(points, axis=0).astype(int)
            if slot.is_occupied:
                duration = slot.get_duration()
                vehicle_type = getattr(slot, 'vehicle_type', 'Vehicle')
                plate = getattr(slot, 'vehicle_plate', None)
                label = f"#{slot.slot_number} {vehicle_type}"
                if plate:
                    label += f" [{plate}]"
                label += f" {int(duration)}s"
            else:
                label = f"#{slot.slot_number} FREE"
            cv2.putText(vis, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Blend overlay with original (translucent effect)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Draw vehicle bboxes with translucent fill and info
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            is_stopped = vehicle.get('is_stopped', False)
            stopped_duration = vehicle.get('stopped_duration', 0)
            vehicle_type = vehicle.get('vehicle_type', 'Vehicle')
            plate = vehicle.get('plate')

            # Color based on motion state
            if is_stopped:
                # Parked - Orange/Yellow box
                box_color = (0, 165, 255)  # Orange
                fill_color = (0, 100, 150)
            else:
                # Moving - Blue box
                box_color = (255, 150, 0)  # Light blue
                fill_color = (150, 100, 0)

            # Draw translucent bounding box
            bbox_overlay = vis.copy()
            cv2.rectangle(bbox_overlay, (x1, y1), (x2, y2), fill_color, -1)
            cv2.addWeighted(bbox_overlay, 0.3, vis, 0.7, 0, vis)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            # Vehicle info label (convert frames to seconds: ~10 frames/sec)
            stopped_seconds = stopped_duration / 10.0
            label_parts = [f"{vehicle_type}"]
            if plate:
                label_parts.append(f"[{plate}]")
            if is_stopped:
                label_parts.append(f"PARKED {stopped_seconds:.0f}s")
            else:
                label_parts.append("MOVING")

            label = " ".join(label_parts)

            # Draw label background (LARGER FONT)
            font_scale = 0.8
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis, (x1, y1 - label_h - 12), (x1 + label_w + 8, y1), box_color, -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Info overlay at top (LARGER FONT)
        occupied = sum(1 for s in self.slots if s.is_occupied)
        total = len(self.slots)
        info = f"{self.parking_name} | {occupied}/{total} occupied"
        # Draw info background
        cv2.rectangle(vis, (5, 5), (550, 50), (0, 0, 0), -1)
        cv2.putText(vis, info, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return vis

    def _detection_loop(self):
        """Main detection loop"""
        if not self.video_source:
            print(f"[Area {self.parking_id}] No video source configured")
            self.running = False
            return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[Area {self.parking_id}] Failed to open: {self.video_source}")
            self.running = False
            return

        print(f"[Area {self.parking_id}] Detection started: {self.video_source}")
        frame_count = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                # Loop video files
                if isinstance(self.video_source, str) and self.video_source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1

            # Detect every 3 frames
            if frame_count % 3 == 0:
                vehicles = self._detect_vehicles(frame)
                self._update_occupancy(vehicles)
                vis_frame = self._draw_frame(frame, vehicles)
            else:
                vis_frame = self._draw_frame(frame, [])

            with self.frame_lock:
                self.current_frame = vis_frame

            time.sleep(0.03)  # ~30 FPS

        cap.release()
        print(f"[Area {self.parking_id}] Detection stopped")
        self.running = False

    def start(self):
        """Start detection thread"""
        if self.running:
            return False

        if not self.slots:
            print(f"[Area {self.parking_id}] No slots configured")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.thread = None

    def get_frame(self):
        """Get current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_status(self):
        """Get detection status"""
        return {
            'parking_id': self.parking_id,
            'parking_name': self.parking_name,
            'running': self.running,
            'slot_count': len(self.slots),
            'occupied_count': sum(1 for s in self.slots if s.is_occupied),
            'video_source': self.video_source,
            'slots': [
                {
                    'slot_number': s.slot_number,
                    'is_occupied': s.is_occupied,
                    'duration': s.get_duration() if s.is_occupied else 0,
                    'vehicle_id': s.locked_vehicle_id
                }
                for s in self.slots
            ]
        }


class DetectionManager:
    """Manages detection for multiple parking areas"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.detectors = {}  # parking_id -> ParkingAreaDetector
        self.model = None
        self.db_config = {
            'host': os.environ.get('DB_HOST', '127.0.0.1'),
            'port': os.environ.get('DB_PORT', '5433'),  # Docker default is 5433
            'dbname': os.environ.get('DB_NAME', 'parking_db'),
            'user': os.environ.get('DB_USER', 'parking_user'),
            'password': os.environ.get('DB_PASS', '')
        }
        print(f"[DetectionManager] DB config: {self.db_config['host']}:{self.db_config['port']}")
        self._initialized = True

    def _ensure_model(self):
        """Load YOLO model if not loaded"""
        if self.model is None:
            model_path = 'yolov8s.pt'
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  Using CPU")

    def start_detection(self, parking_id):
        """Start detection for a parking area"""
        self._ensure_model()

        if parking_id in self.detectors:
            if self.detectors[parking_id].running:
                return {'status': 'already_running'}
            # Restart existing detector
            self.detectors[parking_id].start()
        else:
            # Create new detector
            detector = ParkingAreaDetector(parking_id, self.model, self.db_config)
            if detector.start():
                self.detectors[parking_id] = detector
            else:
                return {'status': 'failed', 'error': 'Could not start detection'}

        return {'status': 'started', 'parking_id': parking_id}

    def stop_detection(self, parking_id):
        """Stop detection for a parking area"""
        if parking_id not in self.detectors:
            return {'status': 'not_found'}

        self.detectors[parking_id].stop()
        return {'status': 'stopped', 'parking_id': parking_id}

    def get_status(self, parking_id=None):
        """Get detection status"""
        if parking_id is not None:
            if parking_id not in self.detectors:
                return {'status': 'not_found'}
            return self.detectors[parking_id].get_status()

        # Return all statuses
        return {
            'active_detections': len([d for d in self.detectors.values() if d.running]),
            'areas': [d.get_status() for d in self.detectors.values()]
        }

    def get_frame(self, parking_id):
        """Get current frame for a parking area"""
        if parking_id not in self.detectors:
            return None
        return self.detectors[parking_id].get_frame()

    def stop_all(self):
        """Stop all detection threads"""
        for detector in self.detectors.values():
            detector.stop()
        self.detectors.clear()


def get_detection_manager():
    """Get singleton instance"""
    return DetectionManager()
