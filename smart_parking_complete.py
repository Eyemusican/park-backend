"""
Smart Parking MVP - Complete System
Integrated vehicle tracking + parking slot occupancy detection
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime
import time
from shapely.geometry import Polygon, box as shapely_box


# ============================================================================
# VEHICLE INFO CLASS
# ============================================================================

class VehicleInfo:
    """Information about a tracked vehicle"""
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # Position tracking
        self.positions = []  # List of (x, y) centers
        self.current_box = None  # (x1, y1, x2, y2)
        
        # Motion analysis
        self.is_moving = True
        self.stopped_time = None
        
        # Parking analysis
        self.parked_slot_id = None
        self.parking_start = None
        self.parking_duration = 0
        
        # Vehicle info
        self.vehicle_class = 'vehicle'
        self.confidence = 0.0
    
    def update_position(self, center, box, vehicle_class, confidence):
        """Update position and analyze motion with smoothing to prevent flickering"""
        self.last_seen = time.time()
        self.current_box = box
        self.vehicle_class = vehicle_class
        self.confidence = confidence
        
        # Add to position history (increased buffer for stability)
        self.positions.append(center)
        if len(self.positions) > 50:  # Increased from 30 to 50 for better history
            self.positions.pop(0)
        
        # ANTI-FLICKER motion detection with smaller sample for quicker response
        if len(self.positions) >= 10:  # Reduced from 20 to 10 for faster initial detection
            # Check last 10 frames for movement (faster response)
            recent = self.positions[-10:]
            
            # Calculate total displacement from start to end
            total_movement = np.sqrt(
                (recent[-1][0] - recent[0][0])**2 + 
                (recent[-1][1] - recent[0][1])**2
            )
            
            # Lower threshold: 3 pixels for quicker motion detection at start
            if total_movement > 3.0:
                self.is_moving = True
                self.stopped_time = None
            else:
                # Only log when first stopping (reduce console spam)
                if self.is_moving:
                    self.stopped_time = time.time()
                    print(f"[MOTION] ID {self.track_id}: STOPPED (movement={total_movement:.1f}px)")
                self.is_moving = False
        else:
            # Default to not moving until we have data
            self.is_moving = False
    
    def get_stopped_duration(self):
        """Get how long vehicle has been stopped"""
        if self.stopped_time is None:
            return 0
        return time.time() - self.stopped_time
    
    def get_parking_duration(self):
        """Get how long vehicle has been parked"""
        if self.parking_start is None:
            return 0
        return time.time() - self.parking_start


# ============================================================================
# PARKING SLOT CLASS
# ============================================================================

class ParkingSlot:
    """A parking slot polygon"""
    
    def __init__(self, slot_data):
        self.id = slot_data['id']
        self.name = slot_data.get('name', f'Slot {slot_data["id"]}')  # Use default name if not provided
        self.points = slot_data['points']
        
        # Create Shapely polygon
        self.polygon = Polygon([(p[0], p[1]) for p in self.points])
                # Custom occupancy thresholds for specific slots
        if self.id == 6:
            self.occupancy_threshold = 0.6  # 60% for slot 6
        elif self.id == 7:
            self.occupancy_threshold = 0.8  # 80% for slot 7
        else:
            self.occupancy_threshold = 0.5  # 50% default for all other slots
                # Occupancy tracking
        self.is_occupied = False
        self.occupied_by = None  # Vehicle track_id
        self.occupied_since = None
        self.last_vacant = time.time()
        
        # Statistics
        self.total_occupancies = 0
        self.total_duration = 0
    
    def check_occupancy(self, vehicle_box):
        """
        Check if vehicle overlaps with this slot
        Returns IoU (intersection over union)
        """
        # Create vehicle box polygon
        x1, y1, x2, y2 = vehicle_box
        vehicle_poly = shapely_box(x1, y1, x2, y2)
        
        # Calculate intersection
        if not self.polygon.intersects(vehicle_poly):
            return 0.0
        
        intersection = self.polygon.intersection(vehicle_poly).area
        vehicle_area = vehicle_poly.area
        
        # Return intersection ratio (what % of vehicle is in slot)
        if vehicle_area > 0:
            return intersection / vehicle_area
        return 0.0
    
    def mark_occupied(self, vehicle_id):
        """Mark slot as occupied"""
        # Reset vacant grace period when marking occupied
        if hasattr(self, '_vacant_grace_start'):
            self._vacant_grace_start = None
        
        if not self.is_occupied:
            self.is_occupied = True
            self.occupied_by = vehicle_id
            self.occupied_since = time.time()
            self.total_occupancies += 1
        else:
            # Update vehicle ID if changed
            self.occupied_by = vehicle_id
    
    def mark_vacant(self):
        """Mark slot as vacant with grace period to prevent flickering"""
        if self.is_occupied:
            # Only mark vacant if slot has been empty for 2+ seconds (prevents flickering)
            if not hasattr(self, '_vacant_grace_start'):
                self._vacant_grace_start = time.time()
            
            # Check if grace period has passed
            if time.time() - self._vacant_grace_start < 2.0:
                # Keep occupied during grace period
                return
            
            # Update statistics
            if self.occupied_since:
                duration = time.time() - self.occupied_since
                self.total_duration += duration
            
            self.is_occupied = False
            self.occupied_by = None
            self.occupied_since = None
            self.last_vacant = time.time()
            self._vacant_grace_start = None
        else:
            # Reset grace period when not occupied
            if hasattr(self, '_vacant_grace_start'):
                self._vacant_grace_start = None
    
    def get_occupancy_duration(self):
        """Get current occupancy duration"""
        if not self.is_occupied or self.occupied_since is None:
            return 0
        return time.time() - self.occupied_since


# ============================================================================
# SMART PARKING SYSTEM
# ============================================================================

class SmartParkingSystem:
    """Complete parking management system"""
    
    def __init__(self, slots_json, model_path='yolov8m.pt'):
        print("="*60)
        print("SMART PARKING SYSTEM - INITIALIZATION")
        print("="*60)
        
        # Load YOLO model with GPU verification
        print(f"Loading model: {model_path}")
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] Using: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] No GPU detected - performance will be slow")
        
        self.model = YOLO(model_path)
        print("[OK] Model loaded")
        
        # Load parking slots
        print(f"Loading slots: {slots_json}")
        with open(slots_json, 'r') as f:
            slots_data = json.load(f)
        
        self.slots = [ParkingSlot(slot) for slot in slots_data['slots']]
        print(f"[OK] Loaded {len(self.slots)} parking slots")
        
        # Vehicle tracking
        self.vehicles = {}  # track_id -> VehicleInfo
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Configuration
        self.occupancy_threshold = 0.5  # 50% overlap to consider occupied
        self.parking_time_threshold = 30.0  # 30 SECONDS before PARKED status (industry standard)
        self.min_confidence = 0.25  # Minimum confidence for detection (lowered for immediate detection at start)
        
        # Violation tracking
        self.violations = []  # List of violation records
        self.double_parking_threshold = 0.3  # 30% overlap for double parking
        
        # Display settings
        self.max_display_width = 1280
        self.max_display_height = 720
        self.display_scale = 1.0
        
        # Video speed control
        self.frame_skip = 0  # Skip 0 frames = process all frames for smooth playback
        self.video_speed_multiplier = 2  # Normal speed (1 = normal, 2 = 2x faster, etc.)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        print("="*60 + "\n")
    
    def update_slot_occupancy(self):
        """Update all slot occupancy based on current vehicles"""
        # First, mark all slots as potentially vacant
        slot_occupants = {slot.id: [] for slot in self.slots}
        
        # Clear old violations (keep only last 100)
        if len(self.violations) > 100:
            self.violations = self.violations[-100:]
        
        # Check ALL stopped vehicles against slots (not just 30+ second ones)
        for track_id, vehicle in self.vehicles.items():
            if vehicle.current_box is None:
                continue
            
            # Check stopped vehicles immediately (don't wait 30 seconds)
            if not vehicle.is_moving:
                # Check against all slots to find best match
                best_slot = None
                best_iou = 0.0
                
                for slot in self.slots:
                    iou = slot.check_occupancy(vehicle.current_box)
                    # Use per-slot occupancy threshold (slot 6=60%, slot 7=80%, others=50%)
                    if iou > slot.occupancy_threshold and iou > best_iou:
                        best_iou = iou
                        best_slot = slot
                
                # If vehicle is in a slot
                if best_slot:
                    slot_occupants[best_slot.id].append((track_id, best_iou))
                    
                    # Assign slot immediately when stopped
                    if vehicle.parked_slot_id != best_slot.id:
                        vehicle.parked_slot_id = best_slot.id
                        vehicle.parking_start = time.time()
                        print(f"[SLOT] ID {track_id}: Assigned to Slot #{best_slot.id} (IoU={best_iou:.2f})")
                    
                    # Mark as PARKED only after 30 seconds
                    if vehicle.get_stopped_duration() >= self.parking_time_threshold:
                        if not hasattr(vehicle, '_parked_logged') or not vehicle._parked_logged:
                            print(f"[PARKED] ID {track_id}: NOW PARKED in Slot #{best_slot.id} (30+ seconds)")
                            vehicle._parked_logged = True
                else:
                    # Vehicle stopped but not in any slot
                    vehicle.parked_slot_id = None
                    # Only log violation if stopped for 30+ seconds outside slot
                    if vehicle.get_stopped_duration() > self.parking_time_threshold:
                        self._log_violation(track_id, 'OUTSIDE_SLOT', vehicle)
            else:
                # Moving vehicle - clear parking info
                vehicle.parked_slot_id = None
                vehicle.parking_start = None
                if hasattr(vehicle, '_parked_logged'):
                    vehicle._parked_logged = False
        
        # Update slot occupancy states and detect double parking
        for slot in self.slots:
            if len(slot_occupants[slot.id]) > 1:
                # VIOLATION: DOUBLE PARKING - multiple vehicles in same slot
                for track_id, iou in slot_occupants[slot.id]:
                    self._log_violation(track_id, 'DOUBLE_PARKING', self.vehicles[track_id])
                # Mark slot as occupied by first vehicle
                slot.mark_occupied(slot_occupants[slot.id][0][0])
            elif len(slot_occupants[slot.id]) == 1:
                # Normal occupancy
                slot.mark_occupied(slot_occupants[slot.id][0][0])
            else:
                slot.mark_vacant()
    
    def _log_violation(self, track_id, violation_type, vehicle):
        """Log a parking violation"""
        # Check if already logged recently (avoid spam)
        for v in self.violations[-10:]:
            if v['track_id'] == track_id and v['type'] == violation_type:
                if time.time() - v['time'] < 5.0:  # Don't log same violation within 5 seconds
                    return
        
        violation = {
            'track_id': track_id,
            'type': violation_type,
            'time': time.time(),
            'vehicle_class': vehicle.vehicle_class,
            'duration': vehicle.get_stopped_duration()
        }
        self.violations.append(violation)
        print(f"[VIOLATION] ID:{track_id} {violation_type} Duration:{violation['duration']:.1f}s")
    
    def process_frame(self, frame):
        """Process a frame and return annotated result"""
        self.frame_count += 1
        
        # Auto-detect device (CPU or CUDA)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_half = True if torch.cuda.is_available() else False
        
        # Run YOLO tracking - LOWER CONFIDENCE for immediate detection at start
        results = self.model.track(
            source=frame,
            persist=True,
            conf=0.25,  # Lower confidence to detect all vehicles immediately at start
            iou=0.65,  # Balanced IoU
            classes=[2, 3, 5, 7],  # Vehicle classes
            verbose=False,
            device=device,  # Auto-detect device
            half=use_half,  # Use FP16 only if CUDA available
            imgsz=960,  # Smaller image size for faster processing
            tracker='bytetrack.yaml'  # Use custom tracker config
        )
        
        # Update vehicles from detections
        current_ids = set()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                
                track_id = int(box.id.item())
                current_ids.add(track_id)
                
                # Get box info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                vehicle_class = self.vehicle_classes.get(cls, 'vehicle')
                
                # Calculate center
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Update or create vehicle
                if track_id not in self.vehicles:
                    self.vehicles[track_id] = VehicleInfo(track_id)
                
                self.vehicles[track_id].update_position(
                    center, (x1, y1, x2, y2), vehicle_class, conf
                )
        
        # Remove vehicles not seen for >3 seconds (balance between stability and responsiveness)
        to_remove = []
        for track_id, vehicle in self.vehicles.items():
            if track_id not in current_ids:
                if time.time() - vehicle.last_seen > 3.0:  # 3 seconds for balance
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.vehicles[track_id]
        
        # Update slot occupancy
        self.update_slot_occupancy()
        
        # Debug: Print vehicle states every 30 frames (~1 second)
        if self.frame_count % 30 == 0 and len(self.vehicles) > 0:
            print(f"\n[DEBUG Frame {self.frame_count}] Vehicle States:")
            for tid, v in self.vehicles.items():
                status = "PARKED" if v.parked_slot_id else ("STOPPED" if not v.is_moving else "MOVING")
                stopped = f"{v.get_stopped_duration():.1f}s" if v.stopped_time else "N/A"
                if status == "STOPPED" and v.stopped_time:
                    remaining = max(0, self.parking_time_threshold - v.get_stopped_duration())
                    print(f"  ID {tid}: {status} | Stopped: {stopped} | Parking in: {remaining:.1f}s | Slot: {v.parked_slot_id}")
                else:
                    print(f"  ID {tid}: {status} | Stopped: {stopped} | Slot: {v.parked_slot_id}")
        
        # Draw visualization
        annotated = self.draw_visualization(frame)
        
        return annotated
    
    def draw_visualization(self, frame):
        """Draw complete visualization"""
        vis = frame.copy()
        
        # 1. Draw parking slots
        for slot in self.slots:
            # Color based on occupancy
            if slot.is_occupied:
                color = (0, 0, 255)  # RED - occupied
                thickness = 3
            else:
                color = (0, 255, 0)  # GREEN - vacant
                thickness = 2
            
            # Draw polygon
            points = np.array(slot.points, dtype=np.int32)
            cv2.polylines(vis, [points], True, color, thickness)
            
            # Fill with transparency
            overlay = vis.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            
            # Draw slot ID
            center_x = int(np.mean([p[0] for p in slot.points]))
            center_y = int(np.mean([p[1] for p in slot.points]))
            
            # Slot label
            label = f"#{slot.id}"
            if slot.is_occupied:
                duration = slot.get_occupancy_duration()
                label += f" [{self.format_duration(duration)}]"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(vis, (center_x - label_w//2 - 5, center_y - label_h - 5),
                         (center_x + label_w//2 + 5, center_y + 5),
                         (0, 0, 0), -1)
            
            # Label text
            cv2.putText(vis, label, (center_x - label_w//2, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 2. Draw vehicles
        for track_id, vehicle in self.vehicles.items():
            if vehicle.current_box is None:
                continue
            
            x1, y1, x2, y2 = vehicle.current_box
            
            # Check if vehicle has active violation
            has_violation = False
            violation_type = ""
            for v in self.violations[-20:]:
                if v['track_id'] == track_id and time.time() - v['time'] < 10.0:
                    has_violation = True
                    violation_type = v['type']
                    break
            
            # Color based on status with 30-second rule
            if has_violation:
                color = (0, 165, 255)  # ORANGE - violation
                status = f"VIOLATION:{violation_type.replace('_', ' ')}"
            elif vehicle.parked_slot_id is not None and vehicle.get_stopped_duration() >= self.parking_time_threshold:
                color = (0, 0, 255)  # RED - parked 30+ seconds
                status = "PARKED"
            elif vehicle.parked_slot_id is not None:
                color = (0, 255, 255)  # YELLOW - in slot but <30 seconds
                status = "STOPPED"
            elif not vehicle.is_moving:
                color = (0, 255, 255)  # YELLOW - stopped outside slot
                status = "STOPPED"
            else:
                color = (0, 255, 0)  # GREEN - moving
                status = "MOVING"
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw trajectory
            if len(vehicle.positions) > 1:
                points = vehicle.positions[-20:]  # Last 20 points
                for i in range(1, len(points)):
                    cv2.line(vis, points[i-1], points[i], (255, 0, 255), 2)
            
            # Vehicle label - LARGER TEXT with countdown to PARKED
            label = f"ID:{track_id} {status}"
            if vehicle.parked_slot_id:
                label += f" Slot#{vehicle.parked_slot_id}"
                parking_duration = vehicle.get_parking_duration()
                label += f" {self.format_duration(parking_duration)}"
            elif not vehicle.is_moving and vehicle.stopped_time:
                # Show countdown for STOPPED vehicles (30s until PARKED)
                stopped_duration = vehicle.get_stopped_duration()
                remaining = max(0, self.parking_time_threshold - stopped_duration)
                label += f" [{self.format_duration(stopped_duration)}/{int(self.parking_time_threshold)}s]"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            # Label background
            cv2.rectangle(vis, (x1, y1 - label_h - 15),
                         (x1 + label_w + 10, y1), color, -1)
            
            # Label text - BIGGER
            cv2.putText(vis, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 3. Draw statistics dashboard
        self.draw_statistics(vis)
        
        return vis
    
    def draw_statistics(self, frame):
        """Draw statistics dashboard"""
        # Calculate stats
        total_capacity = len(self.slots)
        occupied_count = sum(1 for slot in self.slots if slot.is_occupied)
        vacant_count = total_capacity - occupied_count
        occupancy_rate = (occupied_count / total_capacity * 100) if total_capacity > 0 else 0
        
        # Vehicle stats
        total_vehicles = len(self.vehicles)
        moving_vehicles = sum(1 for v in self.vehicles.values() if v.is_moving)
        stopped_vehicles = sum(1 for v in self.vehicles.values() if not v.is_moving and v.parked_slot_id is None)
        parked_vehicles = sum(1 for v in self.vehicles.values() if v.parked_slot_id is not None)
        
        # Count active violations (within last 10 seconds)
        current_time = time.time()
        active_violations = len([v for v in self.violations if current_time - v['time'] < 10.0])
        violations = active_violations
        
        # FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Build stats text - BIGGER AND CLEARER
        stats = [
            "SMART PARKING SYSTEM",
            f"FPS: {fps:.1f} | Frame: {self.frame_count}",
            "",
            "PARKING CAPACITY:",
            f"  Occupied: {occupied_count}/{total_capacity} ({occupancy_rate:.0f}%)",
            f"  Vacant: {vacant_count}",
            "",
            "VEHICLES:",
            f"  Tracked: {total_vehicles}",
            f"  Moving: {moving_vehicles}",
            f"  Parked: {parked_vehicles}",
            f"  Violations: {violations}",
            "",
            "COLOR CODE:",
            "  GREEN = Vacant/Moving",
            "  RED = Occupied/Parked",
            "  YELLOW = Stopped",
            "  ORANGE = Violation"
        ]
        
        # Stats background - BIGGER
        max_width = 450
        height = len(stats) * 38 + 30
        cv2.rectangle(frame, (10, 10), (max_width, height),
                     (0, 0, 0), -1)
        
        # Stats text - LARGER FONT
        y_offset = 45
        for stat in stats:
            if stat == "":
                y_offset += 20
                continue
            
            # Color based on content
            if "SMART PARKING" in stat:
                color = (0, 255, 255)  # YELLOW
                font_scale = 0.9
            elif "COLOR CODE" in stat:
                color = (0, 255, 255)  # YELLOW
                font_scale = 0.7
            elif "GREEN" in stat:
                color = (0, 255, 0)  # GREEN
                font_scale = 0.6
            elif "RED" in stat:
                color = (0, 0, 255)  # RED
                font_scale = 0.6
            elif "YELLOW" in stat:
                color = (0, 255, 255)  # YELLOW
                font_scale = 0.6
            elif occupied_count == total_capacity and "Occupied:" in stat:
                color = (0, 0, 255)  # RED - full
                font_scale = 0.7
            elif violations > 0 and "Violations:" in stat:
                color = (0, 0, 255)  # RED
                font_scale = 0.7
            else:
                color = (255, 255, 255)  # WHITE
                font_scale = 0.7
            
            cv2.putText(frame, stat, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            y_offset += 38
    
    def format_duration(self, seconds):
        """Format duration as 1m30s"""
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    
    def run(self, video_source):
        """Run the parking system on a video"""
        print("Starting smart parking system...")
        print(f"Video: {video_source}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("[ERROR] Cannot open video")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate display scale
        scale_w = self.max_display_width / frame_width
        scale_h = self.max_display_height / frame_height
        self.display_scale = min(scale_w, scale_h, 1.0)
        
        display_width = int(frame_width * self.display_scale)
        display_height = int(frame_height * self.display_scale)
        
        print(f"Original size: {frame_width}x{frame_height}")
        print(f"Display size: {display_width}x{display_height}")
        print(f"Scale: {self.display_scale:.2f}x\n")
        
        print("Press:")
        print("  Q - Quit")
        print("  S - Save snapshot")
        print("  R - Reset statistics")
        print("\n" + "="*60 + "\n")
        
        window_name = 'Smart Parking System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[END] Video finished")
                break
            
            # Frame skipping for faster processing
            frame_counter += 1
            if self.frame_skip > 0 and frame_counter % (self.frame_skip + 1) != 0:
                continue
            
            # Process frame
            annotated = self.process_frame(frame)
            
            # Resize for display if needed
            if self.display_scale < 1.0:
                annotated = cv2.resize(annotated, (display_width, display_height))
            
            # Display
            cv2.imshow(window_name, annotated)
            
            # Handle keys - FASTER VIDEO with speed multiplier
            # Lower waitKey value = faster video (1 = very fast, 25-30 = normal speed)
            wait_time = max(1, int(25 / self.video_speed_multiplier))  # 25ms is normal, divide by speed multiplier
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                print("\n[QUIT] User stopped")
                break
            
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'output/snapshots/parking_{timestamp}.jpg'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, annotated)
                print(f"[SAVED] Snapshot: {filename}")
            
            elif key == ord('r'):
                # Reset statistics
                self.frame_count = 0
                self.start_time = time.time()
                print("[RESET] Statistics reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print final statistics report"""
        print("\n" + "="*60)
        print("FINAL PARKING REPORT")
        print("="*60)
        
        # Slot statistics
        print(f"\nParking Slots: {len(self.slots)}")
        for slot in self.slots:
            avg_duration = slot.total_duration / slot.total_occupancies if slot.total_occupancies > 0 else 0
            print(f"  {slot.name}:")
            print(f"    Total uses: {slot.total_occupancies}")
            print(f"    Avg duration: {self.format_duration(avg_duration)}")
            print(f"    Current: {'OCCUPIED' if slot.is_occupied else 'VACANT'}")
        
        # Vehicle statistics
        print(f"\nVehicles Tracked: {len(self.vehicles)}")
        
        print("\n" + "="*60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    # Get slots JSON
    if len(sys.argv) > 1:
        slots_json = sys.argv[1]
    else:
        slots_json = 'configs/parking_slots.json'
    
    # Check if slots exist
    if not os.path.exists(slots_json):
        print("[ERROR] Parking slots not found!")
        print(f"Expected: {slots_json}")
        print("\nPlease run slot_mapper.py first to define parking slots:")
        print("  python slot_mapper.py parking_video.mp4.mp4")
        return
    
    # Get video source
    if len(sys.argv) > 2:
        video_source = sys.argv[2]
    else:
        video_source = 'parking_video.mp4.mp4'
    
    # Check if video exists
    if not os.path.exists(video_source):
        print(f"[ERROR] Video not found: {video_source}")
        return
    
    # Run system
    try:
        system = SmartParkingSystem(slots_json)
        system.run(video_source)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
