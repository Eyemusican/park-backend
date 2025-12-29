"""
SMART PARKING SYSTEM MVP - DATABASE INTEGRATED
Connects YOLO detection with PostgreSQL database for persistent storage
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
from db_helper import ParkingDB

# ============================================================================
# PARKING SLOT CLASS - WITH DATABASE INTEGRATION
# ============================================================================
class ParkingSlot:
    """Represents one parking slot with occupancy tracking and database sync"""
    def __init__(self, slot_id, points, db_slot_id=None):
        self.id = slot_id
        self.points = points
        self.polygon = Polygon(points)
        self.db_slot_id = db_slot_id  # Database slot_id
        
        # Occupancy state
        self.is_occupied = False
        self.occupied_since = None
        self.last_vacant = time.time()
        self.occupying_vehicle_id = None
        
        # Database event tracking
        self.current_event_id = None  # Current parking event ID in database
        
        # Simple counters for stability
        self.vacant_frames = 0
        self.occupied_frames = 0
        
        # Statistics
        self.total_occupancies = 0
        self.total_duration = 0.0

    def mark_occupied(self, vehicle_id=None):
        """Mark slot as occupied - returns True if this is a NEW occupation"""
        new_occupation = False
        
        if not self.is_occupied:
            # NEW occupation
            self.is_occupied = True
            self.occupied_since = time.time()
            self.occupying_vehicle_id = vehicle_id
            self.total_occupancies += 1
            new_occupation = True
            
        elif vehicle_id and self.occupying_vehicle_id and vehicle_id != self.occupying_vehicle_id:
            # Different vehicle - reset timer
            if self.occupied_since:
                duration = time.time() - self.occupied_since
                self.total_duration += duration
            self.occupied_since = time.time()
            self.occupying_vehicle_id = vehicle_id
            self.total_occupancies += 1
            new_occupation = True

        self.is_occupied = True
        self.vacant_frames = 0
        return new_occupation

    def mark_vacant(self):
        """Mark slot as vacant - returns True if this is a NEW vacancy"""
        new_vacancy = False
        
        if self.is_occupied:
            if self.occupied_since:
                duration = time.time() - self.occupied_since
                self.total_duration += duration
            self.is_occupied = False
            self.occupied_since = None
            self.occupying_vehicle_id = None
            self.last_vacant = time.time()
            new_vacancy = True
            
        self.occupied_frames = 0
        return new_vacancy

    def get_duration(self):
        """Get current parking duration in seconds"""
        if not self.is_occupied or not self.occupied_since:
            return 0
        return time.time() - self.occupied_since


# ============================================================================
# SMART PARKING MVP CLASS - DATABASE INTEGRATED
# ============================================================================
class SmartParkingMVP:
    """
    Core parking system with database integration
    Records all parking events to PostgreSQL
    """
    
    def __init__(self, slots_json, parking_area_name="Norzin Lam Parking", model_path='yolov8m.pt'):
        print("="*70)
        print("SMART PARKING SYSTEM MVP - DATABASE INTEGRATED")
        print("="*70)

        # Initialize database connection
        print("\nInitializing database connection...")
        self.db = ParkingDB()
        self.parking_area_name = parking_area_name
        self.parking_id = None
        
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

        # Load parking slots
        print(f"\nLoading slots: {slots_json}")
        with open(slots_json, 'r') as f:
            data = json.load(f)

        # Initialize database parking area and slots
        self._initialize_database(data['slots'])
        
        print(f"✅ Loaded {len(self.slots)} parking slots")

        # Configuration
        self.overlap_threshold = 0.4
        self.conf_threshold = 0.25
        self.iou_threshold = 0.3

        # Vehicle tracking
        self.vehicle_type_history = {}
        self.history_window = 10
        self.vehicle_positions = {}
        self.position_history_window = 5
        self.stationary_threshold = 10
        self.vehicle_classes = [2, 3, 5, 7]

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()

        print("\nConfiguration:")
        print(f"  Parking Area: {self.parking_area_name} (ID: {self.parking_id})")
        print(f"  Overlap threshold: {self.overlap_threshold*100:.0f}%")
        print(f"  Detection confidence: {self.conf_threshold}")
        print(f"  Database: Connected ✅")
        print("="*70 + "\n")

    def _initialize_database(self, slot_data):
        """Initialize parking area and slots in database"""
        # Create or get parking area
        self.parking_id = self.db.add_parking_area(self.parking_area_name, len(slot_data))
        
        if not self.parking_id:
            print("❌ Failed to create parking area in database")
            sys.exit(1)
        
        # Create parking slots
        slot_numbers = [s['id'] for s in slot_data]
        success = self.db.add_parking_slots(self.parking_id, slot_numbers)
        
        if not success:
            print("❌ Failed to create parking slots in database")
            sys.exit(1)
        
        # Create ParkingSlot objects with database IDs
        self.slots = []
        for s in slot_data:
            db_slot_id = self.db.get_slot_id(self.parking_id, s['id'])
            slot = ParkingSlot(s['id'], s['points'], db_slot_id)
            self.slots.append(slot)
        
        print(f"✅ Database initialized: {self.parking_area_name}")

    def _record_arrival(self, slot):
        """Record vehicle arrival in database"""
        if slot.db_slot_id:
            event_id = self.db.record_arrival(slot.db_slot_id)
            slot.current_event_id = event_id
            if event_id:
                print(f"📍 DB: Slot #{slot.id} - Arrival recorded (Event #{event_id})")

    def _record_departure(self, slot):
        """Record vehicle departure in database"""
        if slot.db_slot_id and slot.current_event_id:
            event_id = self.db.record_departure(slot.db_slot_id)
            if event_id:
                print(f"🚗 DB: Slot #{slot.id} - Departure recorded (Event #{event_id})")
            slot.current_event_id = None

    def process_frame(self, frame):
        """Process one frame: detect vehicles, update occupancy, sync with database"""
        self.frame_count += 1

        # Downscale if needed
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            process_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            scale_back = 1.0 / scale
        else:
            process_frame = frame
            scale_back = 1.0

        # YOLO detection with tracking
        results = self.model.track(
            process_frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.vehicle_classes,
            verbose=False,
            tracker='bytetrack.yaml'
        )

        vehicles = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1*scale_back), int(y1*scale_back), int(x2*scale_back), int(y2*scale_back)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                track_id = int(box.id[0]) if box.id is not None else None
                
                vehicles.append({
                    'id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls
                })

        # Update slot occupancy AND sync with database
        self.update_occupancy(vehicles)
        
        return vehicles

    def update_occupancy(self, vehicles):
        """Update slot occupancy and sync with database"""
        for slot in self.slots:
            best_overlap = 0
            best_vehicle_id = None
            
            for vehicle in vehicles:
                if vehicle['id'] is None:
                    continue
                    
                x1, y1, x2, y2 = vehicle['bbox']
                vehicle_box = shapely_box(x1, y1, x2, y2)
                
                try:
                    overlap_area = slot.polygon.intersection(vehicle_box).area
                    vehicle_area = vehicle_box.area
                    overlap_ratio = overlap_area / vehicle_area if vehicle_area > 0 else 0
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_vehicle_id = vehicle['id']
                except:
                    continue
            
            # Update occupancy state
            if best_overlap >= self.overlap_threshold:
                # Mark as occupied
                is_new = slot.mark_occupied(best_vehicle_id)
                if is_new:
                    # Record arrival in database
                    self._record_arrival(slot)
                    
                slot.occupied_frames += 1
                if slot.occupied_frames >= 3:
                    slot.vacant_frames = 0
                    
            else:
                # Check if should mark as vacant
                if slot.is_occupied:
                    if best_overlap < 0.2:
                        slot.vacant_frames += 1
                        if slot.vacant_frames >= 5:
                            # Mark as vacant
                            is_new = slot.mark_vacant()
                            if is_new:
                                # Record departure in database
                                self._record_departure(slot)
                else:
                    slot.vacant_frames += 1

    def visualize(self, frame, vehicles):
        """Draw visualization"""
        overlay = frame.copy()
        
        # Draw slots
        for slot in self.slots:
            pts = np.array(slot.points, np.int32)
            
            if slot.is_occupied:
                color = (0, 0, 255)  # RED
                thickness = 3
            else:
                color = (0, 255, 0)  # GREEN
                thickness = 2
            
            cv2.polylines(frame, [pts], True, color, thickness)
            cv2.fillPoly(overlay, [pts], color)
            
            # Label
            center = np.mean(pts, axis=0).astype(int)
            label = f"#{slot.id}"
            
            if slot.is_occupied:
                duration = slot.get_duration()
                label += f" {self.format_duration(duration)}"
            
            cv2.putText(frame, label, tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw vehicles
        for vehicle in vehicles:
            if vehicle['id'] is None:
                continue
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"ID:{vehicle['id']}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Stats overlay
        total_slots = len(self.slots)
        occupied = sum(1 for s in self.slots if s.is_occupied)
        vacant = total_slots - occupied
        occupancy_rate = (occupied / total_slots * 100) if total_slots > 0 else 0
        
        stats_y = 40
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Parking: {self.parking_area_name}", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 30
        cv2.putText(frame, f"Capacity: {occupied}/{total_slots} ({occupancy_rate:.1f}%)",
                   (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 30
        cv2.putText(frame, f"Occupied: {occupied}", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        stats_y += 25
        cv2.putText(frame, f"Vacant: {vacant}", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        stats_y += 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        stats_y += 20
        cv2.putText(frame, f"DB: Connected", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

    def format_duration(self, seconds):
        """Format duration as XmYs"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def run(self, video_source, output_video=None):
        """Run the parking system"""
        print("Starting Smart Parking MVP with Database Integration...")
        print(f"Video: {video_source}\n")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("[ERROR] Cannot open video")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}\n")

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        max_display_width = 1280
        display_scale = min(max_display_width / frame_width, 1.0)
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)

        print("Controls: Q - Quit | S - Save | SPACE - Pause\n" + "="*70 + "\n")

        window_name = 'Smart Parking MVP - Database Integrated'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[END] Video finished")
                    break

                vehicles = self.process_frame(frame)
                vis_frame = self.visualize(frame, vehicles)

                if writer:
                    writer.write(vis_frame)

                display_frame = cv2.resize(vis_frame, (display_width, display_height))
                cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, vis_frame)
                print(f"Saved: {filename}")
            elif key == ord(' '):
                paused = not paused

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        self.print_summary()

    def print_summary(self):
        """Print final summary with database stats"""
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)

        total_time = time.time() - self.start_time
        total_slots = len(self.slots)
        occupied = sum(1 for s in self.slots if s.is_occupied)

        print(f"\nSession Duration: {self.format_duration(total_time)}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Capacity: {occupied}/{total_slots} occupied")

        # Get database stats
        db_stats = self.db.get_parking_stats(self.parking_id)
        if db_stats:
            print(f"\nDatabase Stats:")
            print(f"  Total Slots: {db_stats['total_slots']}")
            print(f"  Currently Occupied: {db_stats['occupied_slots']}")
            print(f"  Available: {db_stats['available_slots']}")

        print(f"\nSlot Statistics:")
        for slot in self.slots:
            avg_duration = slot.total_duration / slot.total_occupancies if slot.total_occupancies > 0 else 0
            status = "OCCUPIED" if slot.is_occupied else "VACANT"
            print(f"  Slot #{slot.id}: {status}")
            print(f"    Total uses: {slot.total_occupancies}")
            print(f"    Avg duration: {self.format_duration(avg_duration)}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("="*70)
        print("SMART PARKING SYSTEM MVP - DATABASE INTEGRATED")
        print("="*70)
        print("\nUsage:")
        print("  python smart_parking_db.py --run <video_file> [output_video]")
        print("\nExample:")
        print("  python smart_parking_db.py --run parking.mp4")
        print("  python smart_parking_db.py --run parking.mp4 output.mp4")
        print("="*70)
        return

    mode = sys.argv[1]

    if mode == '--run':
        if len(sys.argv) < 3:
            print("[ERROR] Video file required")
            return

        video_file = sys.argv[2]
        if not os.path.exists(video_file):
            print(f"[ERROR] Video not found: {video_file}")
            return

        slots_json = 'configs/parking_slots.json'
        if not os.path.exists(slots_json):
            print(f"[ERROR] Slots not found: {slots_json}")
            return

        output_video = sys.argv[3] if len(sys.argv) > 3 else None

        # Run system with database integration
        system = SmartParkingMVP(slots_json, parking_area_name="Norzin Lam Parking")
        system.run(video_file, output_video)

    else:
        print(f"[ERROR] Unknown mode: {mode}")


if __name__ == "__main__":
    main()