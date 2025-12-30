"""
Smart Parking Duration Tracker - Production Ready
Tracks vehicle parking duration with high accuracy using:
- Vehicle tracking (ByteTrack/SORT)
- ROI-based slot detection
- Stability logic to prevent flickering
- Backend integration for real-time updates
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import requests
import json


class ParkingSession:
    """Represents a single parking session for a vehicle in a slot"""
    
    def __init__(self, vehicle_id: int, slot_id: str, entry_time: float, 
                 license_plate: str = None, color: str = None, vehicle_type: str = None):
        self.vehicle_id = vehicle_id
        self.slot_id = slot_id
        self.entry_time = entry_time
        self.exit_time: Optional[float] = None
        self.duration: float = 0.0
        self.is_active = True
        self.backend_notified_entry = False
        self.backend_notified_exit = False
        # Vehicle attributes
        self.license_plate = license_plate
        self.color = color
        self.vehicle_type = vehicle_type
        
    def get_current_duration(self) -> float:
        """Get current parking duration in seconds"""
        if self.is_active:
            return time.time() - self.entry_time
        return self.duration
    
    def end_session(self):
        """End this parking session"""
        if self.is_active:
            self.exit_time = time.time()
            self.duration = self.exit_time - self.entry_time
            self.is_active = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API calls"""
        return {
            'vehicle_id': self.vehicle_id,
            'slot_id': self.slot_id,
            'entry_time': datetime.fromtimestamp(self.entry_time, tz=timezone.utc).isoformat(),
            'exit_time': datetime.fromtimestamp(self.exit_time, tz=timezone.utc).isoformat() if self.exit_time else None,
            'duration': self.get_current_duration(),
            'status': 'PARKED' if self.is_active else 'LEFT'
        }


class VehicleInSlot:
    """Tracks a vehicle's presence in a specific slot"""
    
    def __init__(self, vehicle_id: int, slot_id: str, stability_threshold: int = 3):
        self.vehicle_id = vehicle_id
        self.slot_id = slot_id
        self.stability_threshold = stability_threshold  # Frames before confirming
        self.present_count = 0
        self.absent_count = 0
        self.is_stable = False
        self.is_present = False
        
    def update_presence(self, is_present: bool) -> Tuple[bool, bool]:
        """
        Update vehicle presence and return (is_stable, state_changed)
        
        Returns:
            (is_stable, state_changed): Tuple indicating if vehicle is stable and if state changed
        """
        state_changed = False
        
        if is_present:
            self.present_count += 1
            self.absent_count = 0
            
            # Became stable and present
            if not self.is_stable and self.present_count >= self.stability_threshold:
                self.is_stable = True
                self.is_present = True
                state_changed = True
        else:
            self.absent_count += 1
            self.present_count = 0
            
            # Became unstable (vehicle left)
            if self.is_stable and self.absent_count >= self.stability_threshold:
                self.is_stable = False
                self.is_present = False
                state_changed = True
        
        return self.is_stable and self.is_present, state_changed


class ParkingDurationTracker:
    """
    Main tracker for parking duration
    - Tracks vehicle-to-slot assignments
    - Manages parking sessions
    - Sends backend updates
    - Handles edge cases (re-entries, flickers, etc.)

    """
    def __init__(self, backend_url: str = "http://localhost:5000",
                parking_area_id: int = 1,
                stability_frames: int = 60,  # 4 seconds at 15fps - MORE STABLE
                min_overlap: float = 0.30):

        """
        Initialize parking duration tracker
        
        Args:
            backend_url: Backend API URL
            stability_frames: Number of frames before confirming entry/exit
            min_overlap: Minimum overlap ratio to consider vehicle in slot (0.0-1.0)
        """
        self.backend_url = backend_url
        self.parking_area_id = parking_area_id 
        self.stability_frames = stability_frames
        self.min_overlap = min_overlap
        
        # Active parking sessions: slot_id -> ParkingSession
        self.active_sessions: Dict[str, ParkingSession] = {}
        
        # Vehicle stability tracking: (vehicle_id, slot_id) -> VehicleInSlot
        self.vehicle_stability: Dict[Tuple[int, str], VehicleInSlot] = {}
        
        # Completed sessions history
        self.completed_sessions: list = []
        
        # Initial detection mode - faster detection for already-parked vehicles
        self.initial_detection_mode = True
        self.initial_detection_frames = 30  # First 2 seconds for quick detection
        
        # Statistics
        self.frame_count = 0
        self.total_sessions = 0
        self.api_calls_sent = 0
        self.api_calls_failed = 0
        
        print("=" * 70)
        print("PARKING DURATION TRACKER INITIALIZED")
        print("=" * 70)
        print(f"Backend URL: {backend_url}")
        print(f"Stability threshold: {stability_frames} frames (~{stability_frames/15:.1f}s at 15fps)")
        print(f"Min overlap: {min_overlap * 100:.0f}%")
        print("=" * 70)
    
    def update(self, parking_slots: list):
        """
        Main update function - call this every frame
        Uses slot's locked vehicle ID instead of detection-based tracking
        
        Args:
            parking_slots: List of ParkingSlot objects with locked vehicle IDs
        """
        self.frame_count += 1
        
        # Update sessions based on slot's locked vehicle IDs
        self._update_sessions_from_slots(parking_slots)
    
    def _update_sessions_from_slots(self, parking_slots: list):
        """
        Update parking sessions based on slot's locked vehicle IDs
        This is simpler and more reliable than detection-based tracking
        
        Args:
            parking_slots: List of ParkingSlot objects with locked_vehicle_id
        """
        current_slot_ids = set()
        
        for slot in parking_slots:
            slot_id = str(slot.id)
            locked_vehicle_id = slot.get_locked_vehicle_id()
            
            if locked_vehicle_id is not None:
                # Slot has a locked vehicle ID
                current_slot_ids.add(slot_id)
                
                # Check if we need to start a new session
                if slot_id not in self.active_sessions:
                    # New parking session - get vehicle attributes from slot
                    self._start_parking_session(
                        locked_vehicle_id, slot_id,
                        license_plate=getattr(slot, 'license_plate', None),
                        color=getattr(slot, 'vehicle_color', None),
                        vehicle_type=getattr(slot, 'vehicle_type', None)
                    )
                elif self.active_sessions[slot_id].vehicle_id != locked_vehicle_id:
                    # Vehicle ID changed (shouldn't happen with locking, but handle it)
                    print(f"⚠️  [SLOT {slot_id}] Vehicle ID changed: {self.active_sessions[slot_id].vehicle_id} -> {locked_vehicle_id}")
                    self._end_parking_session(self.active_sessions[slot_id].vehicle_id, slot_id)
                    self._start_parking_session(
                        locked_vehicle_id, slot_id,
                        license_plate=getattr(slot, 'license_plate', None),
                        color=getattr(slot, 'vehicle_color', None),
                        vehicle_type=getattr(slot, 'vehicle_type', None)
                    )
        
        # Check for ended sessions (slots that became vacant)
        slots_to_end = []
        for slot_id, session in self.active_sessions.items():
            if slot_id not in current_slot_ids:
                # Session ended (vehicle left)
                slots_to_end.append((session.vehicle_id, slot_id))
        
        # End sessions that are no longer active
        for vehicle_id, slot_id in slots_to_end:
            self._end_parking_session(vehicle_id, slot_id)
    
    def _start_parking_session(self, vehicle_id: int, slot_id: str, 
                              license_plate: str = None, color: str = None, vehicle_type: str = None):
        """Start a new parking session"""
        # Check if slot already has an active session
        if slot_id in self.active_sessions:
            old_session = self.active_sessions[slot_id]
            
            # If same vehicle, ignore (already parked)
            if old_session.vehicle_id == vehicle_id:
                return
            
            # Different vehicle -> end old session first
            print(f"⚠️  [SLOT {slot_id}] Vehicle swap detected")
            self._end_parking_session(old_session.vehicle_id, slot_id)
        
        # Create new session
        session = ParkingSession(vehicle_id, slot_id, time.time(), 
                                license_plate=license_plate, 
                                color=color, 
                                vehicle_type=vehicle_type)
        self.active_sessions[slot_id] = session
        self.total_sessions += 1
        
        vehicle_info = f" {vehicle_type or ''}" if vehicle_type else ""
        vehicle_info += f" {color or ''}" if color else ""
        vehicle_info += f" [{license_plate}]" if license_plate else ""
        print(f"🅿️  [SLOT {slot_id}] PARKING STARTED - Vehicle ID: {vehicle_id}{vehicle_info}")
        
        # Notify backend
        self._notify_backend_entry(session)
    
    def _end_parking_session(self, vehicle_id: int, slot_id: str):
        """End an active parking session"""
        if slot_id not in self.active_sessions:
            return
        
        session = self.active_sessions[slot_id]
        
        # Verify it's the same vehicle
        if session.vehicle_id != vehicle_id:
            print(f"⚠️  [SLOT {slot_id}] Vehicle ID mismatch: {session.vehicle_id} != {vehicle_id}")
            return
        
        # End session
        session.end_session()
        duration = session.duration
        
        print(f"🚗 [SLOT {slot_id}] PARKING ENDED - Vehicle ID: {vehicle_id}, Duration: {duration:.1f}s ({duration/60:.1f}min)")
        
        # Notify backend
        self._notify_backend_exit(session)
        
        # Move to completed sessions
        self.completed_sessions.append(session)
        del self.active_sessions[slot_id]
    
    def _notify_backend_entry(self, session: ParkingSession):
        """Send parking entry event to backend"""
        if session.backend_notified_entry:
            return
        
        try:
            payload = {
                'slot_id': session.slot_id,
                'vehicle_id': session.vehicle_id,
                'parking_area_id': self.parking_area_id,
                'entry_time': datetime.fromtimestamp(session.entry_time, tz=timezone.utc).isoformat(),
                'status': 'PARKED',
                'license_plate': session.license_plate,
                'color': session.color,
                'vehicle_type': session.vehicle_type
            }
            
            response = requests.post(
                f"{self.backend_url}/api/parking/entry",
                json=payload,
                timeout=2
            )
            
            if response.status_code in [200, 201]:
                session.backend_notified_entry = True
                self.api_calls_sent += 1
                print(f"✅ Backend notified: ENTRY for slot {session.slot_id}")
            else:
                print(f"⚠️  Backend error: {response.status_code}")
                self.api_calls_failed += 1
                
        except Exception as e:
            print(f"❌ Backend connection failed: {e}")
            self.api_calls_failed += 1
    
    def _notify_backend_exit(self, session: ParkingSession):
        """Send parking exit event to backend"""
        if session.backend_notified_exit or not session.backend_notified_entry:
            return
        
        try:
            payload = {
                'slot_id': session.slot_id,
                'vehicle_id': session.vehicle_id,
                'exit_time': datetime.fromtimestamp(session.exit_time, tz=timezone.utc).isoformat(),
                'duration': session.duration,
                'parking_area_id': self.parking_area_id,
                'status': 'LEFT'
            }
            
            response = requests.post(
                f"{self.backend_url}/api/parking/exit",
                json=payload,
                timeout=2
            )
            
            if response.status_code in [200, 201]:
                session.backend_notified_exit = True
                self.api_calls_sent += 1
                print(f"✅ Backend notified: EXIT for slot {session.slot_id}, duration: {session.duration:.1f}s")
            else:
                print(f"⚠️  Backend error: {response.status_code}")
                self.api_calls_failed += 1
                
        except Exception as e:
            print(f"❌ Backend connection failed: {e}")
            self.api_calls_failed += 1
    
    def get_active_sessions(self) -> Dict[str, dict]:
        """Get all active parking sessions with current durations"""
        return {
            slot_id: {
                'slot_id': session.slot_id,
                'vehicle_id': session.vehicle_id,
                'entry_time': datetime.fromtimestamp(session.entry_time, tz=timezone.utc).isoformat(),
                'duration_seconds': session.get_current_duration(),
                'duration_minutes': session.get_current_duration() / 60,
                'status': 'PARKED'
            }
            for slot_id, session in self.active_sessions.items()
        }
    
    def get_slot_duration(self, slot_id: str) -> Optional[float]:
        """Get current parking duration for a specific slot"""
        if slot_id in self.active_sessions:
            return self.active_sessions[slot_id].get_current_duration()
        return None
    
    def get_statistics(self) -> dict:
        """Get tracker statistics"""
        return {
            'total_sessions': self.total_sessions,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'api_calls_sent': self.api_calls_sent,
            'api_calls_failed': self.api_calls_failed
        }
