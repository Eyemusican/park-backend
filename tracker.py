"""
Smart Parking System - Phase 2: Vehicle Tracking (ENHANCED)
Individual vehicle tracking with stability filtering, occlusion handling, and duplicate removal
"""
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional
import time
import config


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes
    
    Args:
        box1: Bounding box (x1, y1, x2, y2)
        box2: Bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU score (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def remove_duplicate_detections(detections: List[Dict], iou_threshold: float = 0.65) -> List[Dict]:
    """
    Remove overlapping detections that represent the same vehicle
    
    This fixes Issue #1: Duplicate IDs on same vehicle
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering boxes as duplicates
        
    Returns:
        Filtered list of detections with duplicates removed
    """
    if len(detections) == 0:
        return detections
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for i, det in enumerate(sorted_dets):
        should_keep = True
        
        # Check against all kept detections
        for kept_det in keep:
            iou = calculate_iou(det['bbox'], kept_det['bbox'])
            
            # If high overlap, this is a duplicate
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(det)
    
    return keep


class VehicleTracker:
    """
    Tracks individual vehicle across frames
    
    Features:
    - Position history tracking
    - Motion detection (moving/stationary)
    - Speed estimation
    - Track persistence management
    """
    
    def __init__(self, track_id: int, class_id: int, class_name: str, 
                 initial_position: Tuple[int, int], initial_bbox: Tuple[int, int, int, int]):
        """
        Initialize vehicle tracker
        
        Args:
            track_id: Unique tracking ID
            class_id: Vehicle class ID (COCO)
            class_name: Vehicle class name (car, truck, etc.)
            initial_position: Initial center position (x, y)
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
        """
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        
        # Position tracking
        self.positions = deque(maxlen=config.TRACK_HISTORY_LENGTH)
        self.positions.append(initial_position)
        
        # Bounding box history
        self.bboxes = deque(maxlen=config.TRACK_HISTORY_LENGTH)
        self.bboxes.append(initial_bbox)
        
        # Motion state
        self.is_moving = False
        self.consecutive_moving_frames = 0
        self.consecutive_stationary_frames = 0
        self.status = "MOVING"  # MOVING, STOPPED, PARKED
        
        # Occlusion handling
        self.occluded = False
        self.confidence_history = deque(maxlen=5)
        self.confidence_history.append(1.0)
        
        # Parking detection
        self.stationary_start_time = None
        self.park_duration = 0.0
        
        # Statistics
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.total_frames = 1
        self.total_distance = 0.0
        self.avg_speed = 0.0
        
        # Current state
        self.current_position = initial_position
        self.current_bbox = initial_bbox
        self.current_confidence = 1.0
        self.lost_frames = 0
    
    def update(self, position: Tuple[int, int], bbox: Tuple[int, int, int, int], confidence: float = 1.0):
        """
        Update tracker with new detection
        
        Args:
            position: New center position (x, y)
            bbox: New bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
        """
        # Update position history
        self.positions.append(position)
        self.bboxes.append(bbox)
        self.confidence_history.append(confidence)
        
        # Calculate movement
        if len(self.positions) >= 2:
            prev_pos = self.positions[-2]
            distance = self._calculate_distance(prev_pos, position)
            self.total_distance += distance
            
            # Update motion state
            self._update_motion_state(distance)
        
        # Update current state
        self.current_position = position
        self.current_bbox = bbox
        self.current_confidence = confidence
        self.last_seen = time.time()
        self.total_frames += 1
        self.lost_frames = 0
        self.occluded = False
        
        # Calculate average speed (pixels per frame)
        if self.total_frames > 1:
            self.avg_speed = self.total_distance / self.total_frames
        
        # Update parking duration
        self._update_parking_status()
    
    def mark_lost(self):
        """Mark tracker as lost (no detection in current frame)"""
        self.lost_frames += 1
    
    def is_lost(self) -> bool:
        """Check if tracker should be removed"""
        return self.lost_frames > config.TRACK_BUFFER
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _update_motion_state(self, distance: float):
        """
        Update vehicle motion state based on movement
        
        Args:
            distance: Distance moved in current frame
        """
        if distance >= config.MOTION_THRESHOLD:
            # Vehicle is moving
            self.consecutive_moving_frames += 1
            self.consecutive_stationary_frames = 0
            
            if self.consecutive_moving_frames >= config.MOVING_FRAMES:
                self.is_moving = True
                self.status = "MOVING"
                self.stationary_start_time = None
        else:
            # Vehicle is stationary
            self.consecutive_stationary_frames += 1
            self.consecutive_moving_frames = 0
            
            if self.consecutive_stationary_frames >= config.STATIONARY_FRAMES:
                self.is_moving = False
                if self.stationary_start_time is None:
                    self.stationary_start_time = time.time()
    
    def _update_parking_status(self):
        """Update vehicle status: MOVING, STOPPED, or PARKED"""
        if self.is_moving:
            self.status = "MOVING"
            self.park_duration = 0.0
        elif self.stationary_start_time is not None:
            self.park_duration = time.time() - self.stationary_start_time
            
            # Consider parked after 1.5 seconds of being stationary
            if self.park_duration >= 1.5:
                self.status = "PARKED"
            else:
                self.status = "STOPPED"
        else:
            self.status = "STOPPED"
            self.park_duration = 0.0
    
    def get_trail(self, max_points: int = None) -> List[Tuple[int, int]]:
        """
        Get position trail for visualization
        
        Args:
            max_points: Maximum number of trail points (default: config.TRAIL_LENGTH)
            
        Returns:
            List of positions
        """
        max_points = max_points or config.TRAIL_LENGTH
        return list(self.positions)[-max_points:]
    
    def get_stats(self) -> Dict:
        """
        Get tracker statistics
        
        Returns:
            Dictionary with tracking stats
        """
        duration = self.last_seen - self.first_seen
        
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'total_frames': self.total_frames,
            'duration': duration,
            'total_distance': self.total_distance,
            'avg_speed': self.avg_speed,
            'is_moving': self.is_moving,
            'status': self.status,
            'park_duration': self.park_duration,
            'occluded': self.occluded,
            'lost_frames': self.lost_frames
        }
    
    def get_park_duration(self) -> float:
        """Get parking duration in seconds"""
        return self.park_duration
    
    def get_average_confidence(self) -> float:
        """Get average confidence from recent detections"""
        if len(self.confidence_history) > 0:
            return np.mean(list(self.confidence_history))
        return 0.0
    
    def __repr__(self):
        status = self.status
        occ_str = " [OCCLUDED]" if self.occluded else ""
        return f"VehicleTracker(ID:{self.track_id}, {self.class_name}, {status}{occ_str})"


class StabilityFilter:
    """
    Filters detections for stability to prevent flickering IDs
    
    This fixes Issue #3: Flickering IDs
    """
    
    def __init__(self, history_length: int = 5, min_consistent_frames: int = 3):
        """
        Initialize stability filter
        
        Args:
            history_length: Number of frames to keep in history
            min_consistent_frames: Minimum frames needed for stable detection
        """
        self.detection_history: Dict[int, deque] = {}
        self.history_length = history_length
        self.min_consistent_frames = min_consistent_frames
    
    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections to only include stable, consistent tracks
        
        Args:
            detections: List of detection dictionaries with track_id
            
        Returns:
            Filtered list of stable detections
        """
        stable_detections = []
        current_ids = set()
        
        for det in detections:
            if 'track_id' not in det or det['track_id'] is None:
                continue
            
            track_id = det['track_id']
            current_ids.add(track_id)
            
            # Initialize history for new tracks
            if track_id not in self.detection_history:
                self.detection_history[track_id] = deque(maxlen=self.history_length)
            
            # Add current detection to history
            self.detection_history[track_id].append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_id': det['class_id']
            })
            
            history = self.detection_history[track_id]
            
            # Accept if:
            # 1. High confidence new track (>0.6)
            # 2. Consistent detections over min_consistent_frames
            if len(history) == 1 and det['confidence'] > 0.6:
                # High confidence new track, accept immediately
                stable_detections.append(det)
            elif len(history) >= self.min_consistent_frames:
                # Check average confidence
                avg_conf = np.mean([h['confidence'] for h in history])
                if avg_conf > 0.35:
                    stable_detections.append(det)
        
        # Clean up old histories
        for track_id in list(self.detection_history.keys()):
            if track_id not in current_ids:
                # Keep for a few more frames in case track returns
                if len(self.detection_history[track_id]) >= self.history_length:
                    del self.detection_history[track_id]
        
        return stable_detections


class TrackingManager:
    """
    Manages all active vehicle trackers with enhanced occlusion handling
    
    Features:
    - Track lifecycle management
    - Occlusion handling (keeps lost tracks for brief period)
    - Statistics aggregation
    - Track ID assignment
    - Lost track cleanup
    
    This fixes Issue #2: ID changes during occlusion
    """
    
    def __init__(self):
        """Initialize tracking manager"""
        self.active_tracks: Dict[int, VehicleTracker] = {}
        self.lost_tracks: Dict[int, Tuple[int, VehicleTracker]] = {}  # {track_id: (frame_num, tracker)}
        self.current_frame = 0
        self.max_lost_frames = 30  # Keep lost tracks visible for 1 second (at 30fps)
        self.total_tracked = 0
        self.total_lost = 0
        
        # Stability filter
        self.stability_filter = StabilityFilter()
    
    def update_trackers(self, detections: List[Dict], frame_number: int = None):
        """
        Update all trackers with new detections
        
        Args:
            detections: List of detection dictionaries with track IDs
            frame_number: Current frame number (for occlusion tracking)
        """
        if frame_number is not None:
            self.current_frame = frame_number
        else:
            self.current_frame += 1
        
        # Apply stability filter
        stable_detections = self.stability_filter.filter_detections(detections)
        
        # Mark all existing trackers as potentially lost
        current_ids = set()
        
        # Update trackers with new detections
        for det in stable_detections:
            if 'track_id' not in det or det['track_id'] is None:
                continue
            
            track_id = int(det['track_id'])
            current_ids.add(track_id)
            
            position = det['center']
            bbox = det['bbox']
            confidence = det.get('confidence', 1.0)
            
            # If this was a lost track, restore it
            if track_id in self.lost_tracks:
                _, tracker = self.lost_tracks[track_id]
                self.active_tracks[track_id] = tracker
                del self.lost_tracks[track_id]
            
            if track_id in self.active_tracks:
                # Update existing tracker
                self.active_tracks[track_id].update(position, bbox, confidence)
            else:
                # Create new tracker
                if len(self.active_tracks) < config.MAX_TRACKED_VEHICLES:
                    tracker = VehicleTracker(
                        track_id=track_id,
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        initial_position=position,
                        initial_bbox=bbox
                    )
                    self.active_tracks[track_id] = tracker
                    self.total_tracked += 1
        
        # Move missing tracks to lost_tracks
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_ids:
                tracker = self.active_tracks[track_id]
                tracker.occluded = True
                self.lost_tracks[track_id] = (self.current_frame, tracker)
                del self.active_tracks[track_id]
        
        # Clean up very old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            lost_frame, _ = self.lost_tracks[track_id]
            if self.current_frame - lost_frame > self.max_lost_frames:
                del self.lost_tracks[track_id]
                self.total_lost += 1
    
    def get_tracker(self, track_id: int) -> Optional[VehicleTracker]:
        """Get tracker by ID (searches both active and lost)"""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        elif track_id in self.lost_tracks:
            _, tracker = self.lost_tracks[track_id]
            return tracker
        return None
    
    def get_all_trackers(self) -> Dict[int, VehicleTracker]:
        """Get all trackers (both active and recently lost for visualization)"""
        all_tracks = {}
        
        # Add active tracks
        for tid, tracker in self.active_tracks.items():
            all_tracks[tid] = tracker
        
        # Add recently lost tracks (show for brief period in gray)
        for tid, (lost_frame, tracker) in self.lost_tracks.items():
            frames_lost = self.current_frame - lost_frame
            if frames_lost <= 10:  # Show for ~0.3 seconds
                all_tracks[tid] = tracker
        
        return all_tracks
    
    def get_active_only(self) -> Dict[int, VehicleTracker]:
        """Get only active tracks (not occluded)"""
        return self.active_tracks
    
    def get_active_count(self) -> int:
        """Get number of active trackers"""
        return len(self.active_tracks)
    
    def get_moving_count(self) -> int:
        """Get number of moving vehicles"""
        return sum(1 for t in self.active_tracks.values() if t.status == "MOVING")
    
    def get_stationary_count(self) -> int:
        """Get number of stationary vehicles (stopped or parked)"""
        return sum(1 for t in self.active_tracks.values() if t.status in ["STOPPED", "PARKED"])
    
    def get_parked_count(self) -> int:
        """Get number of parked vehicles"""
        return sum(1 for t in self.active_tracks.values() if t.status == "PARKED")
    
    def get_stats(self) -> Dict:
        """
        Get overall tracking statistics
        
        Returns:
            Dictionary with tracking stats
        """
        return {
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'moving_vehicles': self.get_moving_count(),
            'stationary_vehicles': self.get_stationary_count(),
            'parked_vehicles': self.get_parked_count(),
            'total_tracked': self.total_tracked,
            'total_lost': self.total_lost
        }
    
    def reset(self):
        """Reset all trackers"""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.total_tracked = 0
        self.total_lost = 0
        self.current_frame = 0
    
    def __len__(self):
        return len(self.active_tracks)
    
    def __repr__(self):
        return f"TrackingManager(active={len(self.active_tracks)}, lost={len(self.lost_tracks)}, total={self.total_tracked})"
