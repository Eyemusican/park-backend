"""
Smart Parking System - Phase 2: Vehicle Detector with Tracking (ENHANCED)
YOLO-based vehicle detection and tracking with duplicate removal and stability filtering
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
from typing import List, Tuple, Dict, Optional
import config
from tracker import TrackingManager, remove_duplicate_detections

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE) if config.LOG_TO_FILE else logging.NullHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    High-performance vehicle detection and tracking system using YOLOv8
    
    Features:
    - Multi-class vehicle detection (cars, trucks, buses, motorcycles)
    - Real-time vehicle tracking with persistent IDs
    - Motion detection (moving/stationary)
    - Position history and trails
    - Real-time performance (15-20+ FPS)
    - GPU acceleration support
    """
    
    def __init__(self, model_path: str = None, enable_tracking: bool = None):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to YOLO model weights (defaults to config.YOLO_MODEL)
            enable_tracking: Enable vehicle tracking (defaults to config.ENABLE_TRACKING)
        """
        self.model_path = model_path or config.YOLO_MODEL
        self.enable_tracking = enable_tracking if enable_tracking is not None else config.ENABLE_TRACKING
        
        logger.info(f"Initializing VehicleDetector with model: {self.model_path}")
        logger.info(f"Tracking enabled: {self.enable_tracking}")
        
        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"[OK] Model loaded successfully: {self.model_path}")
            
            # Configure GPU if available
            if config.USE_GPU:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"[OK] Using device: {self.device}")
            else:
                self.device = 'cpu'
                logger.info("[OK] Using CPU")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise
        
        # Detection parameters
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        self.vehicle_classes = config.DETECT_CLASSES
        self.image_size = config.IMAGE_SIZE
        
        # Performance tracking
        self.fps = 0
        self.process_time = 0
        self.frame_count = 0
        self.total_detections = 0
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        
        # Tracking manager (Phase 2)
        if self.enable_tracking:
            self.tracking_manager = TrackingManager()
            logger.info("[OK] Tracking manager initialized")
        else:
            self.tracking_manager = None
        
        logger.info("[OK] VehicleDetector initialized successfully")
        logger.info(f"  - Confidence threshold: {self.conf_threshold}")
        logger.info(f"  - IoU threshold: {self.iou_threshold}")
        logger.info(f"  - Image size: {self.image_size}")
        logger.info(f"  - Vehicle classes: {list(config.VEHICLE_CLASSES.values())}")
    
    def detect(self, frame: np.ndarray, frame_number: int = None) -> Tuple[List[Dict], float, int]:
        """
        Detect and track vehicles in a frame with duplicate removal
        
        Args:
            frame: Input image/frame (numpy array)
            frame_number: Current frame number (for tracking)
            
        Returns:
            Tuple of (detections, process_time, vehicle_count)
            - detections: List of detection dictionaries
            - process_time: Processing time in seconds
            - vehicle_count: Number of vehicles detected
        """
        start_time = time.time()
        
        # Run YOLO detection/tracking
        try:
            if self.enable_tracking:
                # Use tracking mode with enhanced parameters
                results = self.model.track(
                    source=frame,
                    conf=self.conf_threshold,     # 0.4 for stability
                    iou=self.iou_threshold,       # 0.6 to prevent duplicates
                    classes=self.vehicle_classes,
                    imgsz=self.image_size,
                    persist=True,
                    tracker=config.TRACKER_TYPE,  # Custom bytetrack.yaml
                    verbose=config.VERBOSE,
                    device=self.device,
                    half=config.HALF_PRECISION,
                    max_det=config.MAX_DET
                )
            else:
                # Use detection-only mode
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=self.vehicle_classes,
                    imgsz=self.image_size,
                    verbose=config.VERBOSE,
                    device=self.device,
                    half=config.HALF_PRECISION,
                    max_det=config.MAX_DET
                )
            
            # Parse results
            detections = self._parse_results(results[0])
            
            # CRITICAL FIX: Remove duplicate detections (Issue #1)
            detections = remove_duplicate_detections(
                detections, 
                iou_threshold=config.DUPLICATE_IOU_THRESHOLD
            )
            
            # Update tracking manager (includes stability filtering)
            if self.enable_tracking and self.tracking_manager:
                self.tracking_manager.update_trackers(detections, frame_number)
            
            # Calculate metrics
            process_time = time.time() - start_time
            vehicle_count = len(detections)
            
            # Update statistics
            self.frame_count += 1
            self.process_time = process_time
            self.total_detections += vehicle_count
            self._update_fps()
            
            # Performance logging (debug mode)
            if config.DEBUG_MODE:
                logger.debug(f"Frame {self.frame_count}: {vehicle_count} vehicles, "
                           f"{process_time*1000:.1f}ms, {self.fps:.1f} FPS")
            
            return detections, process_time, vehicle_count
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], 0, 0
    
    def _parse_results(self, result) -> List[Dict]:
        """
        Parse YOLO results into standardized detection format
        
        Args:
            result: YOLO result object
            
        Returns:
            List of detection dictionaries with bbox, class, confidence, track_id
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        
        # Extract track IDs if tracking is enabled
        track_ids = None
        if self.enable_tracking and hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Create detection dictionaries
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': config.VEHICLE_CLASSES.get(int(cls), 'unknown'),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1),
                'track_id': int(track_ids[i]) if track_ids is not None else None
            }
            
            detections.append(detection)
        
        return detections
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= self.fps_update_interval:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def get_stats(self) -> Dict:
        """
        Get detection and tracking statistics
        
        Returns:
            Dictionary with performance and tracking metrics
        """
        stats = {
            'fps': self.fps,
            'process_time_ms': self.process_time * 1000,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / max(self.frame_count, 1)
        }
        
        # Add tracking stats if enabled
        if self.enable_tracking and self.tracking_manager:
            tracking_stats = self.tracking_manager.get_stats()
            stats.update(tracking_stats)
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.fps = 0
        self.process_time = 0
        self.frame_count = 0
        self.total_detections = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        if self.tracking_manager:
            self.tracking_manager.reset()
        
        logger.info("Statistics reset")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'model'):
            del self.model
            logger.info("VehicleDetector cleaned up")
