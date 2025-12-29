"""
Smart Parking System - Phase 2: Vehicle Detection and Tracking
Main application entry point

Usage:
    python main.py                    # Use webcam with tracking
    python main.py --video path.mp4   # Use video file with tracking
    python main.py --no-tracking      # Disable tracking (Phase 1 mode)
"""
import cv2
import argparse
import sys
import logging
from pathlib import Path
import time
import numpy as np

import config
from detector import VehicleDetector
from utils import (
    create_visualization, 
    resize_for_display,
    save_snapshot,
    draw_performance_warning
)

logger = logging.getLogger(__name__)


class VehicleDetectionApp:
    """
    Main application class for vehicle detection and tracking
    """
    
    def __init__(self, video_source=None, enable_tracking=None):
        """
        Initialize the application
        
        Args:
            video_source: Video source (webcam index, video file path, or None for default)
            enable_tracking: Enable vehicle tracking (None uses config default)
        """
        logger.info("=" * 80)
        phase = "PHASE 2: VEHICLE TRACKING" if (enable_tracking if enable_tracking is not None else config.ENABLE_TRACKING) else "PHASE 1: VEHICLE DETECTION"
        logger.info(f"SMART PARKING SYSTEM - {phase}")
        logger.info("=" * 80)
        
        # Video source
        if video_source is None:
            self.video_source = config.DEFAULT_VIDEO_SOURCE
        elif isinstance(video_source, str) and video_source.isdigit():
            self.video_source = int(video_source)
        else:
            self.video_source = video_source
        
        logger.info(f"Video source: {self.video_source}")
        
        # Initialize detector with tracking
        logger.info("Initializing vehicle detector...")
        self.detector = VehicleDetector(enable_tracking=enable_tracking)
        self.enable_tracking = self.detector.enable_tracking
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Snapshot management
        self.last_snapshot_time = time.time()
        self.snapshot_counter = 0
        
        logger.info("[OK] Application initialized successfully")
    
    def start(self):
        """Start the detection system"""
        logger.info("\nStarting vehicle detection and tracking...")
        logger.info("Press 'Q' to quit, 'S' to save snapshot, 'R' to reset stats, 'T' to toggle tracking")
        
        # Open video source
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            logger.error(f"[ERROR] Failed to open video source: {self.video_source}")
            return False
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"[OK] Video opened: {width}x{height} @ {fps} FPS")
        
        # Main detection loop
        self.running = True
        frame_skip_counter = 0
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("End of video or failed to read frame")
                    break
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter < config.FRAME_SKIP:
                    continue
                frame_skip_counter = 0
                
                # Run detection
                detections, process_time, vehicle_count = self.detector.detect(frame, frame_skip_counter)
                
                # Get statistics
                stats = self.detector.get_stats()
                
                # Get tracking manager if enabled
                tracking_manager = self.detector.tracking_manager if self.enable_tracking else None
                
                # Create visualization
                vis_frame = create_visualization(
                    frame, 
                    detections, 
                    stats, 
                    vehicle_count,
                    tracking_manager=tracking_manager,
                    show_header=True
                )
                
                # Add performance warning if needed
                vis_frame = draw_performance_warning(vis_frame, stats)
                
                # Resize for display
                display_frame = resize_for_display(vis_frame)
                
                # Show frame
                window_title = f"Smart Parking - {'Tracking' if self.enable_tracking else 'Detection'}"
                cv2.imshow(window_title, display_frame)
                
                # Auto-save snapshots if enabled
                if config.SAVE_SNAPSHOTS:
                    current_time = time.time()
                    if current_time - self.last_snapshot_time >= config.SNAPSHOT_INTERVAL:
                        self._save_snapshot(vis_frame)
                        self.last_snapshot_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s') or key == ord('S'):
                    self._save_snapshot(vis_frame)
                elif key == ord('r') or key == ord('R'):
                    self.detector.reset_stats()
                    logger.info("Statistics reset")
                elif key == ord('t') or key == ord('T'):
                    # Toggle tracking visualization (not implemented yet)
                    logger.info("Tracking toggle (feature not yet implemented)")
                elif key == 27:  # ESC key
                    logger.info("Quit requested by user (ESC)")
                    break
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user (Ctrl+C)")
        
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
        
        finally:
            self._cleanup()
        
        return True
    
    def _save_snapshot(self, frame: np.ndarray):
        """Save snapshot to disk"""
        self.snapshot_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"vehicle_detection_{timestamp}_{self.snapshot_counter:04d}.jpg"
        filepath = config.SNAPSHOTS_PATH / filename
        
        if save_snapshot(frame, filepath):
            logger.info(f"[OK] Snapshot saved: {filename}")
        else:
            logger.error(f"[ERROR] Failed to save snapshot: {filename}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("\nCleaning up...")
        
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.detector.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total frames processed: {stats['frame_count']}")
        logger.info(f"Total vehicles detected: {stats['total_detections']}")
        logger.info(f"Average vehicles per frame: {stats['avg_detections_per_frame']:.2f}")
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Average process time: {stats['process_time_ms']:.2f}ms")
        
        # Tracking statistics
        if 'active_tracks' in stats:
            logger.info(f"\nTracking Statistics:")
            logger.info(f"Total vehicles tracked: {stats.get('total_tracked', 0)}")
            logger.info(f"Active tracks: {stats.get('active_tracks', 0)}")
            logger.info(f"Lost tracks: {stats.get('total_lost', 0)}")
        
        logger.info("=" * 80)
        
        # Performance evaluation
        if stats['fps'] >= config.TARGET_FPS:
            logger.info("[OK] PERFORMANCE: EXCELLENT (Target FPS achieved)")
        else:
            logger.warning(f"[WARNING] PERFORMANCE: BELOW TARGET (Current: {stats['fps']:.1f} FPS, Target: {config.TARGET_FPS} FPS)")
        
        logger.info("\nApplication stopped successfully")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Smart Parking System - Phase 2: Vehicle Detection and Tracking"
    )
    parser.add_argument(
        '--video', '--source', '-v',
        type=str,
        default=None,
        help='Video source (webcam index or video file path)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='YOLO model path (default: yolov8m.pt)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help='Disable vehicle tracking (Phase 1 mode)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Apply command line overrides
    if args.model:
        config.YOLO_MODEL = args.model
    if args.conf:
        config.CONFIDENCE_THRESHOLD = args.conf
    if args.debug:
        config.DEBUG_MODE = True
        logger.setLevel(logging.DEBUG)
    
    enable_tracking = not args.no_tracking
    
    # Create and start application
    try:
        app = VehicleDetectionApp(video_source=args.video, enable_tracking=enable_tracking)
        app.start()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
