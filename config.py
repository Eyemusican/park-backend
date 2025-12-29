"""
Smart Parking System - Phase 1: Vehicle Detection Configuration
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
VIDEOS_PATH = PROJECT_ROOT / "videos"
OUTPUT_PATH = PROJECT_ROOT / "output"
SNAPSHOTS_PATH = OUTPUT_PATH / "snapshots"
LOGS_PATH = PROJECT_ROOT / "logs"
CONFIGS_PATH = PROJECT_ROOT / "configs"
MODELS_PATH = PROJECT_ROOT / "models"

# Create directories if they don't exist
for path in [OUTPUT_PATH, SNAPSHOTS_PATH, LOGS_PATH, CONFIGS_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# YOLO MODEL CONFIGURATION
# ============================================================================
# Model Selection: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), 
#                  yolov8l.pt (large), yolov8x.pt (xlarge)
YOLO_MODEL = "yolov8n.pt"  # Nano model - optimized for RTX 3050 multi-video (fast & low VRAM)

# YOLO Detection Parameters
CONFIDENCE_THRESHOLD = 0.25        # Minimum confidence for detections (lowered for immediate detection)
IOU_THRESHOLD = 0.65               # Non-Maximum Suppression threshold (balanced)
IMAGE_SIZE = 640                   # Optimized for RTX 3050 (640 = fast, 960 = accurate but slower)

# Vehicle Classes (COCO dataset class IDs)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}
DETECT_CLASSES = list(VEHICLE_CLASSES.keys())  # [2, 3, 5, 7]

# ============================================================================
# PERFORMANCE REQUIREMENTS
# ============================================================================
TARGET_FPS = 15                # Minimum target FPS
MAX_PROCESS_TIME_MS = 50       # Maximum processing time per frame in milliseconds

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
DEFAULT_VIDEO_SOURCE = 0       # 0 for webcam, or path to video file
DISPLAY_WIDTH = 1280           # Display window width
DISPLAY_HEIGHT = 720           # Display window height
FRAME_SKIP = 1                 # Process every Nth frame (1 = process all frames)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Bounding Box Colors (BGR format for OpenCV)
BBOX_COLORS = {
    "car": (0, 255, 0),        # Green
    "motorcycle": (255, 0, 255),  # Magenta
    "bus": (0, 165, 255),      # Orange
    "truck": (0, 0, 255)       # Red
}
DEFAULT_BBOX_COLOR = (0, 255, 0)  # Green for unrecognized classes
BBOX_THICKNESS = 2

# Text Settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)   # White
TEXT_BG_COLOR = (0, 0, 0)      # Black background

# Stats Display
SHOW_FPS = True
SHOW_PROCESS_TIME = True
SHOW_VEHICLE_COUNT = True
SHOW_CONFIDENCE = True         # Show confidence score on each detection

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
LOG_FILE = LOGS_PATH / "vehicle_detection.log"

# ============================================================================
# PHASE 2: VEHICLE TRACKING SETTINGS
# ============================================================================
ENABLE_TRACKING = True         # Enable vehicle tracking across frames
TRACKER_TYPE = "bytetrack.yaml"  # Custom tracker configuration file with enhanced occlusion handling

# Tracking Parameters
MAX_TRACKED_VEHICLES = 30      # Maximum number of simultaneously tracked vehicles
TRACK_HISTORY_LENGTH = 50      # Number of frames to keep in position history (balanced)
MIN_TRACK_CONFIDENCE = 0.25    # Minimum confidence to start tracking (lowered for immediate detection)
TRACK_BUFFER = 90              # Frames to keep lost tracks before deletion (balanced)

# Motion Detection
MOTION_THRESHOLD = 3.0         # Minimum pixel movement to consider vehicle moving (reduced for faster response)
STATIONARY_FRAMES = 10         # Frames vehicle must be still to be "stationary" (reduced for faster response)
MOVING_FRAMES = 3              # Frames vehicle must move to be "moving"

# Duplicate Detection
DUPLICATE_IOU_THRESHOLD = 0.65 # IoU threshold for removing duplicate detections on same vehicle

# Track Visualization
SHOW_TRACK_IDS = True          # Display track ID on vehicles
SHOW_MOTION_TRAIL = True       # Show vehicle movement trail
SHOW_MOTION_STATUS = True      # Show if vehicle is moving/stopped/parked
TRAIL_LENGTH = 30              # Number of points in motion trail
TRAIL_THICKNESS = 2            # Trail line thickness

# Color Coding System (Issue #4 Fix)
COLOR_MOVING = (0, 255, 0)         # GREEN - Vehicle is moving
COLOR_STOPPED = (0, 255, 255)      # YELLOW - Temporarily stopped (<1.5s)
COLOR_PARKED = (0, 0, 255)         # RED - Parked (stationary >1.5s)
COLOR_OCCLUDED = (128, 128, 128)   # GRAY - Recently lost/occluded
TRAIL_COLOR = (255, 0, 255)        # MAGENTA - Motion trail

# ============================================================================
# DEBUG & DEVELOPMENT
# ============================================================================
DEBUG_MODE = False             # Enable additional debug information
SAVE_SNAPSHOTS = False         # Save frames with detections
SNAPSHOT_INTERVAL = 30         # Save snapshot every N seconds
VERBOSE = False                # YOLO verbose output

# ============================================================================
# PERFORMANCE OPTIMIZATION - RTX 3050 (4GB VRAM)
# ============================================================================
USE_GPU = True                 # Use GPU if available (CUDA) - ✅ ENABLED
HALF_PRECISION = True          # Use FP16 for 2x speed boost on RTX 3050 - ✅ RECOMMENDED
MAX_DET = 300                  # Maximum number of detections per image
