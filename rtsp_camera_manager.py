"""
Smart Parking System - RTSP Camera Manager
Multi-camera streaming with connection resilience and health monitoring
"""
import cv2
import os
import time
import threading
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Set RTSP transport options before any VideoCapture operations
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000"

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera connection status"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    FALLBACK = "FALLBACK"  # Using fallback video source


@dataclass
class CameraHealth:
    """Health metrics for a camera"""
    status: CameraStatus = CameraStatus.DISCONNECTED
    fps: float = 0.0
    frame_latency_ms: float = 0.0
    last_frame_time: Optional[datetime] = None
    last_error: Optional[str] = None
    reconnect_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class CameraConfig:
    """Configuration for an RTSP camera"""
    camera_id: int
    name: str
    rtsp_url: str
    parking_area_id: int
    username: Optional[str] = None
    password: Optional[str] = None
    buffer_size: int = 1
    timeout_seconds: int = 10
    retry_interval_seconds: int = 5
    max_retries: int = 3
    is_active: bool = True
    fallback_video: Optional[str] = None  # Path to fallback video file

    def get_authenticated_url(self) -> str:
        """Get RTSP URL with embedded credentials if provided"""
        if self.username and self.password:
            # Parse URL and insert credentials
            if "://" in self.rtsp_url:
                protocol, rest = self.rtsp_url.split("://", 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.rtsp_url


class RTSPCamera:
    """
    Single RTSP camera with connection management

    Features:
    - Thread-safe frame access
    - Automatic reconnection with exponential backoff
    - Automatic fallback to video file when RTSP unavailable
    - Health monitoring
    - Configurable buffer size and timeout
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._current_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._using_fallback = False  # Track if using fallback video

        # Health tracking
        self.health = CameraHealth()
        self._connected_at: Optional[datetime] = None
        self._frame_count = 0
        self._fps_start_time = time.time()

        logger.info(f"RTSPCamera initialized: {config.name} (ID: {config.camera_id})")

    def connect(self) -> bool:
        """
        Establish RTSP connection with OpenCV

        Returns:
            bool: True if connection successful
        """
        if self._running:
            logger.warning(f"Camera {self.config.name} already running")
            return True

        self.health.status = CameraStatus.CONNECTING
        logger.info(f"Connecting to camera: {self.config.name}")

        try:
            # Create VideoCapture with RTSP URL
            url = self.config.get_authenticated_url()
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                raise ConnectionError(f"Failed to open RTSP stream: {self.config.rtsp_url}")

            # Configure capture settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            # Try to set timeout (may not be supported on all backends)
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.timeout_seconds * 1000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.timeout_seconds * 1000)
            except:
                pass  # Timeout settings not supported

            # Verify we can read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ConnectionError("Failed to read initial frame from stream")

            # Connection successful
            self.health.status = CameraStatus.CONNECTED
            self._connected_at = datetime.now()
            self._running = True

            # Store initial frame
            with self._frame_lock:
                self._current_frame = frame.copy()
                self.health.last_frame_time = datetime.now()

            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"Camera-{self.config.camera_id}",
                daemon=True
            )
            self._capture_thread.start()

            logger.info(f"Successfully connected to camera: {self.config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to camera {self.config.name}: {e}")
            self._cleanup_capture()

            # Try fallback video if RTSP connection failed
            if self.config.fallback_video:
                logger.info(f"Camera {self.config.name}: Trying fallback video...")
                if self._try_fallback_video():
                    self._running = True
                    # Start capture thread for fallback video
                    self._capture_thread = threading.Thread(
                        target=self._capture_loop,
                        name=f"Camera-{self.config.camera_id}",
                        daemon=True
                    )
                    self._capture_thread.start()
                    return True

            self.health.status = CameraStatus.ERROR
            self.health.last_error = str(e)
            return False

    def disconnect(self) -> None:
        """Graceful shutdown of camera connection"""
        logger.info(f"Disconnecting camera: {self.config.name}")
        self._running = False

        # Wait for capture thread to stop
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        self._cleanup_capture()
        self.health.status = CameraStatus.DISCONNECTED
        logger.info(f"Camera disconnected: {self.config.name}")

    def get_frame(self) -> Tuple[bool, Optional[Any]]:
        """
        Thread-safe frame retrieval

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        with self._frame_lock:
            if self._current_frame is None:
                return False, None
            return True, self._current_frame.copy()

    def health_check(self) -> CameraHealth:
        """Get current health metrics"""
        if self._connected_at:
            self.health.uptime_seconds = (datetime.now() - self._connected_at).total_seconds()
        return self.health

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread"""
        consecutive_failures = 0

        while self._running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._handle_reconnect()
                    continue

                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Success - update frame and metrics
                    with self._frame_lock:
                        self._current_frame = frame.copy()
                        self.health.last_frame_time = datetime.now()

                    consecutive_failures = 0
                    self._update_fps()

                else:
                    # Failed to read frame
                    consecutive_failures += 1

                    # If using fallback video, try to loop it
                    if self._using_fallback:
                        self._handle_video_loop()
                        consecutive_failures = 0  # Reset since we're looping
                        continue

                    logger.warning(f"Camera {self.config.name}: Failed to read frame ({consecutive_failures})")

                    if consecutive_failures >= 5:
                        self._handle_reconnect()
                        consecutive_failures = 0

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Camera {self.config.name} capture error: {e}")
                self.health.last_error = str(e)
                consecutive_failures += 1

                if consecutive_failures >= 5:
                    self._handle_reconnect()
                    consecutive_failures = 0

    def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff, then fallback to video"""
        self.health.status = CameraStatus.RECONNECTING
        self.health.reconnect_count += 1

        for attempt in range(self.config.max_retries):
            if not self._running:
                return

            # Exponential backoff
            wait_time = self.config.retry_interval_seconds * (2 ** attempt)
            logger.info(f"Camera {self.config.name}: Reconnecting in {wait_time}s (attempt {attempt + 1}/{self.config.max_retries})")
            time.sleep(wait_time)

            if not self._running:
                return

            # Cleanup old capture
            self._cleanup_capture()

            # Try to reconnect
            try:
                url = self.config.get_authenticated_url()
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                    ret, frame = self.cap.read()

                    if ret and frame is not None:
                        self.health.status = CameraStatus.CONNECTED
                        self._connected_at = datetime.now()
                        self._using_fallback = False

                        with self._frame_lock:
                            self._current_frame = frame.copy()
                            self.health.last_frame_time = datetime.now()

                        logger.info(f"Camera {self.config.name}: Reconnected successfully")
                        return

            except Exception as e:
                logger.error(f"Camera {self.config.name}: Reconnect attempt failed: {e}")

        # All retries exhausted - try fallback video if configured
        if self.config.fallback_video and self._try_fallback_video():
            return

        # No fallback available or fallback failed
        self.health.status = CameraStatus.ERROR
        self.health.last_error = "Max reconnection attempts exceeded and no fallback available"
        logger.error(f"Camera {self.config.name}: Failed to reconnect after {self.config.max_retries} attempts")

    def _try_fallback_video(self) -> bool:
        """
        Try to use fallback video file when RTSP is unavailable

        Returns:
            bool: True if fallback video opened successfully
        """
        if not self.config.fallback_video:
            return False

        # Resolve fallback path relative to park-backend directory
        fallback_path = self.config.fallback_video
        if not os.path.isabs(fallback_path):
            # Get directory of this file and resolve relative path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.normpath(os.path.join(base_dir, fallback_path))

        if not os.path.exists(fallback_path):
            logger.error(f"Camera {self.config.name}: Fallback video not found: {fallback_path}")
            return False

        self._cleanup_capture()

        try:
            self.cap = cv2.VideoCapture(fallback_path)

            if not self.cap.isOpened():
                logger.error(f"Camera {self.config.name}: Failed to open fallback video: {fallback_path}")
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Camera {self.config.name}: Failed to read frame from fallback video")
                return False

            self._using_fallback = True
            self.health.status = CameraStatus.FALLBACK
            self._connected_at = datetime.now()
            self.health.last_error = f"Using fallback video: {os.path.basename(fallback_path)}"

            with self._frame_lock:
                self._current_frame = frame.copy()
                self.health.last_frame_time = datetime.now()

            logger.info(f"Camera {self.config.name}: Switched to fallback video: {fallback_path}")
            return True

        except Exception as e:
            logger.error(f"Camera {self.config.name}: Failed to open fallback video: {e}")
            return False

    def _handle_video_loop(self) -> None:
        """Handle video file looping when using fallback"""
        if self._using_fallback and self.cap is not None:
            # Reset video to beginning when it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.debug(f"Camera {self.config.name}: Looping fallback video")

    def _update_fps(self) -> None:
        """Update FPS calculation"""
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time

        if elapsed >= 1.0:
            self.health.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

    def _cleanup_capture(self) -> None:
        """Release VideoCapture resources"""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None


class RTSPCameraManager:
    """
    Manages multiple RTSP cameras

    Features:
    - Add/remove cameras dynamically
    - Get frames from any camera
    - Monitor health of all cameras
    - Start/stop all cameras
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global camera manager"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cameras: Dict[int, RTSPCamera] = {}
        self._configs: Dict[int, CameraConfig] = {}
        self._cameras_lock = threading.Lock()
        self._initialized = True

        logger.info("RTSPCameraManager initialized")

    def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a new camera to the manager

        Args:
            config: Camera configuration

        Returns:
            bool: True if camera added successfully
        """
        with self._cameras_lock:
            if config.camera_id in self._cameras:
                logger.warning(f"Camera {config.camera_id} already exists")
                return False

            camera = RTSPCamera(config)
            self._cameras[config.camera_id] = camera
            self._configs[config.camera_id] = config

            logger.info(f"Camera added: {config.name} (ID: {config.camera_id})")

            # Auto-connect if active
            if config.is_active:
                return camera.connect()

            return True

    def remove_camera(self, camera_id: int) -> bool:
        """
        Remove a camera from the manager

        Args:
            camera_id: ID of camera to remove

        Returns:
            bool: True if camera removed
        """
        with self._cameras_lock:
            if camera_id not in self._cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False

            camera = self._cameras[camera_id]
            camera.disconnect()

            del self._cameras[camera_id]
            del self._configs[camera_id]

            logger.info(f"Camera removed: ID {camera_id}")
            return True

    def get_camera_frame(self, camera_id: int) -> Tuple[bool, Optional[Any]]:
        """
        Get the latest frame from a specific camera

        Args:
            camera_id: ID of the camera

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        with self._cameras_lock:
            if camera_id not in self._cameras:
                return False, None

            return self._cameras[camera_id].get_frame()

    def get_camera_health(self, camera_id: int) -> Optional[CameraHealth]:
        """
        Get health metrics for a specific camera

        Args:
            camera_id: ID of the camera

        Returns:
            CameraHealth or None if camera not found
        """
        with self._cameras_lock:
            if camera_id not in self._cameras:
                return None

            return self._cameras[camera_id].health_check()

    def get_all_statuses(self) -> Dict[int, Dict]:
        """
        Get status of all cameras

        Returns:
            Dict mapping camera_id to status info
        """
        statuses = {}

        with self._cameras_lock:
            for camera_id, camera in self._cameras.items():
                config = self._configs[camera_id]
                health = camera.health_check()

                statuses[camera_id] = {
                    "camera_id": camera_id,
                    "name": config.name,
                    "parking_area_id": config.parking_area_id,
                    "status": health.status.value,
                    "fps": round(health.fps, 1),
                    "last_frame_time": health.last_frame_time.isoformat() if health.last_frame_time else None,
                    "uptime_seconds": round(health.uptime_seconds, 1),
                    "reconnect_count": health.reconnect_count,
                    "last_error": health.last_error
                }

        return statuses

    def connect_camera(self, camera_id: int) -> bool:
        """Force connect a specific camera"""
        with self._cameras_lock:
            if camera_id not in self._cameras:
                return False
            return self._cameras[camera_id].connect()

    def disconnect_camera(self, camera_id: int) -> None:
        """Force disconnect a specific camera"""
        with self._cameras_lock:
            if camera_id in self._cameras:
                self._cameras[camera_id].disconnect()

    def start_all(self) -> None:
        """Start all active cameras"""
        with self._cameras_lock:
            for camera_id, camera in self._cameras.items():
                if self._configs[camera_id].is_active:
                    camera.connect()

    def stop_all(self) -> None:
        """Stop all cameras"""
        with self._cameras_lock:
            for camera in self._cameras.values():
                camera.disconnect()

    def get_camera_ids(self) -> list:
        """Get list of all camera IDs"""
        with self._cameras_lock:
            return list(self._cameras.keys())

    def camera_exists(self, camera_id: int) -> bool:
        """Check if camera exists"""
        with self._cameras_lock:
            return camera_id in self._cameras


# Global instance getter
def get_camera_manager() -> RTSPCameraManager:
    """Get the global camera manager instance"""
    return RTSPCameraManager()


def load_cameras_from_config(config_path: str = "configs/cameras.json") -> int:
    """
    Load cameras from JSON configuration file and add them to the manager.
    Automatically uses fallback videos when RTSP streams are unavailable.

    Args:
        config_path: Path to cameras.json config file

    Returns:
        int: Number of cameras successfully loaded
    """
    import json

    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        logger.error(f"Camera config file not found: {config_path}")
        return 0

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load camera config: {e}")
        return 0

    manager = get_camera_manager()
    loaded_count = 0

    # Load cameras from config
    for cam_data in config.get('cameras', []):
        cam_config = CameraConfig(
            camera_id=cam_data['camera_id'],
            name=cam_data['name'],
            rtsp_url=cam_data['rtsp_url'],
            parking_area_id=cam_data['parking_area_id'],
            username=cam_data.get('auth', {}).get('username'),
            password=cam_data.get('auth', {}).get('password'),
            buffer_size=cam_data.get('settings', {}).get('buffer_size', 1),
            timeout_seconds=cam_data.get('settings', {}).get('timeout_seconds', 10),
            retry_interval_seconds=cam_data.get('settings', {}).get('retry_interval_seconds', 5),
            max_retries=cam_data.get('settings', {}).get('max_retries', 3),
            is_active=cam_data.get('is_active', True),
            fallback_video=cam_data.get('fallback_video')
        )

        if manager.add_camera(cam_config):
            loaded_count += 1
            logger.info(f"Loaded camera: {cam_config.name} (parking_area: {cam_config.parking_area_id}, fallback: {cam_config.fallback_video})")

    # If global setting prefers file over RTSP, log it
    global_settings = config.get('global_settings', {})
    if not global_settings.get('prefer_rtsp_over_file', True):
        logger.info("Global setting: prefer_rtsp_over_file=false, fallback videos will be used when RTSP unavailable")

    return loaded_count


def get_fallback_for_parking_area(parking_area_id: int, config_path: str = "configs/cameras.json") -> Optional[str]:
    """
    Get the fallback video path for a specific parking area.

    Args:
        parking_area_id: ID of the parking area
        config_path: Path to cameras.json config file

    Returns:
        str or None: Path to fallback video, or None if not found
    """
    import json

    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # First check cameras for fallback_video
        for cam_data in config.get('cameras', []):
            if cam_data.get('parking_area_id') == parking_area_id:
                fallback = cam_data.get('fallback_video')
                if fallback:
                    base_dir = os.path.dirname(os.path.abspath(config_path))
                    return os.path.normpath(os.path.join(base_dir, fallback))

        # Then check fallback_sources
        for source in config.get('fallback_sources', []):
            if source.get('parking_area_id') == parking_area_id:
                fallback = source.get('source')
                if fallback:
                    base_dir = os.path.dirname(os.path.abspath(config_path))
                    return os.path.normpath(os.path.join(base_dir, fallback))

        # Return default fallback if exists
        for source in config.get('fallback_sources', []):
            if source.get('is_default'):
                fallback = source.get('source')
                if fallback:
                    base_dir = os.path.dirname(os.path.abspath(config_path))
                    return os.path.normpath(os.path.join(base_dir, fallback))

    except Exception as e:
        logger.error(f"Failed to get fallback for parking area {parking_area_id}: {e}")

    return None


# Convenience function for backward compatibility with existing video sources
def create_video_capture(source) -> Tuple[cv2.VideoCapture, bool]:
    """
    Create a VideoCapture for any source type (RTSP, file, webcam)

    Args:
        source: RTSP URL string, file path string, or camera index int

    Returns:
        Tuple[cv2.VideoCapture, bool]: (capture object, is_rtsp)
    """
    is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")

    if is_rtsp:
        # Use RTSP-optimized settings
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # Standard capture for files/webcams
        cap = cv2.VideoCapture(source)

    return cap, is_rtsp
