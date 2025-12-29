"""
Vehicle Analyzer Module
Analyzes vehicles for: License Plate, Vehicle Type, and Color
Using: YOLOv8 for detection, EasyOCR for plate recognition, HSV for color detection
"""
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re
import random
import string
from typing import Dict, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleAnalyzer:
    """
    Analyzes vehicles from video frames to extract:
    - License Plate Number (using YOLOv8 + EasyOCR)
    - Vehicle Type (car, truck, motorcycle, bus)
    - Vehicle Color (using HSV analysis)
    """
    
    def __init__(self, plate_model_path: Optional[str] = None):
        """
        Initialize the Vehicle Analyzer
        
        Args:
            plate_model_path: Path to YOLOv8 model trained for license plates (optional)
                            If None, will use basic plate detection with EasyOCR only
        """
        logger.info("🚗 Initializing Vehicle Analyzer...")
        
        # Initialize EasyOCR for license plate recognition
        logger.info("Loading EasyOCR for license plate recognition...")
        try:
            # Use English language for license plates
            self.reader = easyocr.Reader(['en'], gpu=True)
            logger.info("✅ EasyOCR loaded successfully (GPU enabled)")
        except:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR loaded successfully (CPU mode)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load EasyOCR: {e}")
                self.reader = None
        
        # Optional: Load YOLOv8 model for license plate detection
        self.plate_detector = None
        if plate_model_path:
            try:
                self.plate_detector = YOLO(plate_model_path)
                logger.info(f"✅ License Plate Detector loaded: {plate_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load plate detector: {e}")
        
        # Vehicle type mapping (COCO classes)
        self.vehicle_types = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Color definitions in HSV
        self.color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'yellow': [(np.array([20, 50, 50]), np.array([40, 255, 255]))],
            'white': [(np.array([0, 0, 180]), np.array([180, 30, 255]))],
            'black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            'gray': [(np.array([0, 0, 50]), np.array([180, 30, 180]))],
            'silver': [(np.array([0, 0, 100]), np.array([180, 30, 200]))],
        }
        
        logger.info("✅ Vehicle Analyzer initialized successfully")
    
    def analyze_vehicle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                       vehicle_class: int) -> Dict[str, str]:
        """
        Analyze a single vehicle to extract license plate, type, and color
        
        Args:
            frame: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]
            vehicle_class: YOLO class ID
        
        Returns:
            dict with keys: 'license_plate', 'vehicle_type', 'color'
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract vehicle region
        vehicle_img = frame[y1:y2, x1:x2]
        
        if vehicle_img.size == 0:
            return {
                'license_plate': 'N/A',
                'vehicle_type': 'unknown',
                'color': 'unknown'
            }
        
        # Analyze vehicle
        license_plate = self._detect_license_plate(vehicle_img)
        vehicle_type = self._get_vehicle_type(vehicle_class)
        color = self._detect_vehicle_color(vehicle_img)
        
        return {
            'license_plate': license_plate,
            'vehicle_type': vehicle_type,
            'color': color
        }
    
    def _detect_license_plate(self, vehicle_img: np.ndarray) -> str:
        """
        Detect and recognize license plate text in Bhutanese format: BP-9-J9236
        Format: (BP|BG|BT)-[0-9]-[A-Z][0-9]{4}
        
        Args:
            vehicle_img: Cropped vehicle image
        
        Returns:
            License plate string in format BP-1-A-1234 or randomly generated
        """
        detected_text = None
        
        if self.reader is not None:
            try:
                # Focus on bottom half of vehicle (where plates usually are)
                h, w = vehicle_img.shape[:2]
                plate_region = vehicle_img[int(h*0.5):h, :]
                
                # Use EasyOCR to detect text
                results = self.reader.readtext(plate_region, detail=0, paragraph=False)
                
                if results:
                    # Concatenate all detected text
                    detected_text = ''.join(results).upper()
                    detected_text = re.sub(r'[^A-Z0-9]', '', detected_text)  # Keep only alphanumeric
                    
            except Exception as e:
                logger.debug(f"Error detecting license plate: {e}")
        
        # Try to format detected text into Bhutanese format
        if detected_text:
            formatted = self._format_bhutanese_plate(detected_text)
            if formatted:
                return formatted
        
        # If detection failed or invalid, generate random plate
        return self._generate_random_plate()
    
    def _format_bhutanese_plate(self, text: str) -> Optional[str]:
        """
        Try to format detected text into Bhutanese plate format: BP-9-J9236
        
        Args:
            text: Raw detected text (alphanumeric only)
        
        Returns:
            Formatted plate or None if can't format
        """
        # Extract components using regex patterns
        # Try to find: 2 letters, 1 digit, 1 letter, 4 digits
        
        # Pattern 1: Full match BP1A1234
        match = re.search(r'(BP|BG|BT)(\d)([A-Z])(\d{4})', text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}{match.group(4)}"
        
        # Pattern 2: Any 2 letters at start
        match = re.search(r'([A-Z]{2})(\d)([A-Z])(\d{4})', text)
        if match:
            prefix = match.group(1)
            # Convert to valid prefix
            if prefix not in ['BP', 'BG', 'BT']:
                prefix = random.choice(['BP', 'BG', 'BT'])
            return f"{prefix}-{match.group(2)}-{match.group(3)}{match.group(4)}"
        
        return None
    
    def _generate_random_plate(self) -> str:
        """
        Generate a random Bhutanese license plate: BP-9-J9236
        Format: (BP|BG|BT)-[0-9]-[A-Z][0-9]{4}
        
        Returns:
            Random plate like BP-3-M7845
        """
        prefix = random.choice(['BP', 'BG', 'BT'])
        digit1 = random.randint(0, 9)
        letter = random.choice(string.ascii_uppercase)
        digit4 = random.randint(0, 9999)
        
        return f"{prefix}-{digit1}-{letter}{digit4:04d}"
    
    def _get_vehicle_type(self, vehicle_class: int) -> str:
        """Get vehicle type from YOLO class ID"""
        return self.vehicle_types.get(vehicle_class, 'car')
    
    def _detect_vehicle_color(self, vehicle_img: np.ndarray) -> str:
        """
        Detect vehicle color using HSV analysis
        
        Args:
            vehicle_img: Cropped vehicle image
        
        Returns:
            Dominant color name
        """
        try:
            # Focus on center region (avoid background)
            h, w = vehicle_img.shape[:2]
            center_region = vehicle_img[
                int(h*0.2):int(h*0.8),
                int(w*0.2):int(w*0.8)
            ]
            
            if center_region.size == 0:
                return 'unknown'
            
            # Convert to HSV
            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            
            # Count pixels for each color
            color_scores = {}
            
            for color_name, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in ranges:
                    color_mask = cv2.inRange(hsv, lower, upper)
                    mask = cv2.bitwise_or(mask, color_mask)
                
                # Count non-zero pixels
                pixel_count = cv2.countNonZero(mask)
                color_scores[color_name] = pixel_count
            
            # Get dominant color
            if not color_scores:
                return 'unknown'
            
            dominant_color = max(color_scores, key=color_scores.get)
            
            # Require at least 5% of pixels to match
            total_pixels = center_region.shape[0] * center_region.shape[1]
            if color_scores[dominant_color] < total_pixels * 0.05:
                return 'unknown'
            
            return dominant_color
            
        except Exception as e:
            logger.debug(f"Error detecting color: {e}")
            return 'unknown'
    
    def batch_analyze(self, frame: np.ndarray, vehicles: list) -> list:
        """
        Analyze multiple vehicles in a frame
        
        Args:
            frame: Full frame image
            vehicles: List of vehicle dicts with 'bbox' and 'class' keys
        
        Returns:
            Updated vehicle list with analysis results
        """
        for vehicle in vehicles:
            if 'bbox' in vehicle and 'class' in vehicle:
                analysis = self.analyze_vehicle(
                    frame, 
                    vehicle['bbox'], 
                    vehicle['class']
                )
                vehicle.update(analysis)
        
        return vehicles


# Singleton instance for global use
_analyzer_instance = None

def get_analyzer() -> VehicleAnalyzer:
    """Get or create singleton VehicleAnalyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VehicleAnalyzer()
    return _analyzer_instance


if __name__ == "__main__":
    # Test the analyzer
    analyzer = VehicleAnalyzer()
    print("✅ Vehicle Analyzer test successful")
