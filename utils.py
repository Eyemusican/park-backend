"""
Smart Parking System - Phase 2: Visualization Utilities (ENHANCED)
Drawing functions with clear color coding: GREEN=moving, YELLOW=stopped, RED=parked, GRAY=occluded
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import config


def get_vehicle_color(tracker) -> Tuple[int, int, int]:
    """
    Get consistent color based on vehicle state
    
    This fixes Issue #4: Clear color coding
    
    Args:
        tracker: VehicleTracker instance
        
    Returns:
        BGR color tuple
    """
    if tracker.occluded:
        return config.COLOR_OCCLUDED  # GRAY for occluded vehicles
    elif tracker.status == "PARKED":
        return config.COLOR_PARKED    # RED for parked (>1.5s stationary)
    elif tracker.status == "STOPPED":
        return config.COLOR_STOPPED   # YELLOW for temporarily stopped
    else:  # MOVING
        return config.COLOR_MOVING    # GREEN for moving vehicles


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h{minutes}m"


def draw_detections(frame: np.ndarray, detections: List[Dict], 
                    tracking_manager=None) -> np.ndarray:
    """
    Draw bounding boxes, labels, track IDs, and motion trails with clear color coding
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        tracking_manager: TrackingManager instance for motion trails and status
        
    Returns:
        Frame with drawn detections
    """
    output = frame.copy()
    
    # If we have tracking manager, use all trackers (includes occluded)
    if tracking_manager:
        all_trackers = tracking_manager.get_all_trackers()
        
        # Draw each tracked vehicle
        for track_id, tracker in all_trackers.items():
            # Get color based on vehicle state
            color = get_vehicle_color(tracker)
            bbox = tracker.current_bbox
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            thickness = 1 if tracker.occluded else 2  # Thin lines for occluded
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw motion trail if tracking is enabled and not occluded
            if config.SHOW_MOTION_TRAIL and not tracker.occluded:
                trail = tracker.get_trail()
                if len(trail) > 1:
                    # Draw trail as connected lines
                    for i in range(1, len(trail)):
                        pt1 = trail[i-1]
                        pt2 = trail[i]
                        # Fade trail (older points are more transparent)
                        alpha = i / len(trail)
                        thickness = max(1, int(config.TRAIL_THICKNESS * alpha))
                        cv2.line(output, pt1, pt2, config.TRAIL_COLOR, thickness)
            
            # Prepare label text
            label_parts = []
            
            if config.SHOW_TRACK_IDS:
                label_parts.append(f"ID:{track_id}")
            
            label_parts.append(tracker.class_name.upper())
            
            # Show status or confidence
            if tracker.occluded:
                label_parts.append("[OCCLUDED]")
            elif tracker.status == "PARKED":
                # Show parking duration
                duration_str = format_duration(tracker.get_park_duration())
                label_parts.append(f"PARKED {duration_str}")
            elif config.SHOW_MOTION_STATUS:
                label_parts.append(tracker.status)
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, 
                config.FONT, 
                config.FONT_SCALE, 
                1  # Thinner text
            )
            
            # Position label above bounding box
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            
            # Draw colored background for text
            cv2.rectangle(
                output,
                (x1, label_y - label_height - baseline),
                (x1 + label_width + 4, label_y + baseline),
                color,
                -1
            )
            
            # Draw text in white
            cv2.putText(
                output,
                label,
                (x1 + 2, label_y),
                config.FONT,
                config.FONT_SCALE,
                (255, 255, 255),
                1
            )
    else:
        # No tracking - simple detection boxes
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Use class-based color
            color = config.BBOX_COLORS.get(class_name, config.DEFAULT_BBOX_COLOR)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            
            # Prepare label
            if config.SHOW_CONFIDENCE:
                label = f"{class_name.upper()}: {confidence:.2f}"
            else:
                label = class_name.upper()
            
            # Draw label
            (label_width, label_height), baseline = cv2.getTextSize(
                label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
            )
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            
            cv2.rectangle(
                output,
                (x1, label_y - label_height - baseline),
                (x1 + label_width, label_y + baseline),
                config.TEXT_BG_COLOR,
                -1
            )
            
            cv2.putText(
                output, label, (x1, label_y),
                config.FONT, config.FONT_SCALE,
                config.TEXT_COLOR, config.FONT_THICKNESS
            )
    
    return output


def draw_stats(frame: np.ndarray, stats: Dict, vehicle_count: int) -> np.ndarray:
    """
    Draw statistics overlay on frame with enhanced tracking info
    
    Args:
        frame: Input frame
        stats: Statistics dictionary from detector
        vehicle_count: Current vehicle count
        
    Returns:
        Frame with stats overlay
    """
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Prepare stats text
    stats_lines = []
    
    if config.SHOW_FPS:
        stats_lines.append(f"FPS: {stats['fps']:.1f}")
    
    if config.SHOW_PROCESS_TIME:
        stats_lines.append(f"Process: {stats['process_time_ms']:.1f}ms")
    
    if config.SHOW_VEHICLE_COUNT:
        stats_lines.append(f"Detected: {vehicle_count}")
    
    # Add tracking stats if available
    if config.ENABLE_TRACKING and 'active_tracks' in stats:
        stats_lines.append(f"Active: {stats['active_tracks']}")
        
        if 'lost_tracks' in stats and stats['lost_tracks'] > 0:
            stats_lines.append(f"Occluded: {stats['lost_tracks']}")
        
        if 'moving_vehicles' in stats:
            stats_lines.append(f"Moving: {stats['moving_vehicles']}")
        
        if 'parked_vehicles' in stats and stats['parked_vehicles'] > 0:
            stats_lines.append(f"Parked: {stats['parked_vehicles']}")
    
    # Add color legend
    stats_lines.append("")
    stats_lines.append("Color Key:")
    stats_lines.append("  GREEN = Moving")
    stats_lines.append("  YELLOW = Stopped")
    stats_lines.append("  RED = Parked")
    stats_lines.append("  GRAY = Occluded")
    
    # Calculate stats panel dimensions
    line_height = 25
    panel_height = len(stats_lines) * line_height + 20
    panel_width = 280
    padding = 10
    
    # Draw semi-transparent background panel (top-left)
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding + panel_width, padding + panel_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Draw border
    cv2.rectangle(
        output,
        (padding, padding),
        (padding + panel_width, padding + panel_height),
        (0, 255, 0),
        2
    )
    
    # Draw stats text
    y_offset = padding + 20
    for i, line in enumerate(stats_lines):
        # Color for color key lines
        if "GREEN" in line:
            text_color = config.COLOR_MOVING
        elif "YELLOW" in line:
            text_color = config.COLOR_STOPPED
        elif "RED" in line:
            text_color = config.COLOR_PARKED
        elif "GRAY" in line:
            text_color = config.COLOR_OCCLUDED
        else:
            text_color = (255, 255, 255)
        
        cv2.putText(
            output,
            line,
            (padding + 10, y_offset),
            config.FONT,
            0.5,
            text_color,
            1
        )
        y_offset += line_height
    
    return output


def draw_header(frame: np.ndarray, title: str = None) -> np.ndarray:
    """
    Draw header title on frame
    
    Args:
        frame: Input frame
        title: Title text (auto-generated based on config if None)
        
    Returns:
        Frame with header
    """
    if title is None:
        if config.ENABLE_TRACKING:
            title = "Smart Parking - Phase 2: Vehicle Tracking"
        else:
            title = "Smart Parking - Phase 1: Vehicle Detection"
    
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        title,
        config.FONT,
        0.8,
        2
    )
    
    # Draw semi-transparent background at top
    header_height = text_height + 20
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Draw title text (centered)
    text_x = (w - text_width) // 2
    text_y = text_height + 10
    cv2.putText(
        output,
        title,
        (text_x, text_y),
        config.FONT,
        0.8,
        (0, 255, 0),
        2
    )
    
    return output


def create_visualization(frame: np.ndarray, detections: List[Dict], 
                        stats: Dict, vehicle_count: int,
                        tracking_manager=None,
                        show_header: bool = True) -> np.ndarray:
    """
    Create complete visualization with detections, tracking, and stats
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        stats: Statistics dictionary
        vehicle_count: Current vehicle count
        tracking_manager: TrackingManager instance for motion trails
        show_header: Whether to show header title
        
    Returns:
        Fully annotated frame
    """
    # Draw detections and tracking
    output = draw_detections(frame, detections, tracking_manager)
    
    # Draw stats overlay
    output = draw_stats(output, stats, vehicle_count)
    
    # Draw header
    if show_header:
        output = draw_header(output)
    
    return output


def resize_for_display(frame: np.ndarray, 
                       max_width: int = None, 
                       max_height: int = None) -> np.ndarray:
    """
    Resize frame for display while maintaining aspect ratio
    
    Args:
        frame: Input frame
        max_width: Maximum width (defaults to config.DISPLAY_WIDTH)
        max_height: Maximum height (defaults to config.DISPLAY_HEIGHT)
        
    Returns:
        Resized frame
    """
    max_width = max_width or config.DISPLAY_WIDTH
    max_height = max_height or config.DISPLAY_HEIGHT
    
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame


def save_snapshot(frame: np.ndarray, filepath: str) -> bool:
    """
    Save frame snapshot to file
    
    Args:
        frame: Frame to save
        filepath: Output file path
        
    Returns:
        True if saved successfully
    """
    try:
        cv2.imwrite(str(filepath), frame)
        return True
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return False


def draw_performance_warning(frame: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Draw performance warning if system is underperforming
    
    Args:
        frame: Input frame
        stats: Statistics dictionary
        
    Returns:
        Frame with warning if needed
    """
    output = frame.copy()
    
    # Check performance thresholds
    fps_warning = stats['fps'] < config.TARGET_FPS
    process_warning = stats['process_time_ms'] > config.MAX_PROCESS_TIME_MS
    
    if fps_warning or process_warning:
        h, w = output.shape[:2]
        warning_text = "!!! PERFORMANCE WARNING"
        
        # Draw warning banner at bottom
        banner_height = 40
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (0, h - banner_height),
            (w, h),
            (0, 0, 255),
            -1
        )
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        
        # Draw warning text
        (text_width, text_height), _ = cv2.getTextSize(
            warning_text,
            config.FONT,
            0.7,
            2
        )
        text_x = (w - text_width) // 2
        text_y = h - banner_height // 2 + text_height // 2
        
        cv2.putText(
            output,
            warning_text,
            (text_x, text_y),
            config.FONT,
            0.7,
            (255, 255, 255),
            2
        )
    
    return output
