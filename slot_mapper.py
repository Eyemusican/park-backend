"""
Smart Parking MVP - Parking Slot Mapper
Interactive tool to define parking slot polygons
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path


class ParkingSlotMapper:
    """Interactive tool for defining parking slot polygons"""
    
    def __init__(self, video_path, output_json='configs/parking_slots.json'):
        self.video_path = video_path
        self.output_json = output_json
        
        # Load reference frame
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read video: {video_path}")
        
        # Get original dimensions
        self.original_height, self.original_width = self.frame.shape[:2]
        
        # Calculate display size (fit to screen: max 1280x720)
        max_width = 1280
        max_height = 720
        
        scale_w = max_width / self.original_width
        scale_h = max_height / self.original_height
        self.display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        self.display_width = int(self.original_width * self.display_scale)
        self.display_height = int(self.original_height * self.display_scale)
        
        # Resize frame for display
        self.display_frame = cv2.resize(self.frame, (self.display_width, self.display_height))
        
        # Slot data
        self.slots = []
        self.current_points = []
        
        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Colors
        self.color_complete = (0, 255, 0)  # GREEN
        self.color_incomplete = (0, 255, 255)  # YELLOW
        self.color_point = (255, 0, 0)  # BLUE
        
        # Window name
        self.window_name = 'Parking Slot Mapper'
        
        print("="*60)
        print("PARKING SLOT MAPPER")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Original size: {self.original_width}x{self.original_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        print(f"Scale: {self.display_scale:.2f}x")
        print(f"Output: {output_json}")
        print("\nInstructions:")
        print("  - Click 4 corners to define a parking slot polygon")
        print("  - Mouse wheel UP/DOWN to ZOOM in/out")
        print("  - Right-click + drag to PAN when zoomed")
        print("  - Press 'c' to complete current slot")
        print("  - Press 'u' to undo last point")
        print("  - Press 'r' to remove last slot")
        print("  - Press 's' to save and quit")
        print("  - Press 'q' to quit without saving")
        print("  - Press 'z' to reset zoom")
        print("="*60 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to original coordinates (accounting for zoom/pan)
            orig_x = int((x - self.pan_x) / (self.display_scale * self.zoom_level))
            orig_y = int((y - self.pan_y) / (self.display_scale * self.zoom_level))
            
            # Add point (in original coordinates)
            if len(self.current_points) < 4:
                self.current_points.append([orig_x, orig_y])
                print(f"[OK] Point {len(self.current_points)}/4: ({orig_x}, {orig_y})")
                
                # Auto-complete if 4 points
                if len(self.current_points) == 4:
                    self.complete_slot()
            
            self.update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Start panning
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
        
        elif event == cv2.EVENT_RBUTTONUP:
            # Stop panning
            self.is_panning = False
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Pan the view
            if self.is_panning:
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.pan_x += dx
                self.pan_y += dy
                self.pan_start_x = x
                self.pan_start_y = y
                self.update_display()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel
            old_zoom = self.zoom_level
            
            if flags > 0:  # Scroll up - zoom in
                self.zoom_level *= 1.2
            else:  # Scroll down - zoom out
                self.zoom_level /= 1.2
            
            # Limit zoom range
            self.zoom_level = max(0.5, min(self.zoom_level, 5.0))
            
            # Adjust pan to zoom towards mouse position
            zoom_ratio = self.zoom_level / old_zoom
            self.pan_x = x - (x - self.pan_x) * zoom_ratio
            self.pan_y = y - (y - self.pan_y) * zoom_ratio
            
            print(f"[ZOOM] {self.zoom_level:.2f}x")
            self.update_display()
    
    def complete_slot(self):
        """Complete current slot and add to slots list"""
        if len(self.current_points) == 4:
            slot_id = len(self.slots) + 1
            slot = {
                'id': slot_id,
                'name': f'Slot-{slot_id}',
                'points': self.current_points.copy()
            }
            self.slots.append(slot)
            print(f"\n[COMPLETED] Slot {slot_id} added")
            print(f"Total slots: {len(self.slots)}\n")
            
            # Reset for next slot
            self.current_points = []
    
    def undo_point(self):
        """Remove last point from current slot"""
        if self.current_points:
            removed = self.current_points.pop()
            print(f"[UNDO] Removed point: {removed}")
            print(f"Current points: {len(self.current_points)}/4\n")
            self.update_display()
    
    def remove_last_slot(self):
        """Remove last completed slot"""
        if self.slots:
            removed = self.slots.pop()
            print(f"[REMOVED] Slot {removed['id']}")
            print(f"Total slots: {len(self.slots)}\n")
            self.update_display()
    
    def update_display(self):
        """Update display with current slots and points"""
        # Create zoomed and panned view
        zoomed_width = int(self.display_width * self.zoom_level)
        zoomed_height = int(self.display_height * self.zoom_level)
        
        # Resize base frame to zoomed size
        zoomed_frame = cv2.resize(self.frame, (zoomed_width, zoomed_height))
        
        # Calculate combined scale
        combined_scale = self.display_scale * self.zoom_level
        
        # Draw completed slots (GREEN)
        for slot in self.slots:
            points = slot['points']
            # Convert to zoomed display coordinates
            pts = [(int(p[0] * combined_scale), int(p[1] * combined_scale)) for p in points]
            
            # Draw polygon
            for i in range(4):
                cv2.line(zoomed_frame, pts[i], pts[(i+1)%4],
                        self.color_complete, 2)
            
            # Fill with transparency
            overlay = zoomed_frame.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)],
                        self.color_complete)
            cv2.addWeighted(overlay, 0.2, zoomed_frame, 0.8, 0, zoomed_frame)
            
            # Draw slot ID
            center_x = int(sum(p[0] for p in points) / 4 * combined_scale)
            center_y = int(sum(p[1] for p in points) / 4 * combined_scale)
            cv2.putText(zoomed_frame, f"#{slot['id']}", 
                       (center_x-20, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0 * self.zoom_level, (255, 255, 255), 
                       max(2, int(2 * self.zoom_level)))
        
        # Draw current incomplete slot (YELLOW)
        if len(self.current_points) > 1:
            pts = [(int(p[0] * combined_scale), int(p[1] * combined_scale)) for p in self.current_points]
            for i in range(len(pts)-1):
                cv2.line(zoomed_frame, pts[i], pts[i+1],
                        self.color_incomplete, max(2, int(2 * self.zoom_level)))
        
        # Draw current points (BLUE)
        for i, point in enumerate(self.current_points):
            disp_x = int(point[0] * combined_scale)
            disp_y = int(point[1] * combined_scale)
            cv2.circle(zoomed_frame, (disp_x, disp_y),
                      max(6, int(6 * self.zoom_level)), self.color_point, -1)
            cv2.putText(zoomed_frame, str(i+1), 
                       (disp_x+10, disp_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.zoom_level, (255, 255, 255), 
                       max(2, int(2 * self.zoom_level)))
        
        # Extract visible region based on pan
        pan_x_int = int(self.pan_x)
        pan_y_int = int(self.pan_y)
        
        # Clamp pan values
        max_pan_x = max(0, zoomed_width - self.display_width)
        max_pan_y = max(0, zoomed_height - self.display_height)
        pan_x_int = max(-self.display_width + 100, min(pan_x_int, max_pan_x))
        pan_y_int = max(-self.display_height + 100, min(pan_y_int, max_pan_y))
        
        # Create display frame
        self.display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Calculate source region
        src_x1 = max(0, -pan_x_int)
        src_y1 = max(0, -pan_y_int)
        src_x2 = min(zoomed_width, src_x1 + self.display_width)
        src_y2 = min(zoomed_height, src_y1 + self.display_height)
        
        # Calculate destination region
        dst_x1 = max(0, pan_x_int)
        dst_y1 = max(0, pan_y_int)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Copy visible region
        if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 <= self.display_width and dst_y2 <= self.display_height:
            self.display_frame[dst_y1:dst_y2, dst_x1:dst_x2] = zoomed_frame[src_y1:src_y2, src_x1:src_x2]
        
        # Draw instructions overlay - COMPACT VERSION (LEFT HALF ONLY)
        instructions = [
            f"Slots: {len(self.slots)} | Points: {len(self.current_points)}/4 | Zoom: {self.zoom_level:.1f}x",
            "Wheel/+/- =zoom  Arrows=pan  Z=reset  C=complete  U=undo  R=remove  S=save"
        ]
        
        # Compact instructions background (left half of screen only)
        box_width = int(self.display_width * 0.5)  # Only 50% of screen width
        cv2.rectangle(self.display_frame, (10, 5), (box_width, 55),
                     (0, 0, 0), -1)
        
        # Instructions text (smaller and closer together)
        y_offset = 20
        for instruction in instructions:
            cv2.putText(self.display_frame, instruction, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_offset += 22
        
        # Show zoom indicator when zoomed
        if self.zoom_level > 1.1:
            zoom_text = f"ZOOMED: {self.zoom_level:.1f}x - Use arrow keys to pan"
            cv2.putText(self.display_frame, zoom_text, (20, self.display_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show
        cv2.imshow(self.window_name, self.display_frame)
    
    def save_slots(self):
        """Save slots to JSON file"""
        # Create output directory if not exists
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        
        # Save
        with open(self.output_json, 'w') as f:
            json.dump({'slots': self.slots}, f, indent=2)
        
        print("\n" + "="*60)
        print("SAVED PARKING SLOTS")
        print("="*60)
        print(f"File: {self.output_json}")
        print(f"Total slots: {len(self.slots)}")
        for slot in self.slots:
            print(f"  Slot {slot['id']}: {slot['name']}")
        print("="*60)
    
    def run(self):
        """Run the mapper tool"""
        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit without saving
                print("\n[QUIT] Exiting without saving")
                break
            
            elif key == ord('s'):
                # Save and quit
                if self.slots:
                    self.save_slots()
                    print("\n[SUCCESS] Slots saved! You can now run the complete system.")
                else:
                    print("\n[WARNING] No slots defined. Nothing saved.")
                break
            
            elif key == ord('c'):
                # Complete current slot
                if len(self.current_points) == 4:
                    self.complete_slot()
                    self.update_display()
                else:
                    print(f"[WARNING] Need 4 points (have {len(self.current_points)})")
            
            elif key == ord('u'):
                # Undo last point
                self.undo_point()
            
            elif key == ord('r'):
                # Remove last slot
                self.remove_last_slot()
            
            elif key == ord('z'):
                # Reset zoom and pan
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                print("[RESET] Zoom and pan reset")
                self.update_display()
            
            elif key == ord('+') or key == ord('='):
                # Zoom in with keyboard
                self.zoom_level *= 1.3
                self.zoom_level = min(self.zoom_level, 5.0)
                print(f"[ZOOM IN] {self.zoom_level:.2f}x")
                self.update_display()
            
            elif key == ord('-') or key == ord('_'):
                # Zoom out with keyboard
                self.zoom_level /= 1.3
                self.zoom_level = max(self.zoom_level, 0.5)
                print(f"[ZOOM OUT] {self.zoom_level:.2f}x")
                self.update_display()
            
            elif key == 0 or key == 2490368:  # Up arrow
                self.pan_y += 50
                print("[PAN] Up")
                self.update_display()
            
            elif key == 1 or key == 2621440:  # Down arrow
                self.pan_y -= 50
                print("[PAN] Down")
                self.update_display()
            
            elif key == 2 or key == 2424832:  # Left arrow
                self.pan_x += 50
                print("[PAN] Left")
                self.update_display()
            
            elif key == 3 or key == 2555904:  # Right arrow
                self.pan_x -= 50
                print("[PAN] Right")
                self.update_display()
        
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import sys
    
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'parking_video.mp4.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        print("\nUsage: python slot_mapper.py <video_path>")
        return
    
    # Get output path
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
    else:
        output_json = 'configs/parking_slots.json'
    
    # Run mapper
    try:
        mapper = ParkingSlotMapper(video_path, output_json)
        mapper.run()
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
