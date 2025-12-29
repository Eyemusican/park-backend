"""
Smart Parking MVP - Simple Slot Mapper
Click area to zoom, then define slots
"""
import cv2
import numpy as np
import json
import os


class SimpleSlotMapper:
    """Simple slot mapper with click-to-zoom"""
    
    def __init__(self, video_path, output_json='configs/parking_slots.json'):
        self.video_path = video_path
        self.output_json = output_json
        
        # Load frame
        cap = cv2.VideoCapture(video_path)
        ret, self.original_frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read video: {video_path}")
        
        self.orig_h, self.orig_w = self.original_frame.shape[:2]
        
        # Display settings
        self.display_w = 1280
        self.display_h = 720
        self.frame = cv2.resize(self.original_frame, (self.display_w, self.display_h))
        
        # Zoom state
        self.mode = 'SELECT_AREA'  # SELECT_AREA or DEFINE_SLOTS
        self.zoom_region = None  # (x1, y1, x2, y2) in original coords
        self.zoom_frame = None
        
        # Slot data - load existing if available
        self.slots = []
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f:
                    existing_data = json.load(f)
                    self.slots = existing_data.get('slots', [])
                print(f"[LOADED] {len(self.slots)} existing slots from {output_json}")
            except:
                print(f"[INFO] Starting fresh (could not load existing slots)")
        else:
            print(f"[INFO] No existing slots found. Starting fresh.")
        
        self.current_points = []
        
        print("="*60)
        print("SIMPLE SLOT MAPPER")
        print("="*60)
        if len(self.slots) > 0:
            print(f"CONTINUING: {len(self.slots)} slots already defined")
            print(f"Next slot will be #{max([s['id'] for s in self.slots]) + 1}")
        else:
            print("STARTING FRESH: No existing slots")
        print("STEP 1: Click and drag to select area to zoom")
        print("STEP 2: Define parking slots with 4 clicks each")
        print("="*60 + "\n")
    
    def select_area_mode(self):
        """Mode 1: Select area to zoom"""
        selecting = False
        start_x, start_y = 0, 0
        current_x, current_y = 0, 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_x, start_y, current_x, current_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_x, start_y = x, y
                current_x, current_y = x, y
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if selecting:
                    current_x, current_y = x, y
            
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                current_x, current_y = x, y
                
                # Convert to original coordinates
                scale_x = self.orig_w / self.display_w
                scale_y = self.orig_h / self.display_h
                
                x1 = int(min(start_x, current_x) * scale_x)
                y1 = int(min(start_y, current_y) * scale_y)
                x2 = int(max(start_x, current_x) * scale_x)
                y2 = int(max(start_y, current_y) * scale_y)
                
                # Must be reasonable size
                if (x2 - x1) > 100 and (y2 - y1) > 100:
                    self.zoom_region = (x1, y1, x2, y2)
                    self.mode = 'DEFINE_SLOTS'
                    print(f"\n[ZOOMED] Region: {x1},{y1} to {x2},{y2}")
                    print("Now define parking slots (4 clicks each)")
                    print("Press 'b' to go back to area selection\n")
        
        cv2.namedWindow('Select Area to Zoom')
        cv2.setMouseCallback('Select Area to Zoom', mouse_callback)
        
        while self.mode == 'SELECT_AREA':
            display = self.frame.copy()
            
            # Draw selection rectangle
            if selecting:
                cv2.rectangle(display, (start_x, start_y), (current_x, current_y),
                             (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(display, "Click and drag to select parking area", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press Q to quit", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Select Area to Zoom', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def define_slots_mode(self):
        """Mode 2: Define parking slots in zoomed area"""
        if self.zoom_region is None:
            return
        
        # Extract and display zoomed region
        x1, y1, x2, y2 = self.zoom_region
        zoomed = self.original_frame[y1:y2, x1:x2].copy()
        
        # Resize to fit screen
        zoom_h, zoom_w = zoomed.shape[:2]
        scale = min(1280 / zoom_w, 720 / zoom_h, 1.0)
        display_w = int(zoom_w * scale)
        display_h = int(zoom_h * scale)
        
        print(f"Zoomed area: {zoom_w}x{zoom_h} -> Display: {display_w}x{display_h}")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert to original coordinates
                orig_x = int(x / scale) + x1
                orig_y = int(y / scale) + y1
                
                if len(self.current_points) < 4:
                    self.current_points.append([orig_x, orig_y])
                    print(f"[OK] Point {len(self.current_points)}/4: ({orig_x}, {orig_y})")
                    
                    if len(self.current_points) == 4:
                        # Get next slot ID
                        existing_ids = [slot['id'] for slot in self.slots]
                        next_id = max(existing_ids) + 1 if existing_ids else 1
                        
                        slot = {
                            'id': next_id,
                            'name': f'Slot-{next_id}',
                            'points': self.current_points.copy()
                        }
                        self.slots.append(slot)
                        print(f"[COMPLETED] Slot {next_id} | Total: {len(self.slots)}\n")
                        self.current_points = []
        
        cv2.namedWindow('Define Parking Slots', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Define Parking Slots', display_w, display_h)
        cv2.setMouseCallback('Define Parking Slots', mouse_callback)
        
        while self.mode == 'DEFINE_SLOTS':
            display = zoomed.copy()
            
            # Resize for display
            display = cv2.resize(display, (display_w, display_h))
            
            # Draw completed slots
            for slot in self.slots:
                pts = []
                for p in slot['points']:
                    # Convert original coords to display coords
                    dx = int((p[0] - x1) * scale)
                    dy = int((p[1] - y1) * scale)
                    pts.append((dx, dy))
                
                # Draw polygon
                for i in range(4):
                    cv2.line(display, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
                
                # Fill
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                
                # Label
                center_x = sum(pt[0] for pt in pts) // 4
                center_y = sum(pt[1] for pt in pts) // 4
                cv2.putText(display, f"#{slot['id']}", (center_x-15, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw current points
            for i, point in enumerate(self.current_points):
                dx = int((point[0] - x1) * scale)
                dy = int((point[1] - y1) * scale)
                cv2.circle(display, (dx, dy), 5, (255, 0, 0), -1)
                cv2.putText(display, str(i+1), (dx+10, dy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw incomplete slot
            if len(self.current_points) > 1:
                pts = []
                for p in self.current_points:
                    dx = int((p[0] - x1) * scale)
                    dy = int((p[1] - y1) * scale)
                    pts.append((dx, dy))
                for i in range(len(pts)-1):
                    cv2.line(display, pts[i], pts[i+1], (0, 255, 255), 2)
            
            # Instructions
            inst = f"Slots: {len(self.slots)} | Points: {len(self.current_points)}/4"
            cv2.putText(display, inst, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "U=undo  R=remove  B=back  S=save  Q=quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Define Parking Slots', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.slots:
                    self.save_slots()
                break
            elif key == ord('u'):
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"[UNDO] Removed point: {removed}")
            elif key == ord('r'):
                if self.slots:
                    removed = self.slots.pop()
                    print(f"[REMOVED] Slot {removed['id']}")
            elif key == ord('b'):
                self.mode = 'SELECT_AREA'
                print("\n[BACK] Returning to area selection\n")
                break
        
        cv2.destroyAllWindows()
    
    def save_slots(self):
        """Save slots to JSON"""
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        
        with open(self.output_json, 'w') as f:
            json.dump({'slots': self.slots}, f, indent=2)
        
        print("\n" + "="*60)
        print("SAVED PARKING SLOTS")
        print("="*60)
        print(f"File: {self.output_json}")
        print(f"Total slots: {len(self.slots)}")
        print("="*60)
    
    def run(self):
        """Run the mapper"""
        while True:
            if self.mode == 'SELECT_AREA':
                if not self.select_area_mode():
                    break
            elif self.mode == 'DEFINE_SLOTS':
                self.define_slots_mode()
                if self.mode == 'DEFINE_SLOTS':
                    break


def main():
    import sys
    
    video = sys.argv[1] if len(sys.argv) > 1 else 'parking_video.mp4.mp4'
    
    if not os.path.exists(video):
        print(f"[ERROR] Video not found: {video}")
        return
    
    try:
        mapper = SimpleSlotMapper(video)
        mapper.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
