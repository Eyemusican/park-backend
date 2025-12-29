"""
ULTRA-SIMPLE SLOT MAPPER - Click 4 corners of ONE slot, system replicates rest
Perfect for painted parking lines - ZERO confusion
"""
import cv2
import numpy as np
import json
import sys
import os


def simple_slot_mapper(video_path, output_json='configs/parking_slots.json'):
    """
    ACCURATE METHOD - Click 4 corners per slot:
    1. Click 4 corners of slot 1 following white painted lines
    2. Press SPACE to save and move to next slot
    3. Click 4 corners of slot 2... and so on
    4. Press S to save all when done
    5. Perfect accuracy!
    """
    
    print("="*70)
    print("ACCURATE PARKING SLOT MAPPER")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Cannot read video")
        return
    
    orig_frame = frame.copy()
    print(f"\nVideo: {video_path}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Resize for display
    max_width = 1400
    display_scale = 1.0
    if frame.shape[1] > max_width:
        display_scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    
    print(f"Display scale: {display_scale:.2f}x\n")
    
    print("="*70)
    print("INSTRUCTIONS - Each slot follows WHITE PAINTED LINES!")
    print("="*70)
    print("FOR EACH PARKING SLOT:")
    print("  1. Click 4 corners following the WHITE LINES:")
    print("     Corner 1: Top-left → Corner 2: Top-right →")
    print("     Corner 3: Bottom-right → Corner 4: Bottom-left")
    print("  2. Press SPACE to save this slot and start next")
    print("  3. Repeat for all visible parking slots")
    print("  4. Press S to SAVE ALL when finished")
    print("")
    print("SHORTCUTS:")
    print("  SPACE = Save current slot, start next")
    print("  U = Undo last corner")
    print("  R = Reset current slot")
    print("  S = Save all slots and exit")
    print("="*70 + "\n")
    
    slots = []
    current_points = []
    current_slot_id = 1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points, slots, current_slot_id
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) < 4:
                current_points.append((x, y))
                print(f"✓ Slot {current_slot_id} - Corner {len(current_points)}/4: ({x}, {y})")
                
                if len(current_points) == 4:
                    print(f"✓ Slot {current_slot_id} complete! Press SPACE to save and continue")
    
    window_name = f'Slot Mapper - SPACE=Next | S=Save All | U=Undo | R=Reset'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"Starting Slot {current_slot_id} - Click 4 corners...\n")
    
    while True:
        display = frame.copy()
        
        # Draw all completed slots
        for slot in slots:
            pts = np.array(slot['display_points'], np.int32)
            cv2.polylines(display, [pts], True, (0, 200, 0), 2)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(display, str(slot['id']), tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw current points being clicked
        for i, point in enumerate(current_points):
            color = (0, 255, 255)
            cv2.circle(display, point, 10, color, -1)
            cv2.putText(display, str(i+1), (point[0]+15, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Draw lines between current points
        if len(current_points) > 1:
            for i in range(len(current_points)):
                if i < len(current_points) - 1:
                    cv2.line(display, current_points[i], current_points[i+1], (0, 255, 255), 2)
        
        # Draw current slot if complete
        if len(current_points) == 4:
            pts = np.array(current_points, np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 4)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        
        # Minimal transparent overlay - bottom right
        box_w = 380
        box_h = 65
        box_x = display.shape[1] - box_w - 10
        box_y = display.shape[0] - box_h - 10
        
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        # Border
        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
        
        # Compact text
        text1 = f"Slot {current_slot_id} | {len(current_points)}/4 | Done: {len(slots)}"
        text2 = "SPACE=Next | S=Save | U=Undo"
        
        cv2.putText(display, text1, (box_x + 10, box_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display, text2, (box_x + 10, box_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE - save current slot
            if len(current_points) == 4:
                # Convert to original resolution
                orig_points = []
                for p in current_points:
                    orig_x = int(p[0] / display_scale)
                    orig_y = int(p[1] / display_scale)
                    orig_points.append([orig_x, orig_y])
                
                slots.append({
                    'id': current_slot_id,
                    'points': orig_points,
                    'display_points': current_points
                })
                
                print(f"✅ Slot {current_slot_id} saved!\n")
                current_slot_id += 1
                current_points = []
                print(f"Starting Slot {current_slot_id} - Click 4 corners...\n")
            else:
                print("⚠️  Need 4 corners before saving!")
        
        elif key == ord('s') or key == ord('S'):  # Save all
            if len(slots) == 0:
                print("⚠️  No slots defined!")
                continue
            
            # Confirm save
            print(f"\n{'='*70}")
            print(f"Ready to save {len(slots)} slots?")
            confirm = input("Type 'yes' to save: ").strip().lower()
            
            if confirm == 'yes':
                break
            else:
                print("Cancelled - continue mapping")
        
        elif key == ord('u') or key == ord('U'):  # Undo last corner
            if len(current_points) > 0:
                removed = current_points.pop()
                print(f"✓ Undone - now {len(current_points)}/4 corners")
        
        elif key == ord('r') or key == ord('R'):  # Reset current slot
            if len(current_points) > 0:
                current_points = []
                print(f"✓ Reset Slot {current_slot_id} - click 4 corners again")
        
        elif key == ord('q') or key == ord('Q'):  # Quit
            print("\n[QUIT] Mapping cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    
    if len(slots) == 0:
        print("[ERROR] No slots saved")
        return None
    
    # Clean up slots - remove display_points
    clean_slots = []
    for slot in slots:
        clean_slots.append({
            'id': slot['id'],
            'points': slot['points']
        })
    
    print(f"\n✅ Mapped {len(clean_slots)} parking slots!")
    
    # Save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({'slots': clean_slots, 'total_slots': len(clean_slots)}, f, indent=2)
    
    print(f"✅ Saved to: {output_json}")
    
    # Preview
    preview = orig_frame.copy()
    if preview.shape[1] > 1400:
        preview_scale = 1400 / preview.shape[1]
        preview = cv2.resize(preview, None, fx=preview_scale, fy=preview_scale)
    else:
        preview_scale = 1.0
    
    for slot in clean_slots:
        if preview_scale != 1.0:
            pts = np.array([[int(p[0]*preview_scale), int(p[1]*preview_scale)] 
                           for p in slot['points']], np.int32)
        else:
            pts = np.array(slot['points'], np.int32)
        
        cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, preview)
        
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(preview, str(slot['id']), tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    cv2.imwrite('parking_slots_preview.jpg', preview)
    print("✅ Preview: parking_slots_preview.jpg")
    
    print("\n" + "="*70)
    print(f"✅ DONE! {len(clean_slots)} slots saved")
    print("Each slot PERFECTLY matches the white painted lines!")
    print("\nRun detection now:")
    print("  python smart_parking_mvp.py --run parking_evening_vedio.mp4")
    print("="*70 + "\n")
    
    cv2.namedWindow('FINAL RESULT', cv2.WINDOW_NORMAL)
    cv2.imshow('FINAL RESULT', preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return clean_slots


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_slot_mapper.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)
    
    simple_slot_mapper(video_path)
