# Smart Parking System - Complete Implementation Guide

## Overview

Complete smart parking management system with vehicle tracking, parking slot occupancy detection, duration tracking, and violation monitoring.

## System Components

### 1. **tracking_test.py** - Basic Tracking Verification

Tests YOLO tracking functionality before building complete system.

**Usage:**

```bash
python tracking_test.py parking_video.mp4.mp4
```

**What it does:**

- Verifies YOLO model.track() works with persist=True
- Shows vehicle track IDs and trajectories
- Displays basic tracking statistics

**Controls:**

- `Q` - Quit

---

### 2. **slot_mapper.py** - Interactive Slot Definition

Interactive tool to define parking slot polygons by clicking corners.

**Usage:**

```bash
python slot_mapper.py parking_video.mp4.mp4
```

**How to use:**

1. Video loads showing first frame
2. Click 4 corners to define a parking slot polygon
3. Slot auto-completes after 4 clicks
4. Repeat for all parking slots
5. Press `S` to save and quit

**Controls:**

- `Left Click` - Add corner point (4 points = 1 slot)
- `C` - Complete current slot manually
- `U` - Undo last point
- `R` - Remove last slot
- `S` - Save slots to JSON and quit
- `Q` - Quit without saving

**Output:**

- Creates `configs/parking_slots.json`
- JSON format:

```json
{
  "slots": [
    {
      "id": 1,
      "name": "Slot-1",
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

---

### 3. **smart_parking_complete.py** - Full Parking System

Complete integrated system with tracking + occupancy detection.

**Usage:**

```bash
python smart_parking_complete.py configs/parking_slots.json parking_video.mp4.mp4
```

**Architecture:**

#### **VehicleInfo Class**

Tracks individual vehicle information:

- `track_id` - Unique vehicle identifier
- `positions` - Position history (last 30 frames)
- `current_box` - Bounding box (x1, y1, x2, y2)
- `is_moving` - Motion status (True/False)
- `stopped_time` - When vehicle stopped
- `parked_slot_id` - Which slot vehicle is parked in
- `parking_start` - When parking began
- `vehicle_class` - car/motorcycle/bus/truck
- `confidence` - Detection confidence

**Methods:**

- `update_position()` - Update position and analyze motion
- `get_stopped_duration()` - How long stopped
- `get_parking_duration()` - How long parked

#### **ParkingSlot Class**

Represents a parking slot polygon:

- `id` - Slot number
- `name` - Slot name
- `points` - Polygon corners
- `polygon` - Shapely Polygon object
- `is_occupied` - Occupancy status
- `occupied_by` - Vehicle track_id
- `occupied_since` - Occupancy start time
- `total_occupancies` - Historical usage count
- `total_duration` - Total time occupied

**Methods:**

- `check_occupancy(vehicle_box)` - Returns IoU (intersection over union)
- `mark_occupied(vehicle_id)` - Mark as occupied
- `mark_vacant()` - Mark as vacant
- `get_occupancy_duration()` - Current occupancy duration

#### **SmartParkingSystem Class**

Main system controller:

**Configuration:**

- `occupancy_threshold = 0.5` - 50% overlap = occupied
- `parking_time_threshold = 2.0` - 2 seconds stopped = parked

**Core Methods:**

- `update_slot_occupancy()` - Updates all slot occupancy states

  1. Checks each stopped vehicle against each slot
  2. Uses Shapely polygon intersection for IoU calculation
  3. Requires 50% vehicle overlap with slot
  4. Vehicle must be stopped for 2+ seconds
  5. Assigns vehicle to best matching slot

- `process_frame(frame)` - Process video frame
  1. Run YOLO tracking with ByteTrack
  2. Update vehicle positions and motion status
  3. Update slot occupancy
  4. Draw visualization

**Visualization:**

**Parking Slots:**

- GREEN polygon = Vacant
- RED polygon = Occupied
- Label shows slot ID and occupancy duration

**Vehicles:**

- GREEN box = Moving
- YELLOW box = Stopped (not in slot)
- RED box = Parked in slot
- Purple trail = Movement trajectory
- Label shows: ID, status, slot number, duration

**Statistics Dashboard:**

```
SMART PARKING SYSTEM
FPS: 16.5 | Frame: 1245

PARKING CAPACITY:
  Occupied: 5/8 (62.5%)
  Vacant: 3

VEHICLES:
  Total Tracked: 12
  Moving: 7
  Parked: 5
  Violations: 0
```

**Controls:**

- `Q` - Quit
- `S` - Save snapshot to output/snapshots/
- `R` - Reset statistics

**Violation Detection:**

- Vehicles stopped outside parking slots are counted as violations
- Shown in statistics dashboard

---

## Step-by-Step Usage

### Step 1: Test Basic Tracking (Optional)

```bash
python tracking_test.py parking_video.mp4.mp4
```

Verify tracking works before proceeding.

### Step 2: Define Parking Slots

```bash
python slot_mapper.py parking_video.mp4.mp4
```

**Instructions:**

1. Video loads - shows first frame
2. Identify parking slots in the scene
3. For each slot, click 4 corners (clockwise or counter-clockwise)
4. Slot completes automatically after 4 clicks
5. Repeat for all slots
6. Press `S` to save

**Tips:**

- Click corners in order to form proper polygon
- Use `U` to undo if you make a mistake
- Use `R` to remove entire slot if needed
- Slots should not overlap

### Step 3: Run Complete System

```bash
python smart_parking_complete.py
```

**What you'll see:**

- Parking slots outlined (GREEN=vacant, RED=occupied)
- Vehicles with track IDs and status colors
- Real-time statistics dashboard
- Occupancy percentages and durations

---

## Configuration

### YOLO Tracking Parameters (in code)

```python
results = model.track(
    source=frame,
    persist=True,              # Enable tracking
    tracker='bytetrack.yaml',  # ByteTrack algorithm
    conf=0.4,                  # Confidence threshold
    iou=0.6,                   # IoU for NMS
    classes=[2, 3, 5, 7],      # Vehicle classes
    device='cuda'              # GPU acceleration
)
```

### Occupancy Detection Parameters

```python
occupancy_threshold = 0.5        # 50% overlap required
parking_time_threshold = 2.0     # 2 seconds stopped = parked
```

**Adjust these if:**

- Too many false occupancies: Increase `occupancy_threshold` to 0.6-0.7
- Missing parked vehicles: Decrease `occupancy_threshold` to 0.3-0.4
- Vehicles marked as parked too quickly: Increase `parking_time_threshold`
- Vehicles not marked as parked: Decrease `parking_time_threshold`

---

## Color Coding Reference

### Parking Slots

| Color    | Status   | Description            |
| -------- | -------- | ---------------------- |
| 🟢 GREEN | Vacant   | Slot is available      |
| 🔴 RED   | Occupied | Vehicle parked in slot |

### Vehicles

| Color     | Status  | Description                              |
| --------- | ------- | ---------------------------------------- |
| 🟢 GREEN  | Moving  | Vehicle in motion                        |
| 🟡 YELLOW | Stopped | Stopped outside parking slot (violation) |
| 🔴 RED    | Parked  | Parked in designated slot                |
| 🟣 PURPLE | Trail   | Movement trajectory                      |

---

## File Structure

```
smart_parking_mvp/
├── tracking_test.py              # Basic tracking verification
├── slot_mapper.py                # Interactive slot definition tool
├── smart_parking_complete.py     # Complete parking system
├── bytetrack.yaml                # ByteTrack configuration
├── parking_video.mp4.mp4         # Input video
├── configs/
│   └── parking_slots.json        # Defined parking slots
├── output/
│   └── snapshots/                # Saved snapshots (S key)
└── .venv/                        # Python virtual environment
```

---

## Dependencies

```
opencv-python>=4.8.0    # Video processing
numpy>=1.24.0           # Numerical operations
ultralytics>=8.0.0      # YOLO model
torch>=2.0.0            # PyTorch (CUDA)
shapely>=2.0.0          # Polygon operations
```

Install all:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### "Parking slots not found"

**Solution:** Run `slot_mapper.py` first to create `configs/parking_slots.json`

### No vehicles detected

**Solutions:**

- Lower confidence threshold: Change `conf=0.4` to `conf=0.3`
- Check video has visible vehicles
- Verify GPU is being used (should see CUDA messages)

### Vehicles not marked as parked

**Solutions:**

- Check slot polygons cover parking areas correctly
- Lower `occupancy_threshold` from 0.5 to 0.3
- Lower `parking_time_threshold` from 2.0 to 1.0

### Tracking IDs keep changing

**Solution:** Already fixed with:

- ByteTrack `track_buffer=60` (keeps tracks during occlusion)
- Higher `iou=0.6` (reduces duplicates)
- `conf=0.4` (filters low-confidence detections)

### False parking violations

**Cause:** Vehicles stopped outside slots shown as violations
**Expected Behavior:** This is correct - use for monitoring illegal parking

### Low FPS

**Solutions:**

- Use smaller YOLO model: `yolov8n.pt` or `yolov8s.pt`
- Reduce video resolution
- Ensure GPU (CUDA) is being used

---

## Performance Tips

### For Better Tracking:

1. Use stable camera (no shaking)
2. Good lighting conditions
3. Clear view of parking area
4. ByteTrack parameters already optimized

### For Better Occupancy Detection:

1. Define slot polygons accurately (click corners precisely)
2. Slots should cover actual parking space boundaries
3. Don't overlap slots
4. Adjust `occupancy_threshold` based on camera angle

---

## Next Steps / Extensions

### Possible Enhancements:

1. **Database Logging** - Store all events in SQLite/PostgreSQL
2. **Web Dashboard** - Flask/FastAPI web interface
3. **Real-time Alerts** - Email/SMS for violations or full capacity
4. **License Plate Recognition** - Track specific vehicles
5. **Payment Integration** - Calculate parking fees
6. **Historical Analytics** - Peak hours, average duration
7. **Multi-camera Support** - Monitor multiple parking areas
8. **Cloud Deployment** - AWS/Azure for scalability

---

## API (For Integration)

### Key Classes and Methods

**VehicleInfo:**

```python
vehicle = VehicleInfo(track_id)
vehicle.update_position(center, box, class, conf)
duration = vehicle.get_parking_duration()
```

**ParkingSlot:**

```python
slot = ParkingSlot(slot_data)
iou = slot.check_occupancy(vehicle_box)
slot.mark_occupied(vehicle_id)
duration = slot.get_occupancy_duration()
```

**SmartParkingSystem:**

```python
system = SmartParkingSystem('configs/parking_slots.json')
annotated_frame = system.process_frame(frame)
system.update_slot_occupancy()
```

---

## Contact / Support

For issues or questions:

1. Check this README
2. Review code comments in each file
3. Adjust configuration parameters
4. Test with `tracking_test.py` first

---

## License

MIT License - Free to use and modify

---

**Built with:** YOLOv8 + OpenCV + Shapely + ByteTrack
**Platform:** Windows with CUDA GPU acceleration
**Status:** ✅ Complete and ready to use!
