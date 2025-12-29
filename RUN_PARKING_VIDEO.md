# Smart Parking Video Processing - Quick Guide

## 🚀 Run the Parking System

### Basic Usage
```bash
cd smart_parking_mvp
python smart_parking_complete.py configs/parking_slots.json parking_video.mp4
```

## ✨ Features

### 1. **Immediate Detection** ✅
- **All vehicles detected at once when video starts**
- Lower confidence threshold (0.25) catches all vehicles immediately
- Faster motion detection (10 frames instead of 20)
- Easier track creation for instant recognition

### 2. **Smooth Normal Playback** ✅
- **Normal speed video playback** (not laggy or too fast)
- Balanced frame processing for smooth display
- Optimized GPU acceleration with FP16
- No frame skipping - processes every frame

### 3. **Stable Tracking** ✅
- 3-second vehicle persistence (prevents losing tracks)
- 2-second slot grace period (prevents flickering)
- Balanced tracking parameters for stability

## 🎮 Keyboard Controls

- **Q** - Quit
- **S** - Save snapshot
- **R** - Reset statistics

## ⚙️ Configuration

### Current Settings (Optimized)
- **Detection Confidence**: 0.25 (low = detect all vehicles)
- **Motion Threshold**: 3 pixels (quick response)
- **Vehicle Persistence**: 3 seconds
- **Slot Grace Period**: 2 seconds
- **Video Speed**: 1x (normal)
- **Frame Skip**: 0 (process all frames)

### To Make Video Faster (if needed)
Edit `smart_parking_complete.py` line ~241:
```python
self.video_speed_multiplier = 2  # Change to 2 for 2x speed, 3 for 3x, etc.
```

### To Make Detection More Aggressive
Edit `smart_parking_complete.py` line ~344:
```python
conf=0.20,  # Lower value = detect more vehicles (min 0.1)
```

## 📊 Expected Results

- ✅ All vehicles detected immediately when video starts
- ✅ Smooth normal-speed playback
- ✅ No lag or stuttering
- ✅ Stable slot occupancy (minimal flickering)
- ✅ Consistent vehicle tracking with stable IDs
- ✅ Real-time processing with GPU acceleration

## 🔧 Technical Details

### Immediate Detection
- **Confidence Threshold**: 0.45 → 0.25 (more sensitive)
- **ByteTrack new_track_thresh**: 0.7 → 0.4 (easier to start tracking)
- **Motion Detection Samples**: 20 frames → 10 frames (faster)
- **Motion Threshold**: 8px → 3px (more responsive)

### Smooth Playback
- **Video Speed**: 3x → 1x (normal)
- **Frame Skip**: 0 (all frames processed)
- **Wait Key**: Dynamic based on speed multiplier
- **GPU Half Precision**: Enabled for faster processing

### Stability Balance
- **Vehicle Timeout**: 5s → 3s (balanced)
- **Slot Grace Period**: 3s → 2s (responsive)
- **Track Buffer**: 180 → 90 frames (balanced)
- **IOU Threshold**: 0.7 → 0.65 (balanced)

## 🐛 Troubleshooting

**If some vehicles not detected at start:**
- Lower confidence to 0.20 in `process_frame()` function
- Lower `new_track_thresh` to 0.3 in `bytetrack.yaml`

**If video is laggy/slow:**
- Ensure GPU is being used (check console output)
- Reduce image size to 640 in `process_frame()`
- Enable frame skipping: `self.frame_skip = 1`

**If slots flicker:**
- Increase grace period to 3 seconds in `mark_vacant()`
- Increase vehicle timeout to 4 seconds in `process_frame()`

**If too many false detections:**
- Increase confidence to 0.35 in `process_frame()`
