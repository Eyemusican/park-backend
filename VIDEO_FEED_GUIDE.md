# Video Feed Integration Guide

## Overview
The parking system now includes live video streaming with real-time vehicle detection and parking slot visualization.

## Backend Changes (server.py)

### New Endpoints Added:

1. **GET /api/video-feed**
   - Streams live video with vehicle detections and parking slot overlays
   - MJPEG stream format
   - Shows real-time parking occupancy status

2. **POST /api/video/start**
   - Starts video processing
   - Body: `{ "video_source": "path/to/video.mp4" OR 0 for webcam, "parking_id": 1 }`
   - Returns: Success message with stream details

3. **POST /api/video/stop**
   - Stops video processing
   - Returns: Success message

4. **GET /api/video/status**
   - Returns current video streaming status
   - Response: `{ "running": boolean, "has_frame": boolean }`

### Features:
- Real-time YOLO vehicle detection (cars, motorcycles, buses, trucks)
- Parking slot overlay with live occupancy status from database
- Color-coded slots: Green (Free), Red (Occupied)
- Auto-loops video files for continuous monitoring
- ~30 FPS streaming performance
- Threaded processing to not block API server

## Frontend Changes

### New Component: VideoFeed
Location: `Frontend/components/video-feed.tsx`

Features:
- Live MJPEG stream display
- Start/Stop/Refresh controls
- Status indicators (Live/Offline)
- Recording indicator overlay
- Error handling and loading states
- Auto-status checking every 5 seconds

### Integration Points:
1. **Dashboard** (`components/sections/real-time-monitoring.tsx`)
   - Video feed shown at the top of monitoring section
   
2. **Parking Area Detail** (`app/parking/[areaId]/page.tsx`)
   - Video feed specific to parking area ID
   - Positioned between area stats and location map

## Usage

### Starting the Video Stream

#### Option 1: Use a video file
```bash
# Backend will be running on port 5000
# Frontend will display video feed component with Start button
```

#### Option 2: API Call
```bash
curl -X POST http://localhost:5000/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"video_source": "videos/parking.mp4", "parking_id": 1}'
```

#### Option 3: Use webcam
```javascript
// Change videoSource to 0 in the component
<VideoFeed parkingId={1} videoSource={0} />
```

### Video Source Options:
- `0` - Default webcam
- `"videos/parking.mp4"` - Video file path (relative to backend directory)
- `"rtsp://camera-url"` - RTSP stream
- `"http://camera-ip/stream"` - HTTP stream

### Viewing the Stream

1. Navigate to the dashboard (`/admin`)
2. The video feed appears in the "Real-time Monitoring" section
3. Click "Start Stream" to begin
4. The feed will show:
   - Blue boxes around detected vehicles
   - Green/Red overlays on parking slots
   - Slot numbers and occupancy status
   - Frame counter and parking area ID

## Configuration

### Parking Slots
The system loads parking slot coordinates from:
```
smart_parking_mvp/configs/parking_slots.json
```

Format:
```json
{
  "slots": [
    {
      "id": 1,
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

### Detection Settings (in server.py):
- Confidence threshold: 0.25
- Frame processing: Every 3rd frame
- Target FPS: ~30
- Vehicle classes: car (2), motorcycle (3), bus (5), truck (7)

## Troubleshooting

### Video not showing:
1. Check if backend server is running on port 5000
2. Verify video file exists at specified path
3. Check browser console for CORS errors
4. Ensure YOLO model files exist (yolov8m.pt or yolov8n.pt)

### Slow performance:
- Reduce frame processing frequency (modify `frame_count % 3`)
- Use smaller YOLO model (yolov8n.pt instead of yolov8m.pt)
- Reduce video resolution
- Ensure GPU acceleration if available

### Parking slots not showing:
- Verify `configs/parking_slots.json` exists
- Check slot coordinates match video dimensions
- Ensure database has parking area and slots created

## Performance Notes

- Video processing runs in a separate thread
- Database queries for occupancy happen every frame
- YOLO detection runs every 3rd frame for optimization
- Recommended: Use video files instead of live camera for testing

## Security Considerations

- The video stream is unencrypted (HTTP)
- No authentication on video endpoints
- For production:
  - Add authentication middleware
  - Use HTTPS/WSS
  - Implement rate limiting
  - Add access controls per parking area

## Future Enhancements

- [ ] Multiple concurrent video streams
- [ ] WebSocket for lower latency
- [ ] Video recording/playback
- [ ] Motion detection alerts
- [ ] Vehicle counting analytics
- [ ] License plate recognition
- [ ] Thermal/night vision support
