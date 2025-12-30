# Smart Parking System - Backend

## Overview

Flask-based backend for the Smart Parking System with integrated YOLOv8 vehicle detection, real-time tracking, REST API, and Telegram bot integration.

## Features

### Core Features
- **Vehicle Detection**: YOLOv8-based detection with ByteTrack tracking
- **Real-Time Monitoring**: MJPEG streaming of detection feeds
- **Parking Slot Management**: Define and monitor parking slots via polygon mapping
- **Violation Detection**: Automatic overtime and unauthorized parking detection
- **Multi-Camera Support**: RTSP camera integration with auto-reconnect

### API Features
- **REST API**: Full CRUD for parking areas, slots, events, and violations
- **Live Statistics**: Real-time occupancy rates and analytics
- **Video Upload**: Upload videos and extract frames for slot mapping

### Telegram Bot
- **View Parking Areas**: Browse all areas with real-time availability
- **Live Feed Links**: Get MJPEG stream URLs for each parking area
- **Availability Notifications**: Subscribe to get notified when spots open up
- **Commands**: `/start`, `/parking`, `/subscriptions`

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL (via Docker)
- (Optional) NVIDIA GPU with CUDA

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your database credentials
```

### Database Setup

```bash
# Start PostgreSQL container
docker run -d \
    --name parking-postgres \
    -e POSTGRES_USER=parking_user \
    -e POSTGRES_PASSWORD=parking_password \
    -e POSTGRES_DB=parking_db \
    -p 5433:5432 \
    postgres:15-alpine

# Apply schema
docker exec -i parking-postgres psql -U parking_user -d parking_db < parking_schema.sql
docker exec -i parking-postgres psql -U parking_user -d parking_db < violations_schema.sql
docker exec -i parking-postgres psql -U parking_user -d parking_db < cameras_schema.sql

# Apply migrations
docker exec -i parking-postgres psql -U parking_user -d parking_db < migrations/001_add_geometry_columns.sql
docker exec -i parking-postgres psql -U parking_user -d parking_db < migrations/002_telegram_subscriptions.sql
```

### Run Server

```bash
source venv/bin/activate
python server.py
```

Server starts at http://localhost:5001

## Telegram Bot Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your bot token
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```
4. Restart the server

See [TELEGRAM_BOT_SETUP.md](TELEGRAM_BOT_SETUP.md) for detailed instructions.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | Database host |
| `DB_PORT` | 5432 | Database port |
| `DB_NAME` | parking_db | Database name |
| `DB_USER` | parking_user | Database user |
| `DB_PASS` | - | Database password |
| `TELEGRAM_BOT_TOKEN` | - | Telegram bot token (optional) |
| `TELEGRAM_API_BASE_URL` | http://localhost:5001/api | API URL for bot feed links |
| `FORCE_CPU` | false | Force CPU-only mode (disable GPU) |

### Detection Config (`config.py`)

```python
YOLO_MODEL = "yolov8s.pt"      # Model: n/s/m/l/x
CONFIDENCE_THRESHOLD = 0.35    # Detection confidence
IMAGE_SIZE = 960               # Input resolution
HALF_PRECISION = True          # FP16 for GPU
```

## API Endpoints

### Parking Areas
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/parking-areas` | List all parking areas |
| POST | `/api/parking-areas` | Create parking area |
| GET | `/api/parking-areas/<id>` | Get parking area |
| PUT | `/api/parking-areas/<id>` | Update parking area |
| DELETE | `/api/parking-areas/<id>` | Delete parking area |
| POST | `/api/parking-areas/create-with-slots` | Create with slot polygons |

### Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/parking-areas/<id>/detection/start` | Start detection |
| POST | `/api/parking-areas/<id>/detection/stop` | Stop detection |
| GET | `/api/parking-areas/<id>/detection/status` | Detection status |
| GET | `/api/parking-areas/<id>/detection/feed` | MJPEG video feed |

### Slots & Events
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/parking-areas/<id>/slots` | Get slots with status |
| GET | `/api/parking-events` | Get parking events |
| GET | `/api/parking/active` | Active parking sessions |

### Violations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/violations` | Get active violations |
| GET | `/api/violations/summary` | Violation statistics |
| POST | `/api/violations/resolve/<id>` | Resolve violation |

### Video Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/video/upload` | Upload video, extract frame |
| GET | `/api/frames/<filename>` | Serve extracted frame |

## Project Structure

```
park-backend/
├── server.py                 # Flask API server (main entry point)
├── telegram_bot.py           # Telegram bot integration
├── detection_manager.py      # Detection thread management
├── detector.py               # YOLOv8 vehicle detector
├── tracker.py                # Vehicle tracking
├── db_helper.py              # Database helper class
├── config.py                 # Configuration settings
├── violation_detector.py     # Violation detection logic
├── rtsp_camera_manager.py    # RTSP camera management
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── TELEGRAM_BOT_SETUP.md     # Bot setup guide
│
├── migrations/
│   ├── 001_add_geometry_columns.sql
│   └── 002_telegram_subscriptions.sql
│
├── uploads/
│   ├── videos/               # Uploaded videos
│   └── frames/               # Extracted frames
│
└── tests/
    ├── test_api_parking_areas.py
    └── test_api_parking_slots.py
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api_parking_areas.py -v

# With coverage
pytest --cov=. --cov-report=html
```

## Database Utilities

```bash
# View database contents
python db_utils.py view

# Live monitoring
python db_utils.py monitor

# Reset parking data
python db_utils.py reset
```

## Troubleshooting

### Detection Issues
- Check GPU availability: Look for "Using device: cuda" in logs
- Reduce `IMAGE_SIZE` to 640 for better performance
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`

### Database Connection
- Verify PostgreSQL container is running: `docker ps`
- Check connection: `docker exec -it parking-postgres psql -U parking_user -d parking_db`

### Telegram Bot Not Responding
- Verify `TELEGRAM_BOT_TOKEN` is set in `.env`
- Check token: `curl https://api.telegram.org/bot<TOKEN>/getMe`
- Check server logs for bot startup message

## License

Smart Parking System - AI-powered parking monitoring solution.
