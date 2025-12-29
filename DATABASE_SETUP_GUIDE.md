# 🗄️ SMART PARKING - DATABASE SETUP GUIDE

## Overview

This guide shows you how to use PostgreSQL database instead of in-memory storage for persistent data.

## 📋 Benefits of Using Database

| Feature | In-Memory | Database |
|---------|-----------|----------|
| Data Persistence | ❌ Lost on restart | ✅ Permanent |
| Historical Data | ❌ No history | ✅ Full history |
| Reports | ❌ Not possible | ✅ Detailed reports |
| Backup | ❌ No backup | ✅ Automatic |
| Capacity | ⚠️ Limited by RAM | ✅ Unlimited |

## 🚀 Quick Start (3 Steps)

### Step 1: Install PostgreSQL

**Windows:**
1. Download PostgreSQL: https://www.postgresql.org/download/windows/
2. Run installer (use default port 5432)
3. Remember your password!

**Already have PostgreSQL?** Skip to Step 2

### Step 2: Create Database User

Open PostgreSQL command line (psql) or pgAdmin and run:

```sql
CREATE USER parking_user WITH PASSWORD 'Tenzin@2005';
ALTER USER parking_user CREATEDB;
```

### Step 3: Run the Setup Script

```bash
cd smart_parking_mvp
python run_with_database.py
```

The script will automatically:
- ✅ Test PostgreSQL connection
- ✅ Create `parking_db` database
- ✅ Create all required tables
- ✅ Start the server on port 5000

## 📊 System Architecture

```
┌─────────────────────────┐
│   Video Processing      │
│   (smart_parking_mvp.py)│
│                         │
│   Detects vehicles,     │
│   tracking, violations  │
└───────────┬─────────────┘
            │ HTTP POST
            ↓
┌─────────────────────────┐
│   API Server            │
│   (server.py)           │
│   Port: 5000            │
└───────────┬─────────────┘
            │ SQL
            ↓
┌─────────────────────────┐
│   PostgreSQL Database   │
│   (parking_db)          │
│                         │
│   Tables:               │
│   - parking_area        │
│   - parking_slots       │
│   - parking_events      │
│   - parking_violations  │
└───────────┬─────────────┘
            │ HTTP GET
            ↓
┌─────────────────────────┐
│   Frontend              │
│   (Next.js)             │
│   Port: 3000            │
└─────────────────────────┘
```

## 🎮 Complete Workflow

### 1. Start Database Server

```bash
python run_with_database.py
```

This starts the API server with database connection on port 5000.

### 2. Start Frontend

Open a new terminal:

```bash
cd ../Frontend
npm run dev
```

Frontend runs on http://localhost:3000

### 3. Run Video Processing

Open another terminal:

```bash
cd smart_parking_mvp
python smart_parking_mvp.py --run your_video.mp4
```

Or for webcam:

```bash
python smart_parking_mvp.py --run 0
```

## 📡 API Endpoints (Database-Backed)

### Parking Sessions
- `GET /api/parking/active` - Active parking sessions
- `GET /api/parking/slots/status` - All slots status
- `POST /api/parking/entry` - Record vehicle entry
- `POST /api/parking/exit` - Record vehicle exit

### Violations
- `GET /api/violations` - All active violations
- `GET /api/violations/summary` - Violation statistics
- `POST /api/violations/from-video` - Receive from video
- `POST /api/violations/resolve/<id>` - Resolve violation

### Statistics
- `GET /api/stats/overview` - System overview
- `GET /api/parking-areas` - All parking areas

### Health Check
- `GET /api/health` - Server health check

## 🗃️ Database Schema

### parking_area
```sql
parking_id    SERIAL PRIMARY KEY
parking_name  VARCHAR(255)
slot_count    NUMERIC
```

### parking_slots
```sql
slot_id       SERIAL PRIMARY KEY
parking_id    INT (FK to parking_area)
slot_number   INT
is_occupied   BOOLEAN
```

### parking_events
```sql
event_id      SERIAL PRIMARY KEY
slot_id       INT (FK to parking_slots)
arrival_time  TIMESTAMP
departure_time TIMESTAMP
parked_time   INT (minutes)
```

### parking_violations
```sql
violation_id    VARCHAR(50) PRIMARY KEY
slot_id         INT
vehicle_id      VARCHAR(50)
license_plate   VARCHAR(50)
violation_type  VARCHAR(100)
severity        VARCHAR(20) (High/Medium/Low)
description     TEXT
duration_minutes DECIMAL(10,2)
detected_at     TIMESTAMP
resolved_at     TIMESTAMP
status          VARCHAR(20) (ACTIVE/RESOLVED)
```

## 🔧 Troubleshooting

### Error: "Cannot connect to PostgreSQL"

**Solution:**
1. Check PostgreSQL is running
2. Verify credentials in script
3. Test connection: `psql -U parking_user -d postgres`

### Error: "Database does not exist"

**Solution:**
Run the setup script - it will create the database automatically:
```bash
python run_with_database.py
```

### Error: "Table does not exist"

**Solution:**
Apply schemas manually:
```bash
psql -U parking_user -d parking_db -f parking_schema.sql
psql -U parking_user -d parking_db -f violations_schema.sql
```

### Video not updating frontend

**Solution:**
1. Ensure server is running: http://localhost:5000/api/health
2. Check video script is sending data (look for POST logs)
3. Verify frontend is fetching: Check browser console

## 🎯 Key Features

### ✅ Persistent Storage
- All parking events saved permanently
- Historical data available for reports
- Survives server restarts

### ✅ Real-Time Updates
- Video processing sends live data
- Frontend polls every 500ms
- Immediate violation detection

### ✅ Violation Tracking
- Duration violations automatically detected
- Stored with evidence and timestamps
- Can generate compliance reports

### ✅ Analytics Ready
- All data timestamped
- Can query by date range
- Revenue calculation ready
- Occupancy trends

## 📈 Next Steps

1. **Run everything together:**
   ```bash
   # Terminal 1: Database server
   python run_with_database.py
   
   # Terminal 2: Frontend
   cd ../Frontend
   npm run dev
   
   # Terminal 3: Video processing
   python smart_parking_mvp.py --run video.mp4
   ```

2. **Access the system:**
   - Frontend: http://localhost:3000
   - API: http://localhost:5000
   - Database: localhost:5432

3. **Monitor logs:**
   - Server logs show all API requests
   - Video script logs detection events
   - Frontend console shows fetch status

## 🆚 In-Memory vs Database

**Use In-Memory (simple_server.py):**
- Quick demos
- Testing
- No PostgreSQL available

**Use Database (server.py):**
- Production deployment ⭐
- Need historical data
- Compliance requirements
- Billing/reports
- Long-term operation

## 📝 Environment Variables (Optional)

Create a `.env` file to customize:

```bash
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=parking_db
DB_USER=parking_user
DB_PASS=Tenzin@2005
```

## ✅ Verification

Check everything is working:

```bash
# 1. Database connection
python -c "import psycopg2; conn = psycopg2.connect(host='127.0.0.1', port='5432', dbname='parking_db', user='parking_user', password='Tenzin@2005'); print('✅ Database connected')"

# 2. Server health
curl http://localhost:5000/api/health

# 3. Frontend
# Open http://localhost:3000 in browser
```

---

**Need Help?**
- Check server logs for errors
- Verify database is running: `psctl status` (Linux) or Services (Windows)
- Ensure all ports are free (5000, 3000, 5432)
