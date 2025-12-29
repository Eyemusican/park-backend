from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import psycopg2
import os
from datetime import datetime, timedelta
import cv2
import json
import threading
import time
from ultralytics import YOLO
import numpy as np
from shapely.geometry import Polygon
from violation_detector import ViolationDetector, ViolationType, Severity
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for video streaming
current_frame = None
frame_lock = threading.Lock()
video_running = False
parking_slots = []
model = None

# In-memory storage for parking sessions (duration tracking)
active_parking_sessions = {}  # slot_id -> {vehicle_id, entry_time, slot_id}

# Violation detector instance
violation_detector = ViolationDetector()

# Background thread for automatic violation checking
violation_check_running = False


def auto_check_violations():
    """Background task to automatically check for violations every 60 seconds"""
    global violation_check_running
    
    while violation_check_running:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Get all active parking events with duration
            cur.execute('''
                SELECT 
                    pe.event_id,
                    pe.slot_id,
                    ps.slot_number,
                    pa.parking_name,
                    pa.parking_id,
                    pe.arrival_time,
                    EXTRACT(EPOCH FROM (NOW() - pe.arrival_time))/60 as duration_minutes
                FROM parking_events pe
                JOIN parking_slots ps ON pe.slot_id = ps.slot_id
                JOIN parking_area pa ON ps.parking_id = pa.parking_id
                WHERE pe.departure_time IS NULL
            ''')
            
            active_sessions = cur.fetchall()
            
            for session in active_sessions:
                event_id, slot_id, slot_number, parking_name, parking_id, arrival_time, duration_minutes = session
                
                # Check for duration violation
                violation = violation_detector.check_duration_violation(
                    slot_id=f"{parking_name.split()[0]}-{str(slot_number).zfill(3)}",
                    vehicle_id=str(event_id),
                    duration_minutes=float(duration_minutes),
                    parking_area=parking_name,
                    license_plate=f"VP-{event_id}"
                )
                
                if violation:
                    try:
                        # Save/update violation in database
                        cur.execute('''
                            INSERT INTO parking_violations 
                            (violation_id, slot_id, vehicle_id, license_plate, violation_type, 
                             severity, description, duration_minutes, detected_at, status, parking_area)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (violation_id) DO UPDATE SET
                                duration_minutes = EXCLUDED.duration_minutes,
                                description = EXCLUDED.description,
                                severity = EXCLUDED.severity
                        ''', (
                            violation['violation_id'],
                            slot_id,
                            violation['vehicle_id'],
                            violation['license_plate'],
                            violation['violation_type'],
                            violation['severity'],
                            violation['description'],
                            violation['duration_minutes'],
                            datetime.now(),
                            'ACTIVE',
                            parking_name
                        ))
                    except Exception as e:
                        print(f"Error saving violation: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"✓ Checked {len(active_sessions)} active parking sessions for violations")
            
        except Exception as e:
            print(f"Error in auto violation check: {e}")
        
        # Wait 60 seconds before next check
        time.sleep(60)


def start_violation_checker():
    """Start the background violation checker"""
    global violation_check_running
    
    if not violation_check_running:
        violation_check_running = True
        checker_thread = threading.Thread(target=auto_check_violations, daemon=True)
        checker_thread.start()
        print("✅ Automatic violation checker started")



# Database connection settings
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')

print("=" * 50)
print("PARKING API SERVER - FULL CRUD")
print("=" * 50)
print("DATABASE CONFIGURATION:")
print(f"Host: {DB_HOST}")
print(f"Port: {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}")
print("=" * 50)

# Helper function to get a database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

# ============================================================================
# PARKING AREA ENDPOINTS - FULL CRUD
# ============================================================================

@app.route('/api/parking-areas', methods=['GET'])
def get_parking_areas():
    """GET - Get all parking areas with current stats"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get parking areas with accurate occupied count
        cur.execute('''
            SELECT 
                pa.parking_id,
                pa.parking_name,
                pa.slot_count,
                (
                    SELECT COUNT(DISTINCT ps2.slot_id)
                    FROM parking_slots ps2
                    WHERE ps2.parking_id = pa.parking_id
                    AND EXISTS (
                        SELECT 1 FROM parking_events pe2
                        WHERE pe2.slot_id = ps2.slot_id
                        AND pe2.departure_time IS NULL
                    )
                ) as occupied_count
            FROM parking_area pa
            ORDER BY pa.parking_id
        ''')
        
        areas = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for area in areas:
            total_slots = area[2] or 0
            occupied = area[3] or 0
            result.append({
                'id': area[0],
                'name': area[1],
                'total_slots': total_slots,
                'occupied_slots': occupied,
                'available_slots': total_slots - occupied,
                'occupancy_rate': (occupied / total_slots * 100) if total_slots > 0 else 0
            })
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-areas/<int:parking_id>', methods=['GET'])
def get_parking_area(parking_id):
    """GET - Get specific parking area details"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get parking area info
        cur.execute('SELECT parking_id, parking_name, slot_count FROM parking_area WHERE parking_id = %s', (parking_id,))
        area = cur.fetchone()
        
        if not area:
            return jsonify({'error': 'Parking area not found'}), 404
        
        # Get occupied slots count
        cur.execute('''
            SELECT COUNT(*) FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            WHERE ps.parking_id = %s AND pe.departure_time IS NULL
        ''', (parking_id,))
        occupied = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        total_slots = area[2] or 0
        result = {
            'id': area[0],
            'name': area[1],
            'total_slots': total_slots,
            'occupied_slots': occupied,
            'available_slots': total_slots - occupied,
            'occupancy_rate': (occupied / total_slots * 100) if total_slots > 0 else 0
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-areas', methods=['POST'])
def create_parking_area():
    """CREATE - Add a new parking area"""
    data = request.get_json()
    parking_name = data.get('parking_name')
    slot_count = data.get('slot_count', 0)
    
    if not parking_name:
        return jsonify({'error': 'parking_name is required'}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert parking area
        cur.execute(
            'INSERT INTO parking_area (parking_name, slot_count) VALUES (%s, %s) RETURNING parking_id',
            (parking_name, slot_count)
        )
        parking_id = cur.fetchone()[0]
        
        # Create slots if slot_count is provided
        if slot_count and slot_count > 0:
            for i in range(1, slot_count + 1):
                cur.execute(
                    'INSERT INTO parking_slots (parking_id, slot_number) VALUES (%s, %s)',
                    (parking_id, i)
                )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'id': parking_id,
            'name': parking_name,
            'total_slots': slot_count,
            'message': 'Parking area created successfully'
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-areas/<int:parking_id>', methods=['PUT'])
def update_parking_area(parking_id):
    """UPDATE - Update parking area details"""
    data = request.get_json()
    parking_name = data.get('parking_name')
    slot_count = data.get('slot_count')
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if parking area exists
        cur.execute('SELECT parking_id FROM parking_area WHERE parking_id = %s', (parking_id,))
        if not cur.fetchone():
            return jsonify({'error': 'Parking area not found'}), 404
        
        # Update parking area
        update_fields = []
        params = []
        
        if parking_name:
            update_fields.append('parking_name = %s')
            params.append(parking_name)
        
        if slot_count is not None:
            update_fields.append('slot_count = %s')
            params.append(slot_count)
        
        if not update_fields:
            return jsonify({'error': 'No fields to update'}), 400
        
        params.append(parking_id)
        query = f"UPDATE parking_area SET {', '.join(update_fields)} WHERE parking_id = %s"
        cur.execute(query, params)
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'message': 'Parking area updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-areas/<int:parking_id>', methods=['DELETE'])
def delete_parking_area(parking_id):
    """DELETE - Delete parking area and all related data"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if parking area exists
        cur.execute('SELECT parking_id FROM parking_area WHERE parking_id = %s', (parking_id,))
        if not cur.fetchone():
            return jsonify({'error': 'Parking area not found'}), 404
        
        # Delete parking events first (foreign key constraint)
        cur.execute('''
            DELETE FROM parking_events 
            WHERE slot_id IN (
                SELECT slot_id FROM parking_slots WHERE parking_id = %s
            )
        ''', (parking_id,))
        
        # Delete parking slots
        cur.execute('DELETE FROM parking_slots WHERE parking_id = %s', (parking_id,))
        
        # Delete parking area
        cur.execute('DELETE FROM parking_area WHERE parking_id = %s', (parking_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'message': 'Parking area deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PARKING SLOTS ENDPOINTS - FULL CRUD
# ============================================================================

@app.route('/api/parking-areas/<int:parking_id>/slots', methods=['GET'])
def get_parking_slots(parking_id):
    """GET - Get all slots for a parking area with current status and accurate duration"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get parking area name for reference
        cur.execute('SELECT parking_name FROM parking_area WHERE parking_id = %s', (parking_id,))
        area_result = cur.fetchone()
        parking_name = area_result[0] if area_result else "Unknown Area"
        
        cur.execute('''
            SELECT 
                ps.slot_id,
                ps.slot_number,
                pe.event_id,
                pe.arrival_time,
                CASE WHEN pe.event_id IS NOT NULL AND pe.departure_time IS NULL THEN true ELSE false END as is_occupied,
                EXTRACT(EPOCH FROM (NOW() - pe.arrival_time))/60 as duration_minutes
            FROM parking_slots ps
            LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
            WHERE ps.parking_id = %s
            ORDER BY ps.slot_number
        ''', (parking_id,))
        
        slots = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for slot in slots:
            is_occupied = slot[4] or False
            duration_mins = int(slot[5]) if slot[5] else 0
            
            slot_data = {
                'slot_id': slot[0],
                'slot_number': slot[1],
                'is_occupied': is_occupied,
                'event_id': slot[2],
                'arrival_time': slot[3].isoformat() if slot[3] else None,
                'duration_minutes': duration_mins if is_occupied else None,
                'parking_name': parking_name,
                'vehicle_id': f"V-{slot[2]}" if slot[2] else None,
            }
            
            result.append(slot_data)
        
        return jsonify(result), 200
    except Exception as e:
        print(f"❌ Error in get_parking_slots: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-areas/<int:parking_id>/slots', methods=['POST'])
def create_parking_slot(parking_id):
    """CREATE - Add a new slot to parking area"""
    data = request.get_json()
    slot_number = data.get('slot_number')
    
    if not slot_number:
        return jsonify({'error': 'slot_number is required'}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if parking area exists
        cur.execute('SELECT parking_id FROM parking_area WHERE parking_id = %s', (parking_id,))
        if not cur.fetchone():
            return jsonify({'error': 'Parking area not found'}), 404
        
        # Insert slot
        cur.execute(
            'INSERT INTO parking_slots (parking_id, slot_number) VALUES (%s, %s) RETURNING slot_id',
            (parking_id, slot_number)
        )
        slot_id = cur.fetchone()[0]
        
        # Update slot count
        cur.execute(
            'UPDATE parking_area SET slot_count = (SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s) WHERE parking_id = %s',
            (parking_id, parking_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'slot_id': slot_id,
            'slot_number': slot_number,
            'message': 'Slot created successfully'
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/slots/<int:slot_id>', methods=['DELETE'])
def delete_parking_slot(slot_id):
    """DELETE - Delete a parking slot"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get parking_id before deletion
        cur.execute('SELECT parking_id FROM parking_slots WHERE slot_id = %s', (slot_id,))
        result = cur.fetchone()
        
        if not result:
            return jsonify({'error': 'Slot not found'}), 404
        
        parking_id = result[0]
        
        # Delete related events
        cur.execute('DELETE FROM parking_events WHERE slot_id = %s', (slot_id,))
        
        # Delete slot
        cur.execute('DELETE FROM parking_slots WHERE slot_id = %s', (slot_id,))
        
        # Update slot count
        cur.execute(
            'UPDATE parking_area SET slot_count = (SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s) WHERE parking_id = %s',
            (parking_id, parking_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'message': 'Slot deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-slots', methods=['GET'])
def get_all_parking_slots():
    """GET - Get all parking slots across all areas"""
    try:
        parking_id = request.args.get('parking_id', type=int)
        conn = get_db_connection()
        cur = conn.cursor()
        
        if parking_id:
            # Get slots for specific parking area
            cur.execute('''
                SELECT 
                    ps.slot_id,
                    ps.slot_number,
                    ps.parking_id,
                    pa.parking_name,
                    pe.event_id,
                    pe.arrival_time,
                    CASE WHEN pe.event_id IS NOT NULL AND pe.departure_time IS NULL THEN true ELSE false END as is_occupied,
                    EXTRACT(EPOCH FROM (NOW() - pe.arrival_time))/60 as duration_minutes
                FROM parking_slots ps
                JOIN parking_area pa ON ps.parking_id = pa.parking_id
                LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
                WHERE ps.parking_id = %s
                ORDER BY ps.slot_number
            ''', (parking_id,))
        else:
            # Get all slots
            cur.execute('''
                SELECT 
                    ps.slot_id,
                    ps.slot_number,
                    ps.parking_id,
                    pa.parking_name,
                    pe.event_id,
                    pe.arrival_time,
                    CASE WHEN pe.event_id IS NOT NULL AND pe.departure_time IS NULL THEN true ELSE false END as is_occupied,
                    EXTRACT(EPOCH FROM (NOW() - pe.arrival_time))/60 as duration_minutes
                FROM parking_slots ps
                JOIN parking_area pa ON ps.parking_id = pa.parking_id
                LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
                ORDER BY ps.parking_id, ps.slot_number
            ''')
        
        slots = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for slot in slots:
            is_occupied = slot[6] or False
            duration_mins = int(slot[7]) if slot[7] else 0
            result.append({
                'slot_id': slot[0],
                'slot_number': slot[1],
                'parking_id': slot[2],
                'parking_name': slot[3],
                'event_id': slot[4],
                'status': 'occupied' if is_occupied else 'free',
                'arrival_time': slot[5].isoformat() if slot[5] else None,
                'duration_minutes': duration_mins
            })
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking/slots/status', methods=['GET'])
def get_parking_slots_status():
    """GET - Get real-time status of all parking slots for live view"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                ps.slot_id,
                ps.slot_number,
                pa.parking_name,
                pe.event_id,
                pe.arrival_time,
                CASE WHEN pe.event_id IS NOT NULL AND pe.departure_time IS NULL THEN 'occupied' ELSE 'free' END as status,
                EXTRACT(EPOCH FROM (NOW() - pe.arrival_time)) as duration_seconds
            FROM parking_slots ps
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
            ORDER BY pa.parking_id, ps.slot_number
        ''')
        
        slots = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for slot in slots:
            duration_seconds = int(slot[6]) if slot[6] else 0
            duration_minutes = duration_seconds // 60
            
            slot_number = slot[1]
            slot_id_str = f"{slot[2]}-{slot_number:03d}"
            
            # Get vehicle details from active sessions (in-memory)
            session = active_parking_sessions.get(str(slot_number))
            
            slot_data = {
                'slot_id': slot_id_str,
                'vehicle_id': slot[3],
                'status': slot[5],
                'entry_time': slot[4].isoformat() if slot[4] else None,
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_minutes,
                # Add vehicle details
                'license_plate': session.get('license_plate', 'N/A') if session else 'N/A',
                'color': session.get('color', 'unknown') if session else 'unknown',
                'vehicle_type': session.get('vehicle_type', 'car') if session else 'car'
            }
            result.append(slot_data)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PARKING EVENTS ENDPOINTS - CRUD
# ============================================================================

@app.route('/api/parking-events', methods=['GET'])
def get_parking_events():
    """GET - Get parking events with filters and accurate duration calculation"""
    try:
        limit = request.args.get('limit', 100, type=int)
        parking_id = request.args.get('parking_id', type=int)
        status = request.args.get('status')  # 'occupied' or 'departed'
        active = request.args.get('active', 'false').lower() == 'true'
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        query = '''
            SELECT 
                pe.event_id,
                ps.slot_id,
                ps.slot_number,
                pa.parking_name,
                pa.parking_id,
                pe.arrival_time,
                pe.departure_time,
                pe.parked_time,
                EXTRACT(EPOCH FROM (
                    CASE 
                        WHEN pe.departure_time IS NULL THEN NOW() - pe.arrival_time
                        ELSE pe.departure_time - pe.arrival_time
                    END
                ))/60 as current_duration_minutes
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE 1=1
        '''
        
        params = []
        
        if parking_id:
            query += ' AND pa.parking_id = %s'
            params.append(parking_id)
        
        if status == 'occupied' or active:
            query += ' AND pe.departure_time IS NULL'
        elif status == 'departed':
            query += ' AND pe.departure_time IS NOT NULL'
        
        query += ' ORDER BY pe.arrival_time DESC LIMIT %s'
        params.append(limit)
        
        cur.execute(query, params)
        events = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for event in events:
            is_active = event[6] is None  # No departure time means active
            
            event_data = {
                'event_id': event[0],
                'slot_id': event[1],
                'slot_number': event[2],
                'parking_name': event[3],
                'parking_id': event[4],
                'arrival_time': event[5].isoformat() if event[5] else None,
                'departure_time': event[6].isoformat() if event[6] else None,
                'parked_time_minutes': event[7],
                'duration_minutes': int(event[8]) if event[8] else 0,
                'status': 'occupied' if is_active else 'departed',
                'vehicle_id': f"V-{event[0]}",
                'license_plate': f"LP-{event[0]:04d}",
            }
            
            result.append(event_data)
        
        return jsonify(result), 200
    except Exception as e:
        print(f"❌ Error in get_parking_events: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-events', methods=['POST'])
def create_parking_event():
    """CREATE - Manually create a parking event (arrival)"""
    data = request.get_json()
    slot_id = data.get('slot_id')
    
    if not slot_id:
        return jsonify({'error': 'slot_id is required'}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if slot exists
        cur.execute('SELECT slot_id FROM parking_slots WHERE slot_id = %s', (slot_id,))
        if not cur.fetchone():
            return jsonify({'error': 'Slot not found'}), 404
        
        # Check if slot is already occupied
        cur.execute(
            'SELECT event_id FROM parking_events WHERE slot_id = %s AND departure_time IS NULL',
            (slot_id,)
        )
        if cur.fetchone():
            return jsonify({'error': 'Slot is already occupied'}), 400
        
        # Create event
        cur.execute(
            'INSERT INTO parking_events (slot_id, arrival_time) VALUES (%s, %s) RETURNING event_id',
            (slot_id, datetime.now())
        )
        event_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'event_id': event_id,
            'message': 'Parking event created successfully'
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking-events/<int:event_id>/departure', methods=['PUT'])
def record_departure(event_id):
    """UPDATE - Record departure for a parking event"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get event details
        cur.execute(
            'SELECT event_id, arrival_time, departure_time FROM parking_events WHERE event_id = %s',
            (event_id,)
        )
        event = cur.fetchone()
        
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        if event[2]:
            return jsonify({'error': 'Departure already recorded'}), 400
        
        # Record departure
        departure_time = datetime.now()
        parked_time = int((departure_time - event[1]).total_seconds() / 60)
        
        cur.execute(
            'UPDATE parking_events SET departure_time = %s, parked_time = %s WHERE event_id = %s',
            (departure_time, parked_time, event_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'message': 'Departure recorded successfully',
            'parked_time_minutes': parked_time
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/parking-events/<int:event_id>', methods=['DELETE'])
def delete_parking_event(event_id):
    """DELETE - Delete a parking event"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('DELETE FROM parking_events WHERE event_id = %s', (event_id,))
        
        if cur.rowcount == 0:
            return jsonify({'error': 'Event not found'}), 404
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'message': 'Event deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PARKING DURATION TRACKING ENDPOINTS
# ============================================================================

@app.route('/api/parking/entry', methods=['POST'])
def parking_entry():
    """Record vehicle entry into a parking slot - saves to database"""
    data = request.get_json()
    
    slot_id = data.get('slot_id')
    vehicle_id = data.get('vehicle_id')
    entry_time_str = data.get('entry_time')
    license_plate = data.get('license_plate')
    color = data.get('color')
    vehicle_type = data.get('vehicle_type')
    
    if not slot_id or vehicle_id is None:
        return jsonify({'error': 'slot_id and vehicle_id are required'}), 400
    
    try:
        # Parse entry time
        if entry_time_str:
            entry_time = datetime.fromisoformat(entry_time_str)
        else:
            entry_time = datetime.now()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Convert slot_id from video (slot number 1-8) to database slot_id
        # The video sends slot_number, we need to look up the actual slot_id
        slot_number = int(slot_id)
        
        # Get the actual database slot_id for this slot_number
        cur.execute(
            '''SELECT slot_id FROM parking_slots 
               WHERE slot_number = %s 
               LIMIT 1''',
            (slot_number,)
        )
        result = cur.fetchone()
        if not result:
            cur.close()
            conn.close()
            return jsonify({'error': f'Slot number {slot_number} not found'}), 404
        
        db_slot_id = result[0]
        
        # Check if there's already an active event for this slot
        cur.execute(
            '''SELECT event_id FROM parking_events 
               WHERE slot_id = %s AND departure_time IS NULL''',
            (db_slot_id,)
        )
        existing_event = cur.fetchone()
        
        if existing_event:
            # Update existing event (vehicle re-entry or ID change)
            cur.execute(
                '''UPDATE parking_events 
                   SET arrival_time = %s 
                   WHERE event_id = %s''',
                (entry_time, existing_event[0])
            )
            event_id = existing_event[0]
            print(f"✅ Entry updated: Slot {slot_id}, Vehicle {vehicle_id} (Event ID: {event_id})")
        else:
            # Create new parking event
            cur.execute(
                '''INSERT INTO parking_events (slot_id, arrival_time, departure_time, parked_time)
                   VALUES (%s, %s, NULL, NULL)
                   RETURNING event_id''',
                (db_slot_id, entry_time)
            )
            event_id = cur.fetchone()[0]
            print(f"✅ Entry recorded: Slot {slot_id}, Vehicle {vehicle_id} (Event ID: {event_id})")
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Also store in memory for quick access with vehicle details
        # Store with both the slot_id string and the slot_number for lookup flexibility
        session_data = {
            'vehicle_id': vehicle_id,
            'slot_id': slot_id,
            'slot_number': slot_number,
            'entry_time': entry_time.isoformat(),
            'status': 'PARKED',
            'event_id': event_id,
            'license_plate': license_plate or 'N/A',
            'color': color or 'unknown',
            'vehicle_type': vehicle_type or 'car'
        }
        
        # Store with both string slot_id and integer slot_number as keys
        active_parking_sessions[str(slot_id)] = session_data
        active_parking_sessions[str(slot_number)] = session_data
        
        vehicle_info = f" {vehicle_type or 'car'} | {color or 'unknown'} | {license_plate or 'N/A'}"
        print(f"  Vehicle Details:{vehicle_info}")
        print(f"  Stored in active_parking_sessions['{slot_id}'] and ['{slot_number}']")
        
        return jsonify({
            'message': 'Parking entry recorded',
            'event_id': event_id,
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'entry_time': entry_time.isoformat(),
            'license_plate': license_plate,
            'color': color,
            'vehicle_type': vehicle_type
        }), 201
        
    except Exception as e:
        print(f"Error in parking_entry: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/exit', methods=['POST'])
def parking_exit():
    """Record vehicle exit from a parking slot"""
    data = request.get_json()
    
    slot_id = data.get('slot_id')
    vehicle_id = data.get('vehicle_id')
    exit_time_str = data.get('exit_time')
    duration = data.get('duration')  # duration in seconds
    
    if not slot_id or vehicle_id is None:
        return jsonify({'error': 'slot_id and vehicle_id are required'}), 400
    
    try:
        # Parse exit time
        if exit_time_str:
            exit_time = datetime.fromisoformat(exit_time_str)
        else:
            exit_time = datetime.now()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Convert slot_id from video (slot number 1-8) to database slot_id
        slot_number = int(slot_id)
        
        # Get the actual database slot_id for this slot_number
        cur.execute(
            '''SELECT slot_id FROM parking_slots 
               WHERE slot_number = %s 
               LIMIT 1''',
            (slot_number,)
        )
        result = cur.fetchone()
        if not result:
            cur.close()
            conn.close()
            return jsonify({'error': f'Slot number {slot_number} not found'}), 404
        
        db_slot_id = result[0]
        
        # Find the active parking event for this slot
        cur.execute(
            '''SELECT event_id, arrival_time FROM parking_events 
               WHERE slot_id = %s AND departure_time IS NULL 
               ORDER BY arrival_time DESC LIMIT 1''',
            (db_slot_id,)
        )
        event = cur.fetchone()
        
        if not event:
            return jsonify({'error': 'No active parking session found for this slot'}), 404
        
        event_id = event[0]
        arrival_time = event[1]
        
        # Calculate parking duration in minutes
        if duration:
            parked_minutes = int(duration / 60)
        else:
            parked_minutes = int((exit_time - arrival_time).total_seconds() / 60)
        
        # Update parking event with departure
        cur.execute(
            'UPDATE parking_events SET departure_time = %s, parked_time = %s WHERE event_id = %s',
            (exit_time, parked_minutes, event_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Remove from active sessions
        if str(slot_id) in active_parking_sessions:
            del active_parking_sessions[str(slot_id)]
        
        print(f"✅ Exit recorded: Slot {slot_id}, Vehicle {vehicle_id}, Duration: {parked_minutes}min")
        
        return jsonify({
            'message': 'Parking exit recorded',
            'event_id': event_id,
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'exit_time': exit_time.isoformat(),
            'duration_minutes': parked_minutes
        }), 200
        
    except Exception as e:
        print(f"Error in parking_exit: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/active', methods=['GET'])
def get_active_parking():
    """Get all currently active parking sessions"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if tables exist first
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'parking_events'
            )
        """)
        tables_exist = cur.fetchone()[0]
        
        if not tables_exist:
            # Tables don't exist, return empty array
            cur.close()
            conn.close()
            return jsonify([]), 200
        
        cur.execute('''
            SELECT 
                pe.event_id,
                ps.slot_number,
                pa.parking_name,
                pe.arrival_time,
                EXTRACT(EPOCH FROM (NOW() - pe.arrival_time)) as duration_seconds
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE pe.departure_time IS NULL
            ORDER BY pe.arrival_time DESC
        ''')
        
        active = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for row in active:
            duration_seconds = row[4] or 0
            slot_number = row[1]
            
            # Get vehicle details from in-memory sessions
            # Try both string formats
            session = active_parking_sessions.get(str(slot_number))
            
            result.append({
                'event_id': row[0],
                'vehicle_id': row[0],  # Use event_id as vehicle_id
                'slot_id': f"{row[2]}-{slot_number:03d}",
                'parking_area': row[2],
                'slot_number': slot_number,
                'entry_time': row[3].isoformat(),
                'duration_seconds': int(duration_seconds),
                'duration_minutes': int(duration_seconds / 60),
                'status': 'PARKED',
                # Add vehicle details from session
                'license_plate': session.get('license_plate', 'N/A') if session else 'N/A',
                'vehicle_type': session.get('vehicle_type', 'unknown') if session else 'unknown',
                'color': session.get('color', 'unknown') if session else 'unknown'
            })
        
        return jsonify(result), 200
    except psycopg2.OperationalError as e:
        # Database connection error - return empty array instead of error
        print(f"❌ Database connection error: {e}")
        return jsonify([]), 200
    except Exception as e:
        print(f"❌ Error in get_active_parking: {e}")
        # Return empty array instead of error to prevent frontend from breaking
        return jsonify([]), 200


# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.route('/api/stats/overview', methods=['GET'])
def get_overview_stats():
    """GET - Overall system statistics with accurate real-time data"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Total parking areas
        cur.execute('SELECT COUNT(*) FROM parking_area')
        total_areas = cur.fetchone()[0] or 0
        
        # Total slots
        cur.execute('SELECT COALESCE(SUM(slot_count), 0) FROM parking_area')
        total_slots = cur.fetchone()[0] or 0
        
        # Currently occupied (with NULL check)
        cur.execute('SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL')
        occupied = cur.fetchone()[0] or 0
        
        # Active violations
        cur.execute("SELECT COUNT(*) FROM parking_violations WHERE status = 'ACTIVE'")
        active_violations = cur.fetchone()[0] or 0
        
        # Get areas with detailed stats and accurate calculations
        cur.execute('''
            SELECT 
                pa.parking_id as id,
                pa.parking_name as name,
                COALESCE(pa.slot_count, 0) as total_slots,
                COALESCE(COUNT(CASE WHEN pe.departure_time IS NULL THEN 1 END), 0) as occupied_slots,
                COALESCE(pa.slot_count, 0) - COALESCE(COUNT(CASE WHEN pe.departure_time IS NULL THEN 1 END), 0) as available_slots,
                CASE 
                    WHEN COALESCE(pa.slot_count, 0) > 0 THEN 
                        (COALESCE(COUNT(CASE WHEN pe.departure_time IS NULL THEN 1 END), 0)::float / pa.slot_count * 100)
                    ELSE 0 
                END as occupancy_rate
            FROM parking_area pa
            LEFT JOIN parking_slots ps ON pa.parking_id = ps.parking_id
            LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
            GROUP BY pa.parking_id, pa.parking_name, pa.slot_count
            ORDER BY pa.parking_id
        ''')
        areas_data = cur.fetchall()
        areas = [
            {
                'id': int(row[0]),
                'name': str(row[1]),
                'total_slots': int(row[2]),
                'occupied_slots': int(row[3]),
                'available_slots': int(row[4]),
                'occupancy_rate': round(float(row[5]), 2)
            }
            for row in areas_data
        ]
        
        cur.close()
        conn.close()
        
        # Calculate overall occupancy rate safely
        overall_occupancy = (occupied / total_slots * 100) if total_slots > 0 else 0
        
        return jsonify({
            'total_areas': int(total_areas),
            'total_slots': int(total_slots),
            'total_occupied': int(occupied),
            'total_available': int(total_slots - occupied),
            'overall_occupancy_rate': round(overall_occupancy, 2),
            'active_events': int(occupied),
            'total_violations': int(active_violations),
            'areas': areas,
            'timestamp': datetime.now().isoformat(),
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/hourly', methods=['GET'])
def get_hourly_stats():
    """GET - Hourly statistics for today"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                EXTRACT(HOUR FROM arrival_time) as hour,
                COUNT(*) as arrivals
            FROM parking_events
            WHERE DATE(arrival_time) = CURRENT_DATE
            GROUP BY hour
            ORDER BY hour
        ''')
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        hourly_data = [{'hour': int(row[0]), 'arrivals': row[1]} for row in results]
        
        return jsonify(hourly_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# VIDEO STREAMING ENDPOINTS
# ============================================================================

def load_parking_config():
    """Load parking slots configuration"""
    global parking_slots
    config_path = 'configs/parking_slots.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            parking_slots = data.get('slots', [])
        print(f"✅ Loaded {len(parking_slots)} parking slots for visualization")
    else:
        print("⚠️  No parking slots config found, video will show detections only")

def initialize_model():
    """Initialize YOLO model for video processing"""
    global model
    if model is None:
        model_path = 'yolov8m.pt'
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
        else:
            print("⚠️  YOLO model not found, using default model")
            model = YOLO('yolov8n.pt')

def get_slot_occupancy(parking_id):
    """Get current slot occupancy from database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                ps.slot_number,
                CASE WHEN pe.event_id IS NOT NULL AND pe.departure_time IS NULL THEN true ELSE false END as is_occupied
            FROM parking_slots ps
            LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id AND pe.departure_time IS NULL
            WHERE ps.parking_id = %s
            ORDER BY ps.slot_number
        ''', (parking_id,))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return {slot_num: occupied for slot_num, occupied in results}
    except Exception as e:
        print(f"Error getting slot occupancy: {e}")
        return {}

def draw_parking_slots(frame, slots_data, occupancy_map):
    """Draw parking slot overlays on frame"""
    for slot in slots_data:
        points = np.array(slot['points'], np.int32)
        slot_id = slot['id']
        is_occupied = occupancy_map.get(slot_id, False)
        
        # Choose color based on occupancy
        color = (0, 0, 255) if is_occupied else (0, 255, 0)  # Red for occupied, Green for free
        
        # Draw polygon
        cv2.polylines(frame, [points], True, color, 2)
        
        # Fill polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add slot number
        center = points.mean(axis=0).astype(int)
        status_text = f"#{slot_id} {'OCC' if is_occupied else 'FREE'}"
        cv2.putText(frame, status_text, (center[0]-20, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_video_stream(video_source, parking_id=1):
    """Process video and update global frame"""
    global current_frame, video_running
    
    initialize_model()
    load_parking_config()
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {video_source}")
        video_running = False
        return
    
    print(f"✅ Video stream started: {video_source}")
    video_running = True
    frame_count = 0
    
    while video_running:
        ret, frame = cap.read()
        
        if not ret:
            # Loop video if it's a file
            if isinstance(video_source, str) and video_source.endswith(('.mp4', '.avi', '.mov')):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        frame_count += 1
        
        # Run detection every 3 frames for performance
        if frame_count % 3 == 0 and model is not None:
            results = model(frame, conf=0.25, verbose=False)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Only draw vehicles (car, motorcycle, bus, truck)
                    if cls in [2, 3, 5, 7]:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        label = f'{model.names[cls]} {conf:.2f}'
                        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw parking slots if available
        if parking_slots:
            occupancy = get_slot_occupancy(parking_id)
            draw_parking_slots(frame, parking_slots, occupancy)
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Parking ID: {parking_id}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Update global frame
        with frame_lock:
            current_frame = frame.copy()
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()
    print("Video stream stopped")
    video_running = False

def generate_frames():
    """Generate frames for video stream"""
    global current_frame
    
    while True:
        with frame_lock:
            if current_frame is None:
                # Send placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No video source active", (180, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
            else:
                ret, buffer = cv2.imencode('.jpg', current_frame)
        
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

@app.route('/api/video-feed')
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Add alternate route for compatibility
@app.route('/video_feed')
def video_feed_alt():
    """Video streaming endpoint (alternate route)"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video/start', methods=['POST'])
def start_video():
    """Start video stream"""
    global video_running
    
    data = request.get_json() or {}
    video_source = data.get('video_source', 0)  # 0 for webcam, or file path
    parking_id = data.get('parking_id', 1)
    
    if video_running:
        return jsonify({'error': 'Video already running'}), 400
    
    # Start video processing in background thread
    thread = threading.Thread(target=process_video_stream, args=(video_source, parking_id), daemon=True)
    thread.start()
    
    return jsonify({
        'message': 'Video stream started',
        'video_source': str(video_source),
        'parking_id': parking_id
    }), 200

@app.route('/api/video/stop', methods=['POST'])
def stop_video():
    """Stop video stream"""
    global video_running
    
    if not video_running:
        return jsonify({'error': 'No video running'}), 400
    
    video_running = False
    
    return jsonify({'message': 'Video stream stopped'}), 200

@app.route('/api/video/status', methods=['GET'])
def video_status():
    """Get video stream status"""
    return jsonify({
        'running': video_running,
        'has_frame': current_frame is not None
    }), 200

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.close()
        conn.close()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# ============================================================================
# VIOLATION DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/violations', methods=['GET'])
def get_violations():
    """GET - Get all active violations"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get violations from database
        cur.execute('''
            SELECT 
                pv.violation_id,
                pv.slot_id,
                ps.slot_number,
                pa.parking_name,
                pa.parking_id,
                pv.vehicle_id,
                pv.license_plate,
                pv.violation_type,
                pv.severity,
                pv.description,
                pv.duration_minutes,
                pv.detected_at,
                pv.status
            FROM parking_violations pv
            LEFT JOIN parking_slots ps ON pv.slot_id = ps.slot_id
            LEFT JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE pv.status = 'ACTIVE'
            ORDER BY pv.detected_at DESC
        ''')
        
        violations = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for v in violations:
            result.append({
                'violation_id': v[0],
                'slot_id': v[1],
                'slot_number': v[2],
                'parking_name': v[3],
                'parking_id': v[4],
                'vehicle_id': v[5],
                'license_plate': v[6],
                'violation_type': v[7],
                'severity': v[8],
                'description': v[9],
                'duration_minutes': float(v[10]) if v[10] else None,
                'detected_at': v[11].isoformat() if v[11] else None,
                'status': v[12]
            })
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error fetching violations: {e}")
        # Fallback to in-memory violations if DB fails
        return jsonify(violation_detector.get_active_violations()), 200


@app.route('/api/violations/summary', methods=['GET'])
def get_violations_summary():
    """GET - Get violation statistics and summary"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get summary from database
        cur.execute('''
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN severity = 'High' THEN 1 END) as high,
                COUNT(CASE WHEN severity = 'Medium' THEN 1 END) as medium,
                COUNT(CASE WHEN severity = 'Low' THEN 1 END) as low,
                COUNT(DISTINCT ps.parking_id) as areas_affected
            FROM parking_violations pv
            LEFT JOIN parking_slots ps ON pv.slot_id = ps.slot_id
            WHERE pv.status = 'ACTIVE'
        ''')
        
        summary = cur.fetchone()
        
        # Get affected area names
        cur.execute('''
            SELECT DISTINCT pa.parking_name
            FROM parking_violations pv
            JOIN parking_slots ps ON pv.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE pv.status = 'ACTIVE'
        ''')
        
        area_names = [row[0] for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'total_violations': summary[0] or 0,
            'high_severity': summary[1] or 0,
            'medium_severity': summary[2] or 0,
            'low_severity': summary[3] or 0,
            'areas_affected': summary[4] or 0,
            'affected_area_names': area_names
        }), 200
    except Exception as e:
        print(f"Error fetching violation summary: {e}")
        # Fallback to in-memory detector
        return jsonify(violation_detector.get_violation_summary()), 200


@app.route('/api/violations/area/<parking_name>', methods=['GET'])
def get_violations_by_area(parking_name):
    """GET - Get violations for a specific parking area"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT 
                pv.violation_id,
                pv.slot_id,
                ps.slot_number,
                pa.parking_name,
                pv.vehicle_id,
                pv.license_plate,
                pv.violation_type,
                pv.severity,
                pv.description,
                pv.duration_minutes,
                pv.detected_at,
                pv.status
            FROM parking_violations pv
            JOIN parking_slots ps ON pv.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE pa.parking_name = %s AND pv.status = 'ACTIVE'
            ORDER BY pv.detected_at DESC
        ''', (parking_name,))
        
        violations = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for v in violations:
            result.append({
                'violation_id': v[0],
                'slot_id': v[1],
                'slot_number': v[2],
                'parking_name': v[3],
                'vehicle_id': v[4],
                'license_plate': v[5],
                'violation_type': v[6],
                'severity': v[7],
                'description': v[8],
                'duration_minutes': float(v[9]) if v[9] else None,
                'detected_at': v[10].isoformat() if v[10] else None,
                'status': v[11]
            })
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error fetching violations for area {parking_name}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/check-duration', methods=['POST'])
def check_duration_violations():
    """POST - Check all active parking sessions for duration violations"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get all active parking events with duration
        cur.execute('''
            SELECT 
                pe.event_id,
                pe.slot_id,
                ps.slot_number,
                pa.parking_name,
                pa.parking_id,
                pe.arrival_time,
                EXTRACT(EPOCH FROM (NOW() - pe.arrival_time))/60 as duration_minutes
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            WHERE pe.departure_time IS NULL
        ''')
        
        active_sessions = cur.fetchall()
        violations_found = []
        
        for session in active_sessions:
            event_id, slot_id, slot_number, parking_name, parking_id, arrival_time, duration_minutes = session
            
            # Check for duration violation
            violation = violation_detector.check_duration_violation(
                slot_id=f"{parking_name}-{slot_number}",
                vehicle_id=str(event_id),
                duration_minutes=float(duration_minutes),
                parking_area=parking_name,
                license_plate=f"VP-{event_id}"
            )
            
            if violation:
                # Save to database
                try:
                    cur.execute('''
                        INSERT INTO parking_violations 
                        (violation_id, slot_id, vehicle_id, license_plate, violation_type, 
                         severity, description, duration_minutes, detected_at, status, parking_area)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (violation_id) DO UPDATE SET
                            duration_minutes = EXCLUDED.duration_minutes,
                            description = EXCLUDED.description,
                            severity = EXCLUDED.severity
                    ''', (
                        violation['violation_id'],
                        slot_id,
                        violation['vehicle_id'],
                        violation['license_plate'],
                        violation['violation_type'],
                        violation['severity'],
                        violation['description'],
                        violation['duration_minutes'],
                        datetime.now(),
                        'ACTIVE',
                        parking_name
                    ))
                    violations_found.append(violation)
                except Exception as e:
                    print(f"Error saving violation: {e}")
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'checked_sessions': len(active_sessions),
            'violations_found': len(violations_found),
            'violations': violations_found
        }), 200
    except Exception as e:
        print(f"Error checking duration violations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/resolve/<violation_id>', methods=['POST'])
def resolve_violation(violation_id):
    """POST - Mark a violation as resolved"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            UPDATE parking_violations
            SET status = 'RESOLVED', resolved_at = %s
            WHERE violation_id = %s
            RETURNING violation_id
        ''', (datetime.now(), violation_id))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        if result:
            violation_detector.resolve_violation(violation_id)
            return jsonify({
                'message': 'Violation resolved successfully',
                'violation_id': violation_id
            }), 200
        else:
            return jsonify({'error': 'Violation not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/from-video', methods=['POST'])
def receive_violation_from_video():
    """POST - Receive violation detected from video feed"""
    try:
        violation = request.get_json()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Save violation to database (or update if exists)
        cur.execute('''
            INSERT INTO parking_violations 
            (violation_id, slot_id, vehicle_id, license_plate, violation_type, 
             severity, description, duration_minutes, detected_at, status, parking_area)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (violation_id) DO UPDATE SET
                duration_minutes = EXCLUDED.duration_minutes,
                description = EXCLUDED.description,
                severity = EXCLUDED.severity,
                detected_at = EXCLUDED.detected_at
        ''', (
            violation.get('violation_id'),
            violation.get('slot_id'),
            violation.get('vehicle_id'),
            violation.get('license_plate'),
            violation.get('violation_type'),
            violation.get('severity'),
            violation.get('description'),
            violation.get('duration_minutes'),
            datetime.now(),
            'ACTIVE',
            violation.get('parking_area', 'Video Feed Area')
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'message': 'Violation received and saved',
            'violation_id': violation.get('violation_id')
        }), 201
        
    except Exception as e:
        print(f"Error receiving violation from video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'message': 'Parking Management API',
        'version': '1.0',
        'endpoints': {
            'parking_areas': {
                'GET /api/parking-areas': 'List all parking areas',
                'GET /api/parking-areas/<id>': 'Get parking area details',
                'POST /api/parking-areas': 'Create parking area',
                'PUT /api/parking-areas/<id>': 'Update parking area',
                'DELETE /api/parking-areas/<id>': 'Delete parking area'
            },
            'slots': {
                'GET /api/parking-areas/<id>/slots': 'Get all slots',
                'POST /api/parking-areas/<id>/slots': 'Create slot',
                'DELETE /api/slots/<id>': 'Delete slot'
            },
            'events': {
                'GET /api/parking-events': 'Get parking events',
                'POST /api/parking-events': 'Create event (arrival)',
                'PUT /api/parking-events/<id>/departure': 'Record departure',
                'DELETE /api/parking-events/<id>': 'Delete event'
            },
            'violations': {
                'GET /api/violations': 'Get all active violations',
                'GET /api/violations/summary': 'Get violation statistics',
                'GET /api/violations/area/<area>': 'Get violations by area',
                'POST /api/violations/check-duration': 'Check duration violations',
                'POST /api/violations/resolve/<id>': 'Resolve a violation'
            },
            'stats': {
                'GET /api/stats/overview': 'Overall statistics',
                'GET /api/stats/hourly': 'Hourly statistics'
            },
            'video': {
                'GET /api/video-feed': 'Live video stream',
                'POST /api/video/start': 'Start video processing',
                'POST /api/video/stop': 'Stop video processing',
                'GET /api/video/status': 'Video stream status'
            },
            'health': {
                'GET /api/health': 'Health check'
            }
        }
    }), 200

if __name__ == '__main__':
    print("\n🚀 Starting Parking Management API Server...")
    print("📍 Server: http://localhost:5000")
    print("📖 Documentation: http://localhost:5000/\n")
    print("📹 Receiving violations from video feed")
    print("   (Auto-checker disabled - using video detection)\n")
    
    # Note: Auto-checker disabled when using video feed
    # Uncomment below to enable manual/database violation checking
    # start_violation_checker()
    
    app.run(host='0.0.0.0', port=5000, debug=True)