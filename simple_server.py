"""
Simple Parking Duration Tracking API Server
In-memory storage - no database required
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from vehicle_generator import VehicleGenerator

app = Flask(__name__)
CORS(app)

# In-memory storage for active parking sessions
active_sessions = {}  # slot_id -> {vehicle_id, entry_time, slot_id}
slot_free_timestamps = {}  # slot_id -> timestamp when slot became free
violations = []  # In-memory violations storage

print("=" * 70)
print("PARKING DURATION TRACKER API - IN-MEMORY")
print("=" * 70)
print("✅ No database required - using in-memory storage")
print("=" * 70)

@app.route('/api/parking/entry', methods=['POST'])
def parking_entry():
    """Record vehicle entry into a parking slot"""
    data = request.get_json()
    
    slot_id = data.get('slot_id')
    vehicle_id = data.get('vehicle_id')
    entry_time_str = data.get('entry_time')
    
    if not slot_id or vehicle_id is None:
        return jsonify({'error': 'slot_id and vehicle_id are required'}), 400
    
    try:
        # Convert entry_time to datetime
        if entry_time_str:
            entry_time = datetime.fromisoformat(entry_time_str)
        else:
            entry_time = datetime.now()
        
        # If slot was previously free, check if it was a different vehicle
        slot_key = str(slot_id)
        is_new_vehicle = True
        
        if slot_key in active_sessions:
            # Check if it's a different vehicle in the same slot
            if active_sessions[slot_key]['vehicle_id'] != vehicle_id:
                # New vehicle in same slot - reset timer
                print(f"🔄 NEW VEHICLE: Slot {slot_id}, Old Vehicle {active_sessions[slot_key]['vehicle_id']}, New Vehicle {vehicle_id}")
                is_new_vehicle = True
            else:
                # Same vehicle - keep existing timer
                is_new_vehicle = False
        
        # Generate vehicle details for new vehicles
        if is_new_vehicle:
            vehicle_info = VehicleGenerator.generate_vehicle()
        else:
            # Keep existing vehicle details
            existing_session = active_sessions.get(slot_key, {})
            vehicle_info = {
                'car_type': existing_session.get('car_type'),
                'color': existing_session.get('color'),
                'number_plate': existing_session.get('number_plate')
            }
        
        # Create/update session with proper timing and vehicle details
        active_sessions[slot_key] = {
            'vehicle_id': vehicle_id,
            'slot_id': slot_key,
            'entry_time': entry_time.isoformat() if is_new_vehicle else active_sessions.get(slot_key, {}).get('entry_time', entry_time.isoformat()),
            'status': 'PARKED',
            'car_type': vehicle_info['car_type'],
            'color': vehicle_info['color'],
            'number_plate': vehicle_info['number_plate']
        }
        
        # Remove from free timestamps if it was free
        if slot_key in slot_free_timestamps:
            del slot_free_timestamps[slot_key]
        
        print(f"✅ ENTRY: Slot {slot_id}, Vehicle {vehicle_id}, Timer {'RESET' if is_new_vehicle else 'CONTINUED'}")
        
        return jsonify({
            'message': 'Parking entry recorded',
            'slot_id': slot_id,
            'vehicle_id': vehicle_id,
            'entry_time': active_sessions[slot_key]['entry_time']
        }), 201
        
    except Exception as e:
        print(f"❌ Error in parking_entry: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/exit', methods=['POST'])
def parking_exit():
    """Record vehicle exit from a parking slot"""
    data = request.get_json()
    
    slot_id = str(data.get('slot_id'))
    vehicle_id = data.get('vehicle_id')
    duration = data.get('duration')
    
    if not slot_id or vehicle_id is None:
        return jsonify({'error': 'slot_id and vehicle_id are required'}), 400
    
    try:
        if slot_id in active_sessions:
            session = active_sessions.pop(slot_id)
            
            # Mark the timestamp when this slot became free
            slot_free_timestamps[slot_id] = datetime.now().timestamp()
            
            print(f"✅ EXIT: Slot {slot_id}, Vehicle {vehicle_id}, Duration: {duration:.1f}s, FREE at {datetime.now().isoformat()}")
            
            return jsonify({
                'message': 'Parking exit recorded',
                'slot_id': slot_id,
                'vehicle_id': vehicle_id,
                'duration_seconds': duration,
                'free_since': slot_free_timestamps[slot_id]
            }), 200
        else:
            # Slot wasn't in active sessions, still mark it as free
            slot_free_timestamps[slot_id] = datetime.now().timestamp()
            return jsonify({'error': 'No active session found'}), 404
        
    except Exception as e:
        print(f"❌ Error in parking_exit: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/active', methods=['GET'])
def get_active_parking():
    """Get all currently active parking sessions"""
    try:
        result = []
        current_time = datetime.now()
        
        for slot_id, session in active_sessions.items():
            entry_time = datetime.fromisoformat(session['entry_time'])
            duration_seconds = (current_time - entry_time).total_seconds()
            
            result.append({
                'slot_id': slot_id,
                'vehicle_id': session['vehicle_id'],
                'entry_time': session['entry_time'],
                'duration_seconds': int(duration_seconds),
                'duration_minutes': int(duration_seconds / 60),
                'status': 'PARKED',
                'parking_area': 'Main Parking Area',  # Default parking area
                'car_type': session.get('car_type', 'Unknown'),
                'color': session.get('color', 'Unknown'),
                'number_plate': session.get('number_plate', 'N/A')
            })
        
        return jsonify(result), 200
    except Exception as e:
        print(f"❌ Error in get_active_parking: {e}")
        return jsonify([]), 200


@app.route('/api/parking/slots/status', methods=['GET'])
def get_all_slots_status():
    """Get status of all slots including free slots with 3-second green delay"""
    try:
        # Assume we have 24 slots (you can adjust this)
        total_slots = 24
        current_time = datetime.now().timestamp()
        
        all_slots = []
        
        for i in range(1, total_slots + 1):
            slot_id = str(i)
            
            if slot_id in active_sessions:
                # Slot is occupied
                session = active_sessions[slot_id]
                entry_time = datetime.fromisoformat(session['entry_time'])
                duration_seconds = (datetime.now() - entry_time).total_seconds()
                
                all_slots.append({
                    'slot_id': slot_id,
                    'vehicle_id': session['vehicle_id'],
                    'status': 'occupied',
                    'entry_time': session['entry_time'],
                    'duration_seconds': int(duration_seconds),
                    'duration_minutes': int(duration_seconds / 60),
                })
            else:
                # Slot is free
                free_since = slot_free_timestamps.get(slot_id)
                time_since_free = 999  # Default to large number (already green)
                
                if free_since:
                    time_since_free = current_time - free_since
                
                # Slot becomes green after 3 seconds
                is_green = time_since_free >= 3
                
                all_slots.append({
                    'slot_id': slot_id,
                    'vehicle_id': None,
                    'status': 'free',
                    'entry_time': None,
                    'duration_seconds': 0,
                    'duration_minutes': 0,
                    'free_since': free_since,
                    'is_green': is_green,
                    'time_until_green': max(0, 3 - time_since_free) if not is_green else 0
                })
        
        return jsonify(all_slots), 200
    except Exception as e:
        print(f"❌ Error in get_all_slots_status: {e}")
        return jsonify([]), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# VIOLATION ENDPOINTS - IN-MEMORY
# ============================================================================

@app.route('/api/violations', methods=['GET'])
def get_violations():
    """GET - Get all active violations"""
    active_violations = [v for v in violations if v.get('status') == 'ACTIVE']
    return jsonify(active_violations), 200


@app.route('/api/violations/summary', methods=['GET'])
def get_violations_summary():
    """GET - Get violation statistics"""
    active = [v for v in violations if v.get('status') == 'ACTIVE']
    
    high = len([v for v in active if v.get('severity') == 'High'])
    medium = len([v for v in active if v.get('severity') == 'Medium'])
    low = len([v for v in active if v.get('severity') == 'Low'])
    
    areas = set(v.get('parking_area') for v in active)
    
    return jsonify({
        'total_violations': len(active),
        'high_severity': high,
        'medium_severity': medium,
        'low_severity': low,
        'areas_affected': len(areas),
        'affected_area_names': list(areas)
    }), 200


@app.route('/api/violations/from-video', methods=['POST'])
def receive_violation_from_video():
    """POST - Receive violation from video processing"""
    try:
        violation = request.get_json()
        
        # Check if violation already exists
        violation_id = violation.get('violation_id')
        existing = next((v for v in violations if v.get('violation_id') == violation_id), None)
        
        if existing:
            # Update existing violation
            existing.update(violation)
            existing['detected_at'] = datetime.now().isoformat()
        else:
            # Add new violation
            violation['detected_at'] = datetime.now().isoformat()
            violation['status'] = 'ACTIVE'
            violations.append(violation)
        
        # Keep only last 100 violations
        if len(violations) > 100:
            violations[:] = violations[-100:]
        
        return jsonify({
            'message': 'Violation received',
            'violation_id': violation_id
        }), 201
        
    except Exception as e:
        print(f"Error receiving violation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/resolve/<violation_id>', methods=['POST'])
def resolve_violation(violation_id):
    """POST - Resolve a violation"""
    violation = next((v for v in violations if v.get('violation_id') == violation_id), None)
    if violation:
        violation['status'] = 'RESOLVED'
        violation['resolved_at'] = datetime.now().isoformat()
        return jsonify({'message': 'Violation resolved'}), 200
    return jsonify({'error': 'Violation not found'}), 404


@app.after_request
def after_request(response):
    """Add headers to prevent caching and enable keep-alive"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'message': 'Parking Duration Tracker API',
        'version': '1.0 - In-Memory',
        'endpoints': {
            'POST /api/parking/entry': 'Record vehicle entry',
            'POST /api/parking/exit': 'Record vehicle exit',
            'GET /api/parking/active': 'Get active sessions',
            'GET /api/violations': 'Get active violations',
            'GET /api/violations/summary': 'Get violation statistics',
            'POST /api/violations/from-video': 'Receive violation from video',
            'GET /api/health': 'Health check'
        },
        'active_sessions': len(active_sessions),
        'active_violations': len([v for v in violations if v.get('status') == 'ACTIVE'])
    }), 200


if __name__ == '__main__':
    print("\n🚀 Starting Parking Duration Tracker API...")
    print("📍 Server: http://localhost:5000")
    print("📖 Documentation: http://localhost:5000/")
    print("📹 Receiving duration tracking + violations from video")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
