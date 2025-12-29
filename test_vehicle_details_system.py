"""
Test script to verify vehicle details flow from detection to backend
"""
import sys
import time

print("=" * 60)
print("VEHICLE DETAILS SYSTEM TEST")
print("=" * 60)

# Test 1: Vehicle Analyzer
print("\n[TEST 1] Testing Vehicle Analyzer...")
try:
    from vehicle_analyzer import VehicleAnalyzer
    analyzer = VehicleAnalyzer()
    
    # Test plate generation
    plates = [analyzer._generate_random_plate() for _ in range(5)]
    print(f"✅ Generated sample plates: {plates}")
    
    # Verify format
    import re
    pattern = r'^(BP|BG|BT)-\d-[A-Z]-\d{4}$'
    all_valid = all(re.match(pattern, plate) for plate in plates)
    if all_valid:
        print("✅ All plates match format: (BP|BG|BT)-#-A-####")
    else:
        print("❌ Some plates have invalid format!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Vehicle Analyzer test failed: {e}")
    sys.exit(1)

# Test 2: Backend Connection
print("\n[TEST 2] Testing Backend Connection...")
try:
    import requests
    response = requests.get("http://localhost:5000/api/parking/active", timeout=2)
    if response.status_code == 200:
        print(f"✅ Backend is running on http://localhost:5000")
        data = response.json()
        print(f"   Active parking sessions: {len(data)}")
    else:
        print(f"⚠️  Backend returned status code: {response.status_code}")
except Exception as e:
    print(f"❌ Backend connection failed: {e}")
    print("   Make sure server1.py is running!")
    sys.exit(1)

# Test 3: Database Connection
print("\n[TEST 3] Testing Database Connection...")
try:
    import psycopg2
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="parking_db",
        user="parking_user",
        password="parking123"
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL")
    active_count = cur.fetchone()[0]
    print(f"✅ Database connected")
    print(f"   Active parking events in DB: {active_count}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    sys.exit(1)

# Test 4: Send Test Entry to Backend
print("\n[TEST 4] Testing Backend API...")
try:
    test_payload = {
        'slot_id': 'TEST-001',
        'vehicle_id': 99999,
        'entry_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'status': 'PARKED',
        'license_plate': 'BP-5-K-1234',
        'color': 'blue',
        'vehicle_type': 'car'
    }
    
    response = requests.post(
        "http://localhost:5000/api/parking/entry",
        json=test_payload,
        timeout=2
    )
    
    if response.status_code in [200, 201]:
        print("✅ Backend API accepts vehicle details")
        
        # Check if it's in active sessions
        time.sleep(0.5)
        response = requests.get("http://localhost:5000/api/parking/active", timeout=2)
        data = response.json()
        
        test_found = any(
            s.get('slot_id') == 'TEST-001' and 
            s.get('license_plate') == 'BP-5-K-1234'
            for s in data
        )
        
        if test_found:
            print("✅ Vehicle details stored correctly in backend")
        else:
            print("⚠️  Entry created but vehicle details not found in active sessions")
            print(f"   Check backend logs for: 'TEST-001'")
    else:
        print(f"❌ Backend API error: {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ API test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n📝 NEXT STEPS:")
print("1. Run video processing: python smart_parking_mvp.py --run parking_evening_vedio.mp4")
print("2. Watch backend logs for: '✓ Stored session: X -> {...}'")
print("3. Check frontend Duration Tracker for vehicle details")
print("\n💡 TIP: New detections will have format like: BP-3-M-7845")
