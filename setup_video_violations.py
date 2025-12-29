"""
Test Video-Based Violation Detection
Clear existing violations and run video processing
"""
import psycopg2
import os

# Database connection settings
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')

print("=" * 70)
print("VIDEO-BASED VIOLATION DETECTION - SETUP")
print("=" * 70)

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cur = conn.cursor()
    
    # Clear existing violations
    cur.execute("DELETE FROM parking_violations")
    count = cur.rowcount
    conn.commit()
    
    print(f"\nCleared {count} existing violations")
    print("\nReady for video-based violation detection!")
    print("\nNext steps:")
    print("  1. Start backend: python server.py")
    print("  2. Run video processing: python smart_parking_mvp.py --run parking_evening_vedio.mp4")
    print("  3. Start frontend: cd ../Frontend && npm run dev")
    print("  4. View violations at: http://localhost:3000/admin")
    print("\nViolations will be detected from:")
    print("  - Duration: vehicles parked > 2 minutes (Low)")
    print("  - Duration: vehicles parked > 4 minutes (Medium)")
    print("  - Duration: vehicles parked > 8 minutes (High)")
    print("  - Double parking: multiple vehicles in one slot")
    print("  - Outside boundary: vehicles parked over lines")
    print("\nViolations are checked every 30 frames (~2 seconds)")
    print("=" * 70)
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
