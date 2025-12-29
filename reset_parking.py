"""
Reset/Clean all parking sessions - Use before starting new video
"""
import psycopg2
import os
from datetime import datetime

def reset_parking_area(parking_id=7):
    """Clear all active parking sessions for a fresh start"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', '127.0.0.1'),
            port=os.environ.get('DB_PORT', '5432'),
            dbname=os.environ.get('DB_NAME', 'parking_db'),
            user=os.environ.get('DB_USER', 'parking_user'),
            password=os.environ.get('DB_PASS', 'Tenzin@2005')
        )
        cur = conn.cursor()
        
        # Get count of active sessions
        cur.execute('''
            SELECT COUNT(*) 
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            WHERE ps.parking_id = %s AND pe.departure_time IS NULL
        ''', (parking_id,))
        active_count = cur.fetchone()[0]
        
        if active_count == 0:
            print(f"✅ No active sessions to clear for parking area {parking_id}")
        else:
            print(f"🧹 Found {active_count} active parking sessions")
            print(f"⚠️  Marking all as departed to start fresh...")
            
            # Mark all active sessions as departed
            cur.execute('''
                UPDATE parking_events pe
                SET departure_time = CURRENT_TIMESTAMP,
                    parked_time = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - pe.arrival_time))/60
                FROM parking_slots ps
                WHERE pe.slot_id = ps.slot_id 
                AND ps.parking_id = %s 
                AND pe.departure_time IS NULL
            ''', (parking_id,))
            
            rows_updated = cur.rowcount
            conn.commit()
            
            print(f"✅ Cleared {rows_updated} active sessions")
            print(f"✅ All slots are now FREE")
            print(f"✅ Ready to start video processing!")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧹 RESET PARKING AREA")
    print("="*60)
    print("\nThis will mark all active parking sessions as departed")
    print("Use this before starting new video processing\n")
    
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        reset_parking_area(7)
    else:
        print("❌ Cancelled")
    
    print("="*60 + "\n")
