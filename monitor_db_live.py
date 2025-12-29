"""
Real-time Database Monitor - Watch parking events as they happen
"""
import psycopg2
import os
import time
from datetime import datetime

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=os.environ.get('DB_PORT', '5432'),
        dbname=os.environ.get('DB_NAME', 'parking_db'),
        user=os.environ.get('DB_USER', 'parking_user'),
        password=os.environ.get('DB_PASS', 'Tenzin@2005')
    )

def get_latest_events(limit=10):
    """Get latest parking events"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT pe.event_id, ps.slot_number, pa.parking_name,
                   pe.arrival_time, pe.departure_time, pe.parked_time
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            ORDER BY pe.event_id DESC
            LIMIT %s
        ''', (limit,))
        
        events = cur.fetchall()
        cur.close()
        conn.close()
        return events
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_current_stats():
    """Get current occupancy stats"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Active sessions
        cur.execute('SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL')
        active = cur.fetchone()[0]
        
        # Total slots
        cur.execute('SELECT COUNT(*) FROM parking_slots')
        total = cur.fetchone()[0]
        
        # Total events
        cur.execute('SELECT COUNT(*) FROM parking_events')
        total_events = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            'active': active,
            'total': total,
            'total_events': total_events,
            'available': total - active
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def monitor_live(refresh_interval=2):
    """Monitor database in real-time"""
    print("\n" + "="*80)
    print("🚗 LIVE DATABASE MONITOR - Press Ctrl+C to stop")
    print("="*80)
    
    last_event_id = 0
    
    try:
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "="*80)
            print(f"🚗 LIVE DATABASE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Get stats
            stats = get_current_stats()
            if stats:
                print(f"\n📊 CURRENT STATUS:")
                print(f"   Total Slots: {stats['total']}")
                print(f"   🚗 Occupied: {stats['active']}")
                print(f"   ✅ Available: {stats['available']}")
                print(f"   📈 Total Events: {stats['total_events']}")
                
                if stats['total'] > 0:
                    occupancy = (stats['active'] / stats['total']) * 100
                    print(f"   📍 Occupancy: {occupancy:.1f}%")
            
            # Get latest events
            events = get_latest_events(15)
            
            print(f"\n📋 LATEST EVENTS (Last 15):")
            print("-"*80)
            print(f"{'Event':<8} {'Slot':<6} {'Area':<20} {'Status':<12} {'Time':<20}")
            print("-"*80)
            
            for event in events:
                event_id, slot_num, area_name, arrival, departure, duration = event
                
                # Highlight new events
                is_new = event_id > last_event_id
                prefix = "🆕 " if is_new else "   "
                
                if departure:
                    status = f"✅ Departed"
                    time_str = departure.strftime("%H:%M:%S")
                else:
                    status = "🚗 ACTIVE"
                    time_str = arrival.strftime("%H:%M:%S")
                
                print(f"{prefix}{event_id:<8} {slot_num:<6} {area_name:<20} {status:<12} {time_str:<20}")
            
            if events:
                last_event_id = max(last_event_id, events[0][0])
            
            print("-"*80)
            print(f"\n⏱️  Refreshing every {refresh_interval} seconds... (Press Ctrl+C to stop)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
        print("="*80 + "\n")

if __name__ == "__main__":
    monitor_live(refresh_interval=2)
