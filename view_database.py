"""
View Database Contents - Display all data in parking database
"""
import psycopg2
import os
from datetime import datetime

def connect_db():
    """Connect to database"""
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=os.environ.get('DB_PORT', '5432'),
        dbname=os.environ.get('DB_NAME', 'parking_db'),
        user=os.environ.get('DB_USER', 'parking_user'),
        password=os.environ.get('DB_PASS', 'Tenzin@2005')
    )

def view_parking_areas():
    """View all parking areas"""
    try:
        conn = connect_db()
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM parking_area ORDER BY parking_id')
        areas = cur.fetchall()
        
        print("\n" + "="*70)
        print("📍 PARKING AREAS")
        print("="*70)
        
        if not areas:
            print("❌ No parking areas found")
        else:
            print(f"{'ID':<5} {'Name':<30} {'Slot Count':<10}")
            print("-"*70)
            for area in areas:
                print(f"{area[0]:<5} {area[1]:<30} {area[2]:<10}")
            print(f"\n✅ Total: {len(areas)} parking area(s)")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing parking areas: {e}")

def view_parking_slots():
    """View all parking slots"""
    try:
        conn = connect_db()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT ps.slot_id, ps.parking_id, pa.parking_name, ps.slot_number
            FROM parking_slots ps
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            ORDER BY ps.parking_id, ps.slot_number
        ''')
        slots = cur.fetchall()
        
        print("\n" + "="*70)
        print("🅿️  PARKING SLOTS")
        print("="*70)
        
        if not slots:
            print("❌ No parking slots found")
        else:
            print(f"{'Slot ID':<10} {'Area ID':<10} {'Area Name':<25} {'Slot #':<10}")
            print("-"*70)
            
            current_area = None
            for slot in slots:
                if current_area != slot[1]:
                    if current_area is not None:
                        print("-"*70)
                    current_area = slot[1]
                
                print(f"{slot[0]:<10} {slot[1]:<10} {slot[2]:<25} {slot[3]:<10}")
            
            print(f"\n✅ Total: {len(slots)} parking slot(s)")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing parking slots: {e}")

def view_parking_events():
    """View all parking events"""
    try:
        conn = connect_db()
        cur = conn.cursor()
        
        cur.execute('''
            SELECT pe.event_id, pe.slot_id, ps.slot_number, pa.parking_name,
                   pe.arrival_time, pe.departure_time, pe.parked_time
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            ORDER BY pe.arrival_time DESC
            LIMIT 50
        ''')
        events = cur.fetchall()
        
        print("\n" + "="*100)
        print("📊 PARKING EVENTS (Last 50)")
        print("="*100)
        
        if not events:
            print("❌ No parking events found")
        else:
            print(f"{'Event':<7} {'Slot':<7} {'Slot#':<7} {'Area':<15} {'Arrival':<20} {'Departure':<20} {'Duration':<10}")
            print("-"*100)
            
            active_count = 0
            for event in events:
                event_id, slot_id, slot_num, area_name, arrival, departure, duration = event
                
                arrival_str = arrival.strftime("%Y-%m-%d %H:%M:%S") if arrival else "N/A"
                
                if departure:
                    departure_str = departure.strftime("%Y-%m-%d %H:%M:%S")
                    duration_str = f"{duration} min" if duration else "N/A"
                else:
                    departure_str = "🚗 ACTIVE"
                    duration_str = "---"
                    active_count += 1
                
                print(f"{event_id:<7} {slot_id:<7} {slot_num:<7} {area_name:<15} {arrival_str:<20} {departure_str:<20} {duration_str:<10}")
            
            print(f"\n✅ Total events: {len(events)}")
            print(f"🚗 Active parking sessions: {active_count}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing parking events: {e}")

def view_statistics():
    """View overall parking statistics"""
    try:
        conn = connect_db()
        cur = conn.cursor()
        
        print("\n" + "="*70)
        print("📈 PARKING STATISTICS")
        print("="*70)
        
        # Total parking areas
        cur.execute('SELECT COUNT(*) FROM parking_area')
        total_areas = cur.fetchone()[0]
        
        # Total slots
        cur.execute('SELECT COUNT(*) FROM parking_slots')
        total_slots = cur.fetchone()[0]
        
        # Active parking sessions
        cur.execute('SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL')
        active_sessions = cur.fetchone()[0]
        
        # Total events
        cur.execute('SELECT COUNT(*) FROM parking_events')
        total_events = cur.fetchone()[0]
        
        # Average parking duration
        cur.execute('SELECT AVG(parked_time) FROM parking_events WHERE parked_time IS NOT NULL')
        avg_duration = cur.fetchone()[0]
        avg_duration_str = f"{avg_duration:.1f} minutes" if avg_duration else "N/A"
        
        print(f"🏢 Total Parking Areas:     {total_areas}")
        print(f"🅿️  Total Parking Slots:     {total_slots}")
        print(f"🚗 Active Parking Sessions: {active_sessions}")
        print(f"📊 Total Parking Events:    {total_events}")
        print(f"⏱️  Average Duration:        {avg_duration_str}")
        
        if total_slots > 0:
            occupancy_rate = (active_sessions / total_slots) * 100
            print(f"📍 Occupancy Rate:          {occupancy_rate:.1f}%")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing statistics: {e}")

def main():
    """Main function to display all database contents"""
    print("\n" + "🚗"*35)
    print("   SMART PARKING SYSTEM - DATABASE VIEWER")
    print("🚗"*35)
    
    try:
        # Test connection first
        conn = connect_db()
        print("\n✅ Database connection successful!")
        conn.close()
        
        # Display all data
        view_statistics()
        view_parking_areas()
        view_parking_slots()
        view_parking_events()
        
        print("\n" + "="*70)
        print("✅ Database inspection complete!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ DATABASE CONNECTION FAILED: {e}")
        print("\n💡 Troubleshooting steps:")
        print("1. Check if PostgreSQL is running: Get-Service postgresql*")
        print("2. Verify database credentials in db_helper.py")
        print("3. Make sure database 'parking_db' exists")
        print("4. Check if user 'parking_user' has access\n")

if __name__ == "__main__":
    main()
