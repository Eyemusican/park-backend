"""
Setup Database for Norzinlam Parking Only
Removes dummy areas and configures only Norzinlam with 8 slots
"""
import psycopg2

def setup_norzinlam_only():
    try:
        conn = psycopg2.connect(
            host='127.0.0.1',
            port=5432,
            database='parking_db',
            user='parking_user',
            password='parking123'
        )
        cur = conn.cursor()
        
        print("=" * 70)
        print("SETTING UP NORZINLAM PARKING ONLY")
        print("=" * 70)
        
        # Delete parking events for areas 2, 3, 4
        print("\n🗑️  Removing parking events for other areas...")
        cur.execute('''
            DELETE FROM parking_events 
            WHERE slot_id IN (
                SELECT slot_id FROM parking_slots WHERE parking_id IN (2, 3, 4)
            )
        ''')
        deleted_events = cur.rowcount
        print(f"   Deleted {deleted_events} parking events")
        
        # Delete parking slots for areas 2, 3, 4
        print("\n🗑️  Removing parking slots for other areas...")
        cur.execute('DELETE FROM parking_slots WHERE parking_id IN (2, 3, 4)')
        deleted_slots = cur.rowcount
        print(f"   Deleted {deleted_slots} parking slots")
        
        # Delete parking areas 2, 3, 4
        print("\n🗑️  Removing other parking areas...")
        cur.execute('DELETE FROM parking_area WHERE parking_id IN (2, 3, 4)')
        deleted_areas = cur.rowcount
        print(f"   Deleted {deleted_areas} parking areas")
        
        # Update parking_area_id 1 to be Norzinlam with 8 slots
        print("\n📝 Updating parking area 1 to Norzinlam...")
        cur.execute('''
            UPDATE parking_area 
            SET parking_name = 'Norzinlam', slot_count = 8
            WHERE parking_id = 1
        ''')
        
        # Delete parking events for area 1 first
        print("🗑️  Clearing old parking events for area 1...")
        cur.execute('''
            DELETE FROM parking_events 
            WHERE slot_id IN (
                SELECT slot_id FROM parking_slots WHERE parking_id = 1
            )
        ''')
        deleted = cur.rowcount
        print(f"   Deleted {deleted} old parking events")
        
        # Delete existing slots for area 1
        cur.execute('DELETE FROM parking_slots WHERE parking_id = 1')
        
        # Create 8 slots for Norzinlam (parking_id = 1)
        print("📍 Creating 8 slots for Norzinlam...")
        for i in range(1, 9):
            cur.execute('''
                INSERT INTO parking_slots (parking_id, slot_number)
                VALUES (1, %s)
                ON CONFLICT DO NOTHING
            ''', (i,))
        
        conn.commit()
        
        # Verify setup
        print("\n" + "=" * 70)
        print("FINAL DATABASE STATE:")
        print("=" * 70)
        
        cur.execute('SELECT parking_id, parking_name, slot_count FROM parking_area ORDER BY parking_id')
        areas = cur.fetchall()
        print(f"\n📊 Parking Areas: {len(areas)}")
        for area in areas:
            print(f"   ID: {area[0]}, Name: {area[1]}, Slots: {area[2]}")
        
        cur.execute('SELECT parking_id, COUNT(*) FROM parking_slots GROUP BY parking_id')
        slot_counts = cur.fetchall()
        print(f"\n🅿️  Parking Slots by Area:")
        for sc in slot_counts:
            print(f"   Area {sc[0]}: {sc[1]} slots")
        
        cur.execute('''
            SELECT ps.parking_id, COUNT(*) 
            FROM parking_events pe 
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id 
            WHERE pe.departure_time IS NULL 
            GROUP BY ps.parking_id
        ''')
        active = cur.fetchall()
        print(f"\n🚗 Active Parking Events:")
        if active:
            for a in active:
                print(f"   Area {a[0]}: {a[1]} active")
        else:
            print("   None (all slots available)")
        
        print("\n" + "=" * 70)
        print("✅ SETUP COMPLETE!")
        print("=" * 70)
        print("\nYour frontend will now show:")
        print("  • Only Norzinlam parking area")
        print("  • 8 total slots")
        print("  • Real-time data from parking_evening_vedio.mp4")
        print("\n🔄 Refresh your browser to see the changes!")
        print("=" * 70)
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_norzinlam_only()
