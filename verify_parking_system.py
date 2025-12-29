"""
Verify Smart Parking System Database Integration
Shows exactly which parking area each video will update
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)

def check_database_structure():
    """Check current database state"""
    print_section("DATABASE STRUCTURE CHECK")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='parking_db',
            user='parking_user',
            password='Tenzin@2005'
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Check all parking areas
        print("\n1️⃣  PARKING AREAS:")
        cursor.execute("SELECT * FROM parking_area ORDER BY parking_id")
        areas = cursor.fetchall()
        
        if not areas:
            print("   ⚠️  No parking areas found!")
        else:
            for area in areas:
                print(f"\n   📍 ID {area['parking_id']}: {area['parking_name']}")
                print(f"      Capacity: {area['slot_count']} slots")
                
                # Count actual slots
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM parking_slots 
                    WHERE parking_id = %s
                """, (area['parking_id'],))
                slot_count = cursor.fetchone()['count']
                
                # Count occupied slots
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM parking_events pe
                    JOIN parking_slots ps ON pe.slot_id = ps.slot_id
                    WHERE ps.parking_id = %s AND pe.departure_time IS NULL
                """, (area['parking_id'],))
                occupied = cursor.fetchone()['count']
                
                print(f"      Configured Slots: {slot_count}")
                print(f"      Currently Occupied: {occupied}")
                print(f"      Available: {slot_count - occupied}")
        
        # 2. Check parking_events table structure
        print("\n\n2️⃣  PARKING_EVENTS TABLE STRUCTURE:")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'parking_events'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"   • {col['column_name']:<20} {col['data_type']:<15} {nullable}")
        
        # 3. Check if parking_id 7 has slots
        print("\n\n3️⃣  NORZIN LAM PARKING (ID: 7) SLOTS:")
        cursor.execute("""
            SELECT slot_id, slot_number, is_occupied, last_updated
            FROM parking_slots 
            WHERE parking_id = 7
            ORDER BY slot_number
        """)
        
        slots = cursor.fetchall()
        if not slots:
            print("   ⚠️  No slots configured for parking_id 7")
            print("   ℹ️  Slots will be created when you run the system")
        else:
            print(f"   ✅ {len(slots)} slots configured:\n")
            for slot in slots:
                status = "🚗 OCCUPIED" if slot['is_occupied'] else "✅ AVAILABLE"
                updated = slot['last_updated'].strftime("%H:%M:%S") if slot['last_updated'] else "Never"
                print(f"      Slot #{slot['slot_number']:2d} (DB ID: {slot['slot_id']:3d}): {status:15s} | Updated: {updated}")
        
        # 4. Check recent events
        print("\n\n4️⃣  RECENT PARKING EVENTS (All Areas):")
        cursor.execute("""
            SELECT 
                pe.event_id,
                pa.parking_name,
                ps.slot_number,
                pe.arrival_time,
                pe.departure_time,
                pe.parked_time
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            ORDER BY pe.arrival_time DESC
            LIMIT 10
        """)
        
        events = cursor.fetchall()
        if not events:
            print("   ℹ️  No events recorded yet")
        else:
            for event in events:
                arrival = event['arrival_time'].strftime("%Y-%m-%d %H:%M:%S")
                departure = event['departure_time'].strftime("%H:%M:%S") if event['departure_time'] else "Still parked"
                duration = f"{event['parked_time']} min" if event['parked_time'] else "---"
                
                print(f"\n      Event #{event['event_id']}: {event['parking_name']}")
                print(f"         Slot #{event['slot_number']}")
                print(f"         Arrival:   {arrival}")
                print(f"         Departure: {departure}")
                print(f"         Duration:  {duration}")
        
        conn.close()
        
        print_section("VERIFICATION COMPLETE")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def check_slot_config():
    """Check if camera_config.json exists and has correct structure"""
    print_section("SLOT CONFIGURATION CHECK")
    
    config_files = ['camera_config.json', 'configs/parking_slots.json']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n✅ Found: {config_file}")
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            print(f"   Total Slots: {data.get('total_slots', 0)}")
            print(f"   Slots defined: {len(data.get('slots', []))}")
            
            if 'slots' in data:
                print("\n   Slot IDs:")
                for slot in data['slots']:
                    print(f"      • Slot #{slot['id']}: {len(slot['points'])} points")
            
            return config_file
    
    print("\n❌ No slot configuration file found!")
    print("   Expected: camera_config.json OR configs/parking_slots.json")
    return None

def show_how_it_works():
    """Explain how the system isolates parking areas"""
    print_section("HOW VIDEO ↔ DATABASE ISOLATION WORKS")
    
    print("""
    📹 VIDEO 1: Norzin Lam Parking
       ↓
       1. System creates/gets parking_id = 7 for "Norzin Lam"
       2. Creates 8 slots with parking_id = 7
          - slot_id: 101, parking_id: 7, slot_number: 1
          - slot_id: 102, parking_id: 7, slot_number: 2
          - ... (8 slots total)
       3. All events recorded with slot_id (101, 102, etc.)
       4. These slot_ids ONLY belong to parking_id 7
       ↓
       ✅ Updates ONLY affect "Norzin Lam" parking area
    
    
    📹 VIDEO 2: Changlimithang Parking (if you add it)
       ↓
       1. System creates/gets parking_id = 8 for "Changlimithang"
       2. Creates 10 slots with parking_id = 8
          - slot_id: 201, parking_id: 8, slot_number: 1
          - slot_id: 202, parking_id: 8, slot_number: 2
          - ... (10 slots total)
       3. All events recorded with slot_id (201, 202, etc.)
       4. These slot_ids ONLY belong to parking_id 8
       ↓
       ✅ Updates ONLY affect "Changlimithang" parking area
    
    
    🔒 ISOLATION GUARANTEE:
       • Each parking area has unique parking_id
       • Each slot belongs to ONE parking_id
       • Events are linked through slot_id → parking_id
       • Video 1 CANNOT update Video 2's data (different slot_ids)
    """)

def test_isolation():
    """Test that parking areas are properly isolated"""
    print_section("ISOLATION TEST")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='parking_db',
            user='parking_user',
            password='Tenzin@2005'
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check for slot_id overlap between parking areas
        cursor.execute("""
            SELECT 
                ps1.parking_id as parking1,
                ps2.parking_id as parking2,
                ps1.slot_id
            FROM parking_slots ps1
            JOIN parking_slots ps2 ON ps1.slot_id = ps2.slot_id
            WHERE ps1.parking_id != ps2.parking_id
        """)
        
        overlaps = cursor.fetchall()
        
        if overlaps:
            print("❌ WARNING: Found slot_id overlap between parking areas!")
            for overlap in overlaps:
                print(f"   Slot ID {overlap['slot_id']} used by both parking {overlap['parking1']} and {overlap['parking2']}")
        else:
            print("✅ PERFECT: No slot_id overlap between parking areas")
            print("   Each parking area has unique slot IDs")
            print("   Videos will ONLY update their own parking area")
        
        # Show slot_id ranges per parking area
        cursor.execute("""
            SELECT 
                pa.parking_name,
                pa.parking_id,
                MIN(ps.slot_id) as min_slot_id,
                MAX(ps.slot_id) as max_slot_id,
                COUNT(ps.slot_id) as slot_count
            FROM parking_area pa
            LEFT JOIN parking_slots ps ON pa.parking_id = ps.parking_id
            GROUP BY pa.parking_id, pa.parking_name
            ORDER BY pa.parking_id
        """)
        
        ranges = cursor.fetchall()
        
        print("\n📊 Slot ID Ranges per Parking Area:")
        for area in ranges:
            if area['slot_count'] > 0:
                print(f"\n   {area['parking_name']} (ID: {area['parking_id']})")
                print(f"      Slot ID Range: {area['min_slot_id']} - {area['max_slot_id']}")
                print(f"      Total Slots: {area['slot_count']}")
            else:
                print(f"\n   {area['parking_name']} (ID: {area['parking_id']})")
                print(f"      No slots configured yet")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║       SMART PARKING SYSTEM - DATABASE VERIFICATION TOOL            ║
    ║                                                                    ║
    ║  This tool verifies that your video feeds are properly isolated    ║
    ║  and will only update their corresponding parking areas            ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all checks
    check_database_structure()
    config_file = check_slot_config()
    show_how_it_works()
    test_isolation()
    
    # Final recommendation
    print_section("RECOMMENDATIONS")
    
    print("""
    ✅ TO RUN YOUR SYSTEM:
    
    1. Make sure your slot config exists:
       • camera_config.json (in root folder)
       OR
       • configs/parking_slots.json
    
    2. Run the system:
       python smart_parking_db.py --run parking_evening_vedio.mp4
    
    3. The system will:
       ✓ Connect to parking_id 7 ("Norzin Lam Parking")
       ✓ Create/update ONLY slots for parking_id 7
       ✓ Record events ONLY for parking_id 7
       ✓ NOT affect any other parking areas
    
    
    🎯 TO ADD MORE PARKING AREAS:
    
    1. Create new slot config: changlimithang_slots.json
    2. Change parking name in code:
       system = SmartParkingMVP(
           'changlimithang_slots.json',
           parking_area_name="Changlimithang Parking"  # ← Change this
       )
    3. Run with different video:
       python smart_parking_db.py --run changlimithang_video.mp4
    
    This will create a NEW parking_id (e.g., 8) and keep data separate!
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()