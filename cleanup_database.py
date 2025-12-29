"""
Database Cleanup Script
Fixes duplicate parking areas and syncs slot status
"""

import psycopg2
from psycopg2.extras import RealDictCursor

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)

def show_current_state():
    """Show current parking areas"""
    print_section("CURRENT DATABASE STATE")
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='parking_db',
        user='parking_user',
        password='Tenzin@2005'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Show all parking areas
    cursor.execute("""
        SELECT 
            pa.parking_id,
            pa.parking_name,
            pa.slot_count,
            COUNT(DISTINCT ps.slot_id) as configured_slots,
            COUNT(DISTINCT pe.event_id) as total_events,
            MAX(pe.arrival_time) as last_activity
        FROM parking_area pa
        LEFT JOIN parking_slots ps ON pa.parking_id = ps.parking_id
        LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id
        GROUP BY pa.parking_id, pa.parking_name, pa.slot_count
        ORDER BY pa.parking_id
    """)
    
    areas = cursor.fetchall()
    
    print("\nParking Areas:")
    for area in areas:
        print(f"\n  📍 ID {area['parking_id']}: {area['parking_name']}")
        print(f"     Capacity: {area['slot_count']} slots")
        print(f"     Configured: {area['configured_slots']} slots")
        print(f"     Total Events: {area['total_events']}")
        last = area['last_activity'].strftime("%Y-%m-%d %H:%M") if area['last_activity'] else "None"
        print(f"     Last Activity: {last}")
    
    conn.close()
    return areas

def cleanup_duplicates(keep_id):
    """Remove duplicate parking areas, keep only specified ID"""
    print_section(f"CLEANING UP - KEEPING PARKING_ID {keep_id}")
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='parking_db',
        user='parking_user',
        password='Tenzin@2005'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get the parking name to keep
    cursor.execute('SELECT parking_name FROM parking_area WHERE parking_id = %s', (keep_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"❌ Parking ID {keep_id} not found!")
        conn.close()
        return False
    
    parking_name = result['parking_name']
    print(f"\nKeeping: {parking_name} (ID: {keep_id})")
    
    # Find duplicates
    cursor.execute(
        'SELECT parking_id FROM parking_area WHERE parking_name = %s AND parking_id != %s',
        (parking_name, keep_id)
    )
    duplicates = [row['parking_id'] for row in cursor.fetchall()]
    
    if not duplicates:
        print("✅ No duplicates found!")
        conn.close()
        return True
    
    print(f"\n🗑️  Removing duplicate parking areas: {duplicates}")
    
    # Delete for each duplicate
    for dup_id in duplicates:
        print(f"\n  Removing parking_id {dup_id}...")
        
        # Count events to delete
        cursor.execute(
            'SELECT COUNT(*) as count FROM parking_events WHERE slot_id IN (SELECT slot_id FROM parking_slots WHERE parking_id = %s)',
            (dup_id,)
        )
        event_count = cursor.fetchone()['count']
        
        # Delete events
        cursor.execute(
            'DELETE FROM parking_events WHERE slot_id IN (SELECT slot_id FROM parking_slots WHERE parking_id = %s)',
            (dup_id,)
        )
        print(f"    ✓ Deleted {event_count} events")
        
        # Delete slots
        cursor.execute('DELETE FROM parking_slots WHERE parking_id = %s', (dup_id,))
        print(f"    ✓ Deleted slots")
        
        # Delete parking area
        cursor.execute('DELETE FROM parking_area WHERE parking_id = %s', (dup_id,))
        print(f"    ✓ Deleted parking area")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Cleanup complete! Only parking_id {keep_id} remains.")
    return True

def sync_slot_status(parking_id):
    """Sync is_occupied based on parking_events"""
    print_section(f"SYNCING SLOT STATUS FOR PARKING_ID {parking_id}")
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='parking_db',
        user='parking_user',
        password='Tenzin@2005'
    )
    cursor = conn.cursor()
    
    # Mark all slots as vacant first
    cursor.execute(
        'UPDATE parking_slots SET is_occupied = FALSE WHERE parking_id = %s',
        (parking_id,)
    )
    
    # Mark slots with active events as occupied
    cursor.execute(
        '''UPDATE parking_slots ps
           SET is_occupied = TRUE, last_updated = CURRENT_TIMESTAMP
           FROM parking_events pe
           WHERE ps.slot_id = pe.slot_id
           AND ps.parking_id = %s
           AND pe.departure_time IS NULL''',
        (parking_id,)
    )
    
    conn.commit()
    
    # Get counts
    cursor.execute(
        'SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s AND is_occupied = TRUE',
        (parking_id,)
    )
    occupied = cursor.fetchone()[0]
    
    cursor.execute(
        'SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s',
        (parking_id,)
    )
    total = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n✅ Slot status synced!")
    print(f"   Total Slots: {total}")
    print(f"   Occupied: {occupied}")
    print(f"   Available: {total - occupied}")
    
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              DATABASE CLEANUP & FIX SCRIPT                       ║
║                                                                  ║
║  This script will:                                               ║
║  1. Show current parking areas                                   ║
║  2. Remove duplicate "Norzin Lam Parking" entries                ║
║  3. Keep only the one you choose                                 ║
║  4. Sync slot occupancy status                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Show current state
    areas = show_current_state()
    
    if len(areas) <= 1:
        print("\n✅ No duplicates found! Database is clean.")
        
        # Just sync status
        if areas:
            sync_slot_status(areas[0]['parking_id'])
        
        return
    
    # Step 2: Ask which to keep
    print_section("CHOOSE WHICH PARKING AREA TO KEEP")
    
    print("\nWhich parking_id should we keep?")
    print("(Recommendation: Keep the one with most recent activity)")
    print()
    
    # Find the one with most events
    most_events = max(areas, key=lambda x: x['total_events'] or 0)
    most_recent = max(areas, key=lambda x: x['last_activity'] or '1970-01-01')
    
    print(f"💡 parking_id {most_recent['parking_id']} has most recent activity")
    print(f"💡 parking_id {most_events['parking_id']} has most events ({most_events['total_events']})")
    print()
    
    # Auto-select the most recent one
    keep_id = most_recent['parking_id']
    print(f"✅ Auto-selected parking_id {keep_id} (most recent activity)")
    print()
    
    user_input = input(f"Press ENTER to keep parking_id {keep_id}, or type a different ID: ").strip()
    
    if user_input:
        keep_id = int(user_input)
    
    # Step 3: Clean up
    success = cleanup_duplicates(keep_id)
    
    if success:
        # Step 4: Sync status
        sync_slot_status(keep_id)
        
        # Step 5: Show final state
        print_section("FINAL DATABASE STATE")
        show_current_state()
        
        print(f"\n{'='*70}")
        print("✅ DATABASE CLEANUP COMPLETE!")
        print(f"{'='*70}")
        print(f"\n🎯 Your system will now use parking_id {keep_id}")
        print(f"   All future runs will update this same parking area")
        print(f"   No more duplicates will be created!")
        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()