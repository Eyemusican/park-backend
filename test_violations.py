"""
Test Violation System - Create sample violations for testing
"""
import psycopg2
import os
from datetime import datetime, timedelta
import random

# Database connection settings
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')


def create_sample_violations():
    """Create sample violations for testing"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        
        print("=" * 60)
        print("Creating Sample Violations for Testing")
        print("=" * 60)
        
        # Check if parking tables exist, if not use simpler approach
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'parking_slots'
            )
        """)
        
        has_parking_slots = cur.fetchone()[0]
        
        violations_created = 0
        
        if has_parking_slots:
            # Get existing slots
            cur.execute("""
                SELECT ps.slot_id, ps.slot_number, pa.parking_name, pa.parking_id
                FROM parking_slots ps
                JOIN parking_area pa ON ps.parking_id = pa.parking_id
                ORDER BY RANDOM()
                LIMIT 5
            """)
            
            slots = cur.fetchall()
        else:
            # Create fake slots for demonstration
            print("Note: Using simulated data (parking_slots table not found)")
            slots = [
                (1, 15, "Main Entrance - North", 1),
                (8, 22, "Building B - West Wing", 2),
                (5, 5, "VIP Parking Zone", 3),
            ]
        
        if not slots:
            print("Note: Creating simulated violations without real slots")
            slots = [(1, 15, "Main Entrance", 1)]
        
        # Create different types of violations
        violation_types = [
            ("Expired Duration", "Medium", "Vehicle parked for 4+ hours"),
            ("Expired Duration", "High", "Vehicle parked for 8+ hours"),
            ("Double Parking", "High", "Multiple vehicles detected in single slot"),
            ("Outside Boundary", "Medium", "Vehicle parked over line markers"),
            ("No Valid Permit", "High", "Unauthorized vehicle in restricted zone"),
        ]
        
        for i, (slot_id, slot_number, parking_name, parking_id) in enumerate(slots[:len(violation_types)]):
            violation_type, severity, description = violation_types[i]
            
            # Generate violation ID
            violation_id = f"V-{int(datetime.now().timestamp() * 1000)}-{i}"
            
            # Random time in the past few minutes
            minutes_ago = random.randint(2, 30)
            detected_at = datetime.now() - timedelta(minutes=minutes_ago)
            
            # Duration violations should have duration_minutes
            duration_minutes = None
            if violation_type == "Expired Duration":
                if severity == "High":
                    duration_minutes = random.randint(480, 720)  # 8-12 hours
                else:
                    duration_minutes = random.randint(240, 420)  # 4-7 hours
            
            # Create violation
            cur.execute("""
                INSERT INTO parking_violations 
                (violation_id, slot_id, vehicle_id, license_plate, violation_type, 
                 severity, description, duration_minutes, detected_at, status, parking_area)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                violation_id,
                slot_id,
                f"VEH-{1000 + i}",
                f"BP-2024-{100 + i}",
                violation_type,
                severity,
                description,
                duration_minutes,
                detected_at,
                'ACTIVE',
                parking_name
            ))
            
            violations_created += 1
            print(f"Success! Created {severity} violation: {violation_type} at {parking_name} (Slot {slot_number})")
        
        conn.commit()
        
        print("-" * 60)
        print(f"Success! Created {violations_created} sample violations!")
        
        # Show summary
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN severity = 'High' THEN 1 END) as high,
                COUNT(CASE WHEN severity = 'Medium' THEN 1 END) as medium,
                COUNT(CASE WHEN severity = 'Low' THEN 1 END) as low
            FROM parking_violations
            WHERE status = 'ACTIVE'
        """)
        
        summary = cur.fetchone()
        print("\nViolation Summary:")
        print(f"  Total Active: {summary[0]}")
        print(f"  High Severity: {summary[1]}")
        print(f"  Medium Severity: {summary[2]}")
        print(f"  Low Severity: {summary[3]}")
        
        print("\nTest the frontend at: http://localhost:3000/admin")
        print("=" * 60)
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating sample violations: {e}")
        import traceback
        traceback.print_exc()


def clear_all_violations():
    """Clear all violations (for testing)"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        
        cur.execute("DELETE FROM parking_violations")
        count = cur.rowcount
        conn.commit()
        
        print(f"Success! Cleared {count} violations")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error clearing violations: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_all_violations()
    else:
        create_sample_violations()
