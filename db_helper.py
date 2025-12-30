import psycopg2
from datetime import datetime, timezone
import os

class ParkingDB:
    def __init__(self):
        self.host = os.environ.get('DB_HOST', '127.0.0.1')
        self.port = os.environ.get('DB_PORT', '5432')
        self.dbname = os.environ.get('DB_NAME', 'parking_db')
        self.user = os.environ.get('DB_USER', 'parking_user')
        self.password = os.environ.get('DB_PASS', 'Tenzin@2005')
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )

    def add_parking_area(self, parking_name, slot_count):
        """
        Add a new parking area OR get existing one
        ✅ FIX: Checks if parking area exists first to prevent duplicates
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # ✅ CHECK: Does this parking area already exist?
            cur.execute(
                'SELECT parking_id, slot_count FROM parking_area WHERE parking_name = %s',
                (parking_name,)
            )
            existing = cur.fetchone()
            
            if existing:
                # ✅ REUSE: Parking area already exists
                parking_id = existing[0]
                existing_slot_count = existing[1]
                
                # Update slot count if changed
                if existing_slot_count != slot_count:
                    cur.execute(
                        'UPDATE parking_area SET slot_count = %s WHERE parking_id = %s',
                        (slot_count, parking_id)
                    )
                    print(f"✅ Updated parking area: {parking_name} (ID: {parking_id})")
                    print(f"   Slot count: {existing_slot_count} → {slot_count}")
                else:
                    print(f"✅ Using existing parking area: {parking_name} (ID: {parking_id})")
                
                conn.commit()
            else:
                # ✅ CREATE NEW: Parking area doesn't exist
                cur.execute(
                    'INSERT INTO parking_area (parking_name, slot_count) VALUES (%s, %s) RETURNING parking_id',
                    (parking_name, slot_count)
                )
                parking_id = cur.fetchone()[0]
                conn.commit()
                print(f"✅ Created new parking area: {parking_name} (ID: {parking_id})")
            
            cur.close()
            conn.close()
            return parking_id
            
        except Exception as e:
            print(f"Error adding parking area: {e}")
            return None

    def add_parking_slots(self, parking_id, slot_numbers):
        """
        Add multiple parking slots for an area
        ✅ FIX: Uses ON CONFLICT to prevent duplicates
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            for slot_num in slot_numbers:
                cur.execute(
                    '''INSERT INTO parking_slots (parking_id, slot_number) 
                       VALUES (%s, %s) 
                       ON CONFLICT (parking_id, slot_number) DO NOTHING''',
                    (parking_id, slot_num)
                )

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding parking slots: {e}")
            return False

    def get_slot_id(self, parking_id, slot_number):
        """Get slot_id for a given parking area and slot number"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                'SELECT slot_id FROM parking_slots WHERE parking_id = %s AND slot_number = %s',
                (parking_id, slot_number)
            )
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"Error getting slot_id: {e}")
            return None

    def record_arrival(self, slot_id):
        """
        Record a car arrival in a slot and update is_occupied
        ✅ FIX: Updates is_occupied status in parking_slots table
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Check if there's already an active parking session
            cur.execute(
                'SELECT event_id FROM parking_events WHERE slot_id = %s AND departure_time IS NULL',
                (slot_id,)
            )
            existing = cur.fetchone()

            if existing:
                cur.close()
                conn.close()
                return existing[0]

            # Insert new parking event
            cur.execute(
                'INSERT INTO parking_events (slot_id, arrival_time) VALUES (%s, %s) RETURNING event_id',
                (slot_id, datetime.now(timezone.utc))
            )
            event_id = cur.fetchone()[0]
            
            # ✅ UPDATE: Mark slot as occupied
            cur.execute(
                'UPDATE parking_slots SET is_occupied = TRUE, last_updated = CURRENT_TIMESTAMP WHERE slot_id = %s',
                (slot_id,)
            )
            
            conn.commit()
            cur.close()
            conn.close()
            return event_id
        except Exception as e:
            print(f"Error recording arrival: {e}")
            return None

    def record_departure(self, slot_id):
        """
        Record a car departure from a slot and update is_occupied
        ✅ FIX: Updates is_occupied status in parking_slots table
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Find the active parking event
            cur.execute(
                '''SELECT event_id, arrival_time
                   FROM parking_events
                   WHERE slot_id = %s AND departure_time IS NULL
                   ORDER BY arrival_time DESC LIMIT 1''',
                (slot_id,)
            )
            result = cur.fetchone()

            if not result:
                cur.close()
                conn.close()
                return None

            event_id, arrival_time = result
            departure_time = datetime.now(timezone.utc)
            # Make naive timestamp timezone-aware (assume UTC)
            if arrival_time and arrival_time.tzinfo is None:
                arrival_time = arrival_time.replace(tzinfo=timezone.utc)

            # Calculate parked time in minutes
            parked_time = int((departure_time - arrival_time).total_seconds() / 60)

            # Update parking event with departure
            cur.execute(
                '''UPDATE parking_events
                   SET departure_time = %s, parked_time = %s
                   WHERE event_id = %s''',
                (departure_time, parked_time, event_id)
            )

            # ✅ UPDATE: Mark slot as vacant
            cur.execute(
                'UPDATE parking_slots SET is_occupied = FALSE, last_updated = CURRENT_TIMESTAMP WHERE slot_id = %s',
                (slot_id,)
            )

            conn.commit()
            cur.close()
            conn.close()
            return event_id
        except Exception as e:
            print(f"Error recording departure: {e}")
            return None

    def get_parking_stats(self, parking_id):
        """
        Get statistics for a parking area
        ✅ FIX: Uses is_occupied column for accurate real-time count
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Get total slots
            cur.execute('SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s', (parking_id,))
            total_slots = cur.fetchone()[0]

            # ✅ FIXED: Use is_occupied column for accurate count
            cur.execute(
                'SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s AND is_occupied = TRUE',
                (parking_id,)
            )
            occupied_slots = cur.fetchone()[0]

            cur.close()
            conn.close()

            return {
                'total_slots': total_slots,
                'occupied_slots': occupied_slots,
                'available_slots': total_slots - occupied_slots
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None
    
    def sync_slot_status(self, parking_id):
        """
        Sync is_occupied status based on parking_events table
        ✅ NEW: Call this to fix any inconsistencies between events and slot status
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Mark all slots as vacant first
            cur.execute(
                'UPDATE parking_slots SET is_occupied = FALSE WHERE parking_id = %s',
                (parking_id,)
            )
            
            # Mark slots with active events as occupied
            cur.execute(
                '''UPDATE parking_slots ps
                   SET is_occupied = TRUE, last_updated = CURRENT_TIMESTAMP
                   FROM parking_events pe
                   WHERE ps.slot_id = pe.slot_id
                   AND ps.parking_id = %s
                   AND pe.departure_time IS NULL''',
                (parking_id,)
            )
            
            conn.commit()
            
            # Get updated counts
            cur.execute('SELECT COUNT(*) FROM parking_slots WHERE parking_id = %s AND is_occupied = TRUE', (parking_id,))
            occupied = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            print(f"✅ Synced slot status for parking_id {parking_id}: {occupied} slots occupied")
            return True
            
        except Exception as e:
            print(f"Error syncing slot status: {e}")
            return False
    
    def cleanup_duplicate_parking_areas(self, keep_parking_id=None):
        """
        ✅ NEW: Clean up duplicate parking areas
        If keep_parking_id is specified, keeps that one and removes others with same name
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            if keep_parking_id:
                # Get the parking name
                cur.execute('SELECT parking_name FROM parking_area WHERE parking_id = %s', (keep_parking_id,))
                result = cur.fetchone()
                if not result:
                    print(f"❌ Parking ID {keep_parking_id} not found")
                    return False
                
                parking_name = result[0]
                
                # Find duplicates
                cur.execute(
                    'SELECT parking_id FROM parking_area WHERE parking_name = %s AND parking_id != %s',
                    (parking_name, keep_parking_id)
                )
                duplicates = [row[0] for row in cur.fetchall()]
                
                if not duplicates:
                    print(f"✅ No duplicates found for '{parking_name}'")
                    return True
                
                print(f"🧹 Removing {len(duplicates)} duplicate parking areas: {duplicates}")
                
                # Delete events for duplicate parking areas
                for dup_id in duplicates:
                    cur.execute(
                        'DELETE FROM parking_events WHERE slot_id IN (SELECT slot_id FROM parking_slots WHERE parking_id = %s)',
                        (dup_id,)
                    )
                    cur.execute('DELETE FROM parking_slots WHERE parking_id = %s', (dup_id,))
                    cur.execute('DELETE FROM parking_area WHERE parking_id = %s', (dup_id,))
                
                conn.commit()
                print(f"✅ Kept parking_id {keep_parking_id}, removed duplicates")
            
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error cleaning up duplicates: {e}")
            return False