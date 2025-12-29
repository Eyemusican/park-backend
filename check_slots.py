"""Quick script to check and fix slots 3, 6, 7"""
import psycopg2
import os

conn = psycopg2.connect(
    host='127.0.0.1',
    port='5432',
    dbname='parking_db',
    user='parking_user',
    password='Tenzin@2005'
)
cur = conn.cursor()

print("\n=== Current State of Slots 3, 6, 7 ===")
cur.execute('''
    SELECT ps.slot_id, ps.slot_number, pe.event_id, pe.arrival_time
    FROM parking_slots ps
    LEFT JOIN parking_events pe ON pe.slot_id = ps.slot_id AND pe.departure_time IS NULL
    WHERE ps.slot_number IN (3, 6, 7)
    ORDER BY ps.slot_number
''')

slots = cur.fetchall()
for slot in slots:
    status = "OCCUPIED" if slot[2] else "FREE"
    print(f"Slot {slot[1]}: {status}")
    if slot[2]:
        print(f"  - Event ID: {slot[2]}, Arrival: {slot[3]}")

print("\n=== Clearing slots 3, 6, 7 ===")
cur.execute('''
    UPDATE parking_events pe
    SET departure_time = CURRENT_TIMESTAMP,
        parked_time = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - pe.arrival_time))/60
    FROM parking_slots ps
    WHERE pe.slot_id = ps.slot_id 
    AND ps.slot_number IN (3, 6, 7)
    AND pe.departure_time IS NULL
''')

cleared = cur.rowcount
conn.commit()

print(f"✅ Cleared {cleared} parking sessions")
print("✅ Slots 3, 6, 7 are now FREE")

cur.close()
conn.close()
