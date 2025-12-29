"""Check database tables for vehicle details"""
import psycopg2

conn = psycopg2.connect(
    host='127.0.0.1',
    port='5432',
    dbname='parking_db',
    user='parking_user',
    password='Tenzin@2005'
)
cur = conn.cursor()

print("\n=== Database Tables ===")
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
tables = cur.fetchall()
for table in tables:
    print(f"  - {table[0]}")

print("\n=== Parking Events Columns ===")
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'parking_events'
    ORDER BY ordinal_position
""")
columns = cur.fetchall()
for col in columns:
    print(f"  - {col[0]}: {col[1]}")

print("\n=== Sample Active Parking Data ===")
cur.execute("""
    SELECT 
        pe.event_id,
        ps.slot_number,
        pa.parking_name,
        pe.arrival_time,
        pe.vehicle_id,
        pe.license_plate,
        pe.vehicle_type,
        pe.color
    FROM parking_events pe
    JOIN parking_slots ps ON pe.slot_id = ps.slot_id
    JOIN parking_area pa ON ps.parking_id = pa.parking_id
    WHERE pe.departure_time IS NULL
    LIMIT 5
""")
data = cur.fetchall()
for row in data:
    print(f"  Event {row[0]}: Slot {row[1]}, LP: {row[5]}, Type: {row[6]}, Color: {row[7]}")

cur.close()
conn.close()
