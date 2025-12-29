"""Add vehicle detail columns to parking_events table"""
import psycopg2

conn = psycopg2.connect(
    host='127.0.0.1',
    port='5432',
    dbname='parking_db',
    user='parking_user',
    password='Tenzin@2005'
)
cur = conn.cursor()

print("\n=== Adding Vehicle Detail Columns to parking_events ===")

# Add columns if they don't exist
columns_to_add = [
    ('vehicle_id', 'VARCHAR(50)'),
    ('license_plate', 'VARCHAR(50)'),
    ('vehicle_type', 'VARCHAR(50)'),
    ('color', 'VARCHAR(50)')
]

for column_name, data_type in columns_to_add:
    try:
        cur.execute(f"""
            ALTER TABLE parking_events 
            ADD COLUMN IF NOT EXISTS {column_name} {data_type}
        """)
        print(f"✅ Added column: {column_name}")
    except Exception as e:
        print(f"❌ Error adding {column_name}: {e}")

conn.commit()

print("\n=== Updated Table Structure ===")
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'parking_events'
    ORDER BY ordinal_position
""")
columns = cur.fetchall()
for col in columns:
    print(f"  - {col[0]}: {col[1]}")

cur.close()
conn.close()

print("\n✅ Database schema updated successfully!")
