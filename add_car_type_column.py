"""
Add car_type column to parking_events table
"""
import psycopg2

DB_HOST = '127.0.0.1'
DB_PORT = '5432'
DB_NAME = 'parking_db'
DB_USER = 'parking_user'
DB_PASS = 'parking123'

def add_car_type_column():
    """Add car_type column to parking_events table if it doesn't exist"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        
        # Check if column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='parking_events' AND column_name='car_type'
        """)
        
        if cur.fetchone() is None:
            print("Adding car_type column to parking_events table...")
            cur.execute("""
                ALTER TABLE parking_events 
                ADD COLUMN car_type VARCHAR(50)
            """)
            conn.commit()
            print("✅ Successfully added car_type column")
        else:
            print("✅ car_type column already exists")
        
        # Verify the column was added
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name='parking_events'
            ORDER BY ordinal_position
        """)
        
        print("\nCurrent parking_events columns:")
        for col in cur.fetchall():
            print(f"  {col[0]:30s} {col[1]}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == '__main__':
    add_car_type_column()
