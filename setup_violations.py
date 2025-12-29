"""
Setup violations table in the database
Run this script to add violations support to your parking system
"""
import psycopg2
import os

# Database connection settings
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')

def setup_violations_table():
    """Create violations table and related indexes"""
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
        print("Setting up Violations Table")
        print("=" * 60)
        
        # Drop any conflicting views first
        try:
            cur.execute("DROP VIEW IF EXISTS violations_by_area CASCADE")
            conn.commit()
            print("Dropped existing views")
        except:
            pass
        
        # Read and execute schema
        with open('violations_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        cur.execute(schema_sql)
        conn.commit()
        
        print("Success! Violations table created successfully!")
        print("Success! Indexes created successfully!")
        
        # Verify table exists
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'parking_violations'
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        print("\nTable Structure:")
        print("-" * 60)
        for col in columns:
            print(f"  {col[0]:<25} {col[1]}")
        print("-" * 60)
        
        cur.close()
        conn.close()
        
        print("\nViolations system is ready to use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error setting up violations table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_violations_table()
