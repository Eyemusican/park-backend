import psycopg2

try:
    conn = psycopg2.connect(
        host='127.0.0.1',
        port=5432,
        dbname='parking_db',
        user='parking_user',
        password='Tenzin@2005'
    )
    print("✅ Database connection successful!")
    cursor = conn.cursor()
    cursor.execute("SELECT current_user, current_database();")
    result = cursor.fetchone()
    print(f"Connected as: {result[0]} to database: {result[1]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
