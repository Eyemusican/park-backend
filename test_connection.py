import psycopg2

# Try without password first
try:
    conn = psycopg2.connect(
        host='127.0.0.1',
        port='5432',
        dbname='parking_db',
        user='parking_user'
        # NO PASSWORD
    )
    print("✅ SUCCESS without password!")
    conn.close()
except Exception as e:
    print(f"❌ Failed without password: {e}")

# Try with connection string
try:
    conn = psycopg2.connect("postgresql://parking_user@127.0.0.1:5432/parking_db")
    print("✅ SUCCESS with connection string!")
    conn.close()
except Exception as e:
    print(f"❌ Failed with connection string: {e}")