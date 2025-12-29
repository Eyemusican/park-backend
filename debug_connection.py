import psycopg2
import os

# Test different connection methods
print("=" * 60)
print("PostgreSQL Connection Debugging")
print("=" * 60)

# Configuration from your server1.py
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')

print(f"\n📋 Configuration:")
print(f"   Host: {DB_HOST}")
print(f"   Port: {DB_PORT}")
print(f"   Database: {DB_NAME}")
print(f"   User: {DB_USER}")
print(f"   Password: {DB_PASS}")
print(f"   Password Length: {len(DB_PASS)}")
print(f"   Password (repr): {repr(DB_PASS)}")

# Test 1: Basic connection
print(f"\n🔄 Test 1: Basic connection")
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    print("✅ Connection successful!")
    cur = conn.cursor()
    cur.execute('SELECT version();')
    version = cur.fetchone()
    print(f"   PostgreSQL version: {version[0]}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")

# Test 2: Connection string format
print(f"\n🔄 Test 2: Connection string format")
try:
    conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASS}"
    print(f"   Connection string: {conn_string}")
    conn = psycopg2.connect(conn_string)
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")

# Test 3: With connection timeout
print(f"\n🔄 Test 3: With timeout")
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=3
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")

print("\n" + "=" * 60)