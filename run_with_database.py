"""
SMART PARKING SYSTEM - DATABASE SETUP AND RUN SCRIPT
This script:
1. Checks PostgreSQL connection
2. Creates database tables if needed
3. Starts the database-backed server
4. Provides instructions for running video processing
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database configuration
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'parking_db')
DB_USER = os.environ.get('DB_USER', 'parking_user')
DB_PASS = os.environ.get('DB_PASS', 'Tenzin@2005')

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def test_connection():
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname='postgres',  # Connect to default database first
            user=DB_USER,
            password=DB_PASS
        )
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Cannot connect to PostgreSQL: {e}")
        return False

def database_exists():
    """Check if parking_db exists"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname='postgres',
            user=DB_USER,
            password=DB_PASS
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cur.fetchone() is not None
        
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def create_database():
    """Create parking_db database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname='postgres',
            user=DB_USER,
            password=DB_PASS
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        
        cur.close()
        conn.close()
        print(f"✅ Created database: {DB_NAME}")
        return True
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def run_schema(schema_file):
    """Run SQL schema file"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        
        with open(schema_file, 'r') as f:
            sql = f.read()
        
        cur.execute(sql)
        conn.commit()
        
        cur.close()
        conn.close()
        print(f"✅ Applied schema: {schema_file}")
        return True
    except Exception as e:
        print(f"❌ Error applying schema {schema_file}: {e}")
        return False

def check_tables():
    """Check if required tables exist"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        
        required_tables = ['parking_area', 'parking_slots', 'parking_events', 'parking_violations']
        existing_tables = []
        
        for table in required_tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            if cur.fetchone()[0]:
                existing_tables.append(table)
        
        cur.close()
        conn.close()
        
        return existing_tables, required_tables
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return [], []

def setup_database():
    """Complete database setup"""
    print_header("SMART PARKING - DATABASE SETUP")
    
    print("\n📋 Database Configuration:")
    print(f"   Host: {DB_HOST}")
    print(f"   Port: {DB_PORT}")
    print(f"   Database: {DB_NAME}")
    print(f"   User: {DB_USER}")
    
    # Step 1: Test connection
    print("\n1️⃣ Testing PostgreSQL connection...")
    if not test_connection():
        print("\n❌ SETUP FAILED: Cannot connect to PostgreSQL")
        print("\n💡 Solutions:")
        print("   1. Install PostgreSQL: https://www.postgresql.org/download/")
        print("   2. Start PostgreSQL service")
        print("   3. Create user and set password:")
        print(f"      CREATE USER {DB_USER} WITH PASSWORD '{DB_PASS}';")
        print(f"      ALTER USER {DB_USER} CREATEDB;")
        return False
    print("✅ PostgreSQL connection successful")
    
    # Step 2: Create database if needed
    print("\n2️⃣ Checking if database exists...")
    if not database_exists():
        print(f"   Creating database: {DB_NAME}")
        if not create_database():
            return False
    else:
        print(f"✅ Database '{DB_NAME}' exists")
    
    # Step 3: Check and create tables
    print("\n3️⃣ Checking database tables...")
    existing, required = check_tables()
    
    if set(existing) == set(required):
        print("✅ All required tables exist:")
        for table in existing:
            print(f"   ✓ {table}")
    else:
        print(f"   Found {len(existing)}/{len(required)} tables")
        missing = set(required) - set(existing)
        if missing:
            print(f"   Missing: {', '.join(missing)}")
        
        print("\n   Applying database schemas...")
        
        # Apply parking schema
        if not run_schema('parking_schema.sql'):
            return False
        
        # Apply violations schema
        if not run_schema('violations_schema.sql'):
            return False
    
    print_header("✅ DATABASE SETUP COMPLETE")
    return True

def main():
    """Main setup and run function"""
    print_header("SMART PARKING SYSTEM - DATABASE MODE")
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🚀 STARTING SERVER WITH DATABASE")
    print("=" * 70)
    print("\n📡 Server will run on: http://localhost:5000")
    print("📊 Using PostgreSQL for data persistence")
    print("\n⏳ Starting server (use Ctrl+C to stop)...\n")
    
    # Start the database-backed server
    os.system('python server.py')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        sys.exit(0)
