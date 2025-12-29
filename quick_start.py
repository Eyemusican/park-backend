#!/usr/bin/env python3
"""
Quick start script for the complete Smart Parking System with Database
This script helps you start all components in the correct order
"""
import subprocess
import sys
import time
import os
import platform

def print_header(text):
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def print_step(number, text):
    print(f"\n{number}️⃣ {text}")

def check_postgres():
    """Check if PostgreSQL is installed and running"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='127.0.0.1',
            port='5432',
            dbname='postgres',
            user='parking_user',
            password='Tenzin@2005',
            connect_timeout=3
        )
        conn.close()
        return True
    except:
        return False

def main():
    print_header("SMART PARKING SYSTEM - QUICK START")
    
    print("\n📋 This script will help you start:")
    print("   1. Database Server (Backend API)")
    print("   2. Frontend (Next.js)")
    print("   3. Video Processing (Optional)")
    
    # Check PostgreSQL
    print_step(1, "Checking PostgreSQL connection...")
    if check_postgres():
        print("   ✅ PostgreSQL is ready")
        use_database = True
    else:
        print("   ⚠️  PostgreSQL not available")
        print("\n   Choose server mode:")
        print("   1. Database mode (requires PostgreSQL)")
        print("   2. In-memory mode (quick start, no PostgreSQL)")
        
        choice = input("\n   Enter choice (1 or 2): ").strip()
        use_database = choice == "1"
    
    # Determine which server to use
    if use_database:
        print_step(2, "Starting DATABASE server...")
        server_script = "run_with_database.py"
    else:
        print_step(2, "Starting IN-MEMORY server...")
        server_script = "simple_server.py"
    
    print(f"\n   📡 Backend: http://localhost:5000")
    print("   💾 Storage:", "PostgreSQL Database" if use_database else "In-Memory (temporary)")
    print("\n   ⏳ Starting server in 3 seconds...")
    print("   (You'll need to open new terminals for frontend and video)")
    time.sleep(3)
    
    # Instructions for manual start
    print("\n" + "=" * 70)
    print("📖 MANUAL START INSTRUCTIONS")
    print("=" * 70)
    
    print("\n🖥️  TERMINAL 1 - Backend Server:")
    if platform.system() == "Windows":
        print(f"   cd {os.getcwd()}")
        print(f"   python {server_script}")
    else:
        print(f"   cd {os.getcwd()}")
        print(f"   python3 {server_script}")
    
    print("\n🌐 TERMINAL 2 - Frontend:")
    frontend_path = os.path.join(os.path.dirname(os.getcwd()), 'Frontend')
    if platform.system() == "Windows":
        print(f"   cd {frontend_path}")
    else:
        print(f"   cd {frontend_path}")
    print("   npm run dev")
    
    print("\n📹 TERMINAL 3 - Video Processing (Optional):")
    if platform.system() == "Windows":
        print(f"   cd {os.getcwd()}")
        print("   python smart_parking_mvp.py --run parking_evening_vedio.mp4")
    else:
        print(f"   cd {os.getcwd()}")
        print("   python3 smart_parking_mvp.py --run parking_evening_vedio.mp4")
    
    print("\n" + "=" * 70)
    print("🔗 ACCESS POINTS")
    print("=" * 70)
    print("   Frontend:  http://localhost:3000")
    print("   Backend:   http://localhost:5000")
    print("   API Docs:  http://localhost:5000/")
    if use_database:
        print("   Database:  localhost:5432 (parking_db)")
    
    print("\n" + "=" * 70)
    print("\nPress Enter to start the backend server now...")
    input()
    
    # Start server
    try:
        if platform.system() == "Windows":
            subprocess.run(["python", server_script])
        else:
            subprocess.run(["python3", server_script])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
