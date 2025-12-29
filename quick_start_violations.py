"""
Quick Start - Setup and Test Violation System
Run this to set everything up automatically
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show the result"""
    print("\n" + "=" * 60)
    print(f"📌 {description}")
    print("=" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ Error:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Smart Parking - Violation System Quick Start          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        ("python setup_violations.py", "Setting up database schema"),
        ("python test_violations.py", "Creating sample violations"),
    ]
    
    success_count = 0
    
    for cmd, desc in steps:
        if run_command(cmd, desc):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✨ Setup Complete: {success_count}/{len(steps)} steps successful")
    print("=" * 60)
    
    if success_count == len(steps):
        print("""
✅ Violation system is ready!

Next steps:
1. Start the backend server:
   python server.py

2. Start the frontend (in another terminal):
   cd ../Frontend
   npm run dev

3. View violations at:
   http://localhost:3000/admin

4. Test API endpoints:
   http://localhost:5000/api/violations
   http://localhost:5000/api/violations/summary
        """)
    else:
        print("\n⚠️  Some steps failed. Please check the errors above.")
        print("   You may need to set up the database first.")

if __name__ == "__main__":
    main()
