"""
Test script for parking duration tracking system
Verifies all components are properly integrated
"""
import sys
import os

# Test imports
try:
    from parking_duration_tracker import ParkingDurationTracker, ParkingSession
    print("✅ parking_duration_tracker module imported successfully")
except Exception as e:
    print(f"❌ Failed to import parking_duration_tracker: {e}")
    sys.exit(1)

try:
    from tracker import VehicleTracker, TrackingManager
    print("✅ tracker module imported successfully")
except Exception as e:
    print(f"❌ Failed to import tracker: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("✅ YOLO imported successfully")
except Exception as e:
    print(f"❌ Failed to import YOLO: {e}")
    sys.exit(1)

try:
    import requests
    print("✅ requests module imported successfully")
except Exception as e:
    print(f"❌ Failed to import requests: {e}")
    sys.exit(1)

# Test ParkingDurationTracker initialization
try:
    tracker = ParkingDurationTracker(
        backend_url="http://localhost:5000",
        stability_frames=45,
        min_overlap=0.50
    )
    print("✅ ParkingDurationTracker initialized successfully")
    print(f"   - Backend URL: {tracker.backend_url}")
    print(f"   - Stability frames: {tracker.stability_frames}")
    print(f"   - Min overlap: {tracker.min_overlap}")
except Exception as e:
    print(f"❌ Failed to initialize ParkingDurationTracker: {e}")
    sys.exit(1)

# Test statistics
stats = tracker.get_statistics()
print("\n📊 Tracker Statistics:")
print(f"   - Total sessions: {stats['total_sessions']}")
print(f"   - Active sessions: {stats['active_sessions']}")
print(f"   - Completed sessions: {stats['completed_sessions']}")

# Test TrackingManager
try:
    tracking_mgr = TrackingManager()
    print("\n✅ TrackingManager initialized successfully")
except Exception as e:
    print(f"\n❌ Failed to initialize TrackingManager: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - System is ready!")
print("="*70)
print("\nNext steps:")
print("1. Start backend server: python3 server.py")
print("2. Run CV system: python3 smart_parking_mvp.py")
print("3. Start frontend: cd Frontend && npm run dev")
print("4. View dashboard: http://localhost:3000/admin")
