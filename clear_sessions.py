"""
Quick script to clear active parking sessions and reset for testing vehicle details
"""
import requests
import psycopg2

# Database connection
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    database="parking_db",
    user="parking_user",
    password="parking123"
)

cur = conn.cursor()

print("🧹 Clearing all active parking sessions...")

# End all active parking events
cur.execute("UPDATE parking_events SET departure_time = NOW(), parked_time = 0 WHERE departure_time IS NULL")
affected = cur.rowcount

conn.commit()
cur.close()
conn.close()

print(f"✅ Cleared {affected} active sessions")
print("\n📝 Next steps:")
print("1. Restart video processing: python smart_parking_mvp.py --run parking_evening_vedio.mp4")
print("2. Wait for NEW vehicles to be detected")
print("3. Refresh the frontend")
print("\nNEW detections will have vehicle details!")
