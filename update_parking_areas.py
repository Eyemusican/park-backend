"""
Update Database with Real Parking Areas from Video Feed
This replaces dummy data with actual parking areas from camera_config.json
"""
import psycopg2
import json

def update_parking_areas():
    try:
        # Connect to database
        conn = psycopg2.connect(
            host='127.0.0.1',
            port=5432,
            database='parking_db',
            user='parking_user',
            password='parking123'
        )
        cur = conn.cursor()
        
        # Load camera config
        with open('camera_config.json', 'r') as f:
            config = json.load(f)
        
        print("=" * 60)
        print("UPDATING PARKING AREAS WITH REAL DATA FROM VIDEO FEED")
        print("=" * 60)
        
        # Update parking areas from camera config
        for camera in config['cameras']:
            parking_id = camera['parking_area_id']
            parking_name = camera['parking_area_name']
            
            # Check if parking area exists
            cur.execute('SELECT parking_name, slot_count FROM parking_area WHERE parking_id = %s', (parking_id,))
            existing = cur.fetchone()
            
            if existing:
                print(f"\n📝 Updating Parking Area {parking_id}:")
                print(f"   Old: {existing[0]} ({existing[1]} slots)")
                print(f"   New: {parking_name}")
                
                # Keep existing slot count or set to 0 if None
                slot_count = existing[1] if existing[1] is not None else 8
                
                # Update the parking area name
                cur.execute('''
                    UPDATE parking_area 
                    SET parking_name = %s 
                    WHERE parking_id = %s
                ''', (parking_name, parking_id))
            else:
                print(f"\n➕ Creating Parking Area {parking_id}: {parking_name}")
                # Insert new parking area
                cur.execute('''
                    INSERT INTO parking_area (parking_id, parking_name, slot_count)
                    VALUES (%s, %s, 8)
                ''', (parking_id, parking_name))
        
        conn.commit()
        
        # Show updated parking areas
        print("\n" + "=" * 60)
        print("UPDATED PARKING AREAS:")
        print("=" * 60)
        cur.execute('SELECT parking_id, parking_name, slot_count FROM parking_area ORDER BY parking_id')
        areas = cur.fetchall()
        
        for area in areas:
            print(f"ID: {area[0]}, Name: {area[1]}, Slots: {area[2]}")
        
        print("\n✅ Database updated successfully!")
        print("=" * 60)
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_parking_areas()
