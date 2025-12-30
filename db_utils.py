"""
Database utility commands for the Smart Parking System.

Usage:
    python db_utils.py <command> [options]

Commands:
    cleanup     - Remove duplicate parking areas and fix data inconsistencies
    clear       - Clear all active parking sessions
    reset       - Reset parking data (mark all sessions as departed)
    check       - Verify database schema is correct
    view        - Display current database contents
    monitor     - Real-time database monitoring

Examples:
    python db_utils.py view
    python db_utils.py monitor --interval 5
    python db_utils.py reset --parking-id 7
"""

import argparse
import os
import sys
import time
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor


def get_db_connection():
    """Create a database connection using environment variables."""
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=os.environ.get('DB_PORT', '5432'),
        dbname=os.environ.get('DB_NAME', 'parking_db'),
        user=os.environ.get('DB_USER', 'parking_user'),
        password=os.environ.get('DB_PASS', 'parking_password')
    )


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)


# =============================================================================
# CLEANUP - Remove duplicates and fix inconsistencies
# =============================================================================

def cleanup_database():
    """Remove duplicate parking areas and sync slot status."""
    print_section("DATABASE CLEANUP")

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Show current state
        cur.execute("""
            SELECT
                pa.parking_id,
                pa.parking_name,
                pa.slot_count,
                COUNT(DISTINCT ps.slot_id) as configured_slots,
                COUNT(DISTINCT pe.event_id) as total_events,
                MAX(pe.arrival_time) as last_activity
            FROM parking_area pa
            LEFT JOIN parking_slots ps ON pa.parking_id = ps.parking_id
            LEFT JOIN parking_events pe ON ps.slot_id = pe.slot_id
            GROUP BY pa.parking_id, pa.parking_name, pa.slot_count
            ORDER BY pa.parking_id
        """)

        areas = cur.fetchall()

        print("\nCurrent Parking Areas:")
        for area in areas:
            print(f"  ID {area['parking_id']}: {area['parking_name']}")
            print(f"    Slots: {area['configured_slots']}/{area['slot_count']}, Events: {area['total_events']}")

        if len(areas) <= 1:
            print("\nNo duplicates found. Database is clean.")
        else:
            # Find duplicates by name
            cur.execute("""
                SELECT parking_name, array_agg(parking_id) as ids
                FROM parking_area
                GROUP BY parking_name
                HAVING COUNT(*) > 1
            """)
            duplicates = cur.fetchall()

            for dup in duplicates:
                name = dup['parking_name']
                ids = dup['ids']
                print(f"\nFound duplicate: '{name}' with IDs: {ids}")

                # Keep the one with most recent activity
                keep_id = ids[0]  # Default to first
                cur.execute("""
                    SELECT ps.parking_id
                    FROM parking_events pe
                    JOIN parking_slots ps ON pe.slot_id = ps.slot_id
                    WHERE ps.parking_id = ANY(%s)
                    ORDER BY pe.arrival_time DESC
                    LIMIT 1
                """, (ids,))
                result = cur.fetchone()
                if result:
                    keep_id = result['parking_id']

                print(f"Keeping parking_id {keep_id}")

                for pid in ids:
                    if pid != keep_id:
                        # Delete events, slots, then area
                        cur.execute("""
                            DELETE FROM parking_events
                            WHERE slot_id IN (SELECT slot_id FROM parking_slots WHERE parking_id = %s)
                        """, (pid,))
                        cur.execute("DELETE FROM parking_slots WHERE parking_id = %s", (pid,))
                        cur.execute("DELETE FROM parking_area WHERE parking_id = %s", (pid,))
                        print(f"  Removed parking_id {pid}")

        conn.commit()
        cur.close()
        conn.close()

        print("\nCleanup complete!")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


# =============================================================================
# CLEAR - Clear all active sessions
# =============================================================================

def clear_sessions():
    """Clear all active parking sessions."""
    print_section("CLEAR ACTIVE SESSIONS")

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE parking_events
            SET departure_time = NOW(),
                parked_time = EXTRACT(EPOCH FROM (NOW() - arrival_time))/60
            WHERE departure_time IS NULL
        """)
        affected = cur.rowcount

        conn.commit()
        cur.close()
        conn.close()

        print(f"\nCleared {affected} active session(s)")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# RESET - Reset parking area for fresh start
# =============================================================================

def reset_parking(parking_id=None):
    """Reset parking data - mark all active sessions as departed."""
    print_section("RESET PARKING DATA")

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if parking_id:
            # Reset specific parking area
            cur.execute("""
                UPDATE parking_events pe
                SET departure_time = CURRENT_TIMESTAMP,
                    parked_time = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - pe.arrival_time))/60
                FROM parking_slots ps
                WHERE pe.slot_id = ps.slot_id
                AND ps.parking_id = %s
                AND pe.departure_time IS NULL
            """, (parking_id,))
            affected = cur.rowcount
            print(f"\nReset parking area {parking_id}: {affected} session(s) cleared")
        else:
            # Reset all parking areas
            cur.execute("""
                UPDATE parking_events
                SET departure_time = CURRENT_TIMESTAMP,
                    parked_time = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - arrival_time))/60
                WHERE departure_time IS NULL
            """)
            affected = cur.rowcount
            print(f"\nReset all parking areas: {affected} session(s) cleared")

        conn.commit()
        cur.close()
        conn.close()

        print("All slots are now FREE")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# CHECK - Verify database schema
# =============================================================================

def check_structure():
    """Verify database schema is correct."""
    print_section("DATABASE SCHEMA CHECK")

    required_tables = ['parking_area', 'parking_slots', 'parking_events', 'parking_violations']

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        print("\nConnection: OK")

        # Check tables exist
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]

        print("\nTables found:")
        for table in tables:
            status = "OK" if table in required_tables else ""
            print(f"  - {table} {status}")

        # Check required tables
        missing = [t for t in required_tables if t not in tables]
        if missing:
            print(f"\nMissing required tables: {missing}")
            return False

        # Check key columns in parking_events
        print("\nParking events columns:")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'parking_events'
            ORDER BY ordinal_position
        """)
        for col in cur.fetchall():
            print(f"  - {col[0]}: {col[1]}")

        cur.close()
        conn.close()

        print("\nSchema check: OK")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# VIEW - Display database contents
# =============================================================================

def view_data():
    """Display current database contents."""
    print_section("DATABASE CONTENTS")

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Statistics
        print("\nStatistics:")

        cur.execute('SELECT COUNT(*) FROM parking_area')
        print(f"  Parking Areas: {cur.fetchone()[0]}")

        cur.execute('SELECT COUNT(*) FROM parking_slots')
        total_slots = cur.fetchone()[0]
        print(f"  Total Slots: {total_slots}")

        cur.execute('SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL')
        active = cur.fetchone()[0]
        print(f"  Active Sessions: {active}")

        cur.execute('SELECT COUNT(*) FROM parking_events')
        print(f"  Total Events: {cur.fetchone()[0]}")

        if total_slots > 0:
            print(f"  Occupancy: {(active/total_slots)*100:.1f}%")

        # Parking Areas
        print("\nParking Areas:")
        cur.execute('SELECT parking_id, parking_name, slot_count FROM parking_area ORDER BY parking_id')
        for row in cur.fetchall():
            print(f"  [{row[0]}] {row[1]} ({row[2]} slots)")

        # Latest Events
        print("\nLatest Events (last 10):")
        cur.execute("""
            SELECT pe.event_id, ps.slot_number, pa.parking_name,
                   pe.arrival_time, pe.departure_time
            FROM parking_events pe
            JOIN parking_slots ps ON pe.slot_id = ps.slot_id
            JOIN parking_area pa ON ps.parking_id = pa.parking_id
            ORDER BY pe.event_id DESC
            LIMIT 10
        """)
        for row in cur.fetchall():
            status = "ACTIVE" if row[4] is None else "departed"
            time_str = row[3].strftime("%H:%M:%S") if row[3] else "N/A"
            print(f"  Event {row[0]}: Slot {row[1]} @ {row[2]} - {status} ({time_str})")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# MONITOR - Real-time database monitoring
# =============================================================================

def monitor_live(interval=2):
    """Real-time database monitoring."""
    print_section("LIVE DATABASE MONITOR")
    print(f"Refresh interval: {interval}s (Press Ctrl+C to stop)")

    last_event_id = 0

    try:
        while True:
            conn = get_db_connection()
            cur = conn.cursor()

            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nLIVE MONITOR - {now}")
            print("="*60)

            # Stats
            cur.execute('SELECT COUNT(*) FROM parking_slots')
            total = cur.fetchone()[0]

            cur.execute('SELECT COUNT(*) FROM parking_events WHERE departure_time IS NULL')
            active = cur.fetchone()[0]

            cur.execute('SELECT COUNT(*) FROM parking_events')
            total_events = cur.fetchone()[0]

            print(f"\nSlots: {active}/{total} occupied ({(active/total*100):.0f}%)" if total > 0 else "\nNo slots")
            print(f"Total Events: {total_events}")

            # Latest events
            print("\nLatest Events:")
            print("-"*60)

            cur.execute("""
                SELECT pe.event_id, ps.slot_number, pa.parking_name,
                       pe.arrival_time, pe.departure_time
                FROM parking_events pe
                JOIN parking_slots ps ON pe.slot_id = ps.slot_id
                JOIN parking_area pa ON ps.parking_id = pa.parking_id
                ORDER BY pe.event_id DESC
                LIMIT 10
            """)

            for row in cur.fetchall():
                is_new = row[0] > last_event_id
                prefix = "NEW " if is_new else "    "
                status = "ACTIVE" if row[4] is None else "left"
                time_str = row[3].strftime("%H:%M:%S") if row[3] else ""
                print(f"{prefix}[{row[0]}] Slot {row[1]}: {status} {time_str}")
                if is_new:
                    last_event_id = row[0]

            cur.close()
            conn.close()

            print("-"*60)
            print(f"Refreshing in {interval}s...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Database utilities for Smart Parking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # cleanup
    subparsers.add_parser('cleanup', help='Remove duplicates and fix inconsistencies')

    # clear
    subparsers.add_parser('clear', help='Clear all active parking sessions')

    # reset
    reset_parser = subparsers.add_parser('reset', help='Reset parking data')
    reset_parser.add_argument('--parking-id', type=int, help='Specific parking area ID to reset')

    # check
    subparsers.add_parser('check', help='Verify database schema')

    # view
    subparsers.add_parser('view', help='Display database contents')

    # monitor
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--interval', type=int, default=2, help='Refresh interval in seconds')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Execute command
    if args.command == 'cleanup':
        cleanup_database()
    elif args.command == 'clear':
        clear_sessions()
    elif args.command == 'reset':
        reset_parking(args.parking_id)
    elif args.command == 'check':
        check_structure()
    elif args.command == 'view':
        view_data()
    elif args.command == 'monitor':
        monitor_live(args.interval)


if __name__ == "__main__":
    main()
