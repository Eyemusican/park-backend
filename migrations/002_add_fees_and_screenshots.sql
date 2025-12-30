-- Migration: Add fees and screenshots support
-- Description: Adds fee configuration to parking_area and fee/screenshot fields to parking_events

BEGIN;

-- Add fee configuration columns to parking_area table
ALTER TABLE parking_area
ADD COLUMN IF NOT EXISTS hourly_rate NUMERIC(10,2) DEFAULT 20.00,
ADD COLUMN IF NOT EXISTS currency VARCHAR(10) DEFAULT 'Nu.',
ADD COLUMN IF NOT EXISTS grace_period_minutes INT DEFAULT 15;

-- Add fee and screenshot columns to parking_events table
ALTER TABLE parking_events
ADD COLUMN IF NOT EXISTS fee_amount NUMERIC(10,2),
ADD COLUMN IF NOT EXISTS entry_photo_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS exit_photo_path VARCHAR(500);

-- Add vehicle_id column if not exists (some events may already have this)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'parking_events' AND column_name = 'vehicle_id'
    ) THEN
        ALTER TABLE parking_events ADD COLUMN vehicle_id VARCHAR(50);
    END IF;
END $$;

-- Create index for faster screenshot lookups
CREATE INDEX IF NOT EXISTS idx_parking_events_entry_photo ON parking_events(entry_photo_path) WHERE entry_photo_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_parking_events_exit_photo ON parking_events(exit_photo_path) WHERE exit_photo_path IS NOT NULL;

COMMIT;
