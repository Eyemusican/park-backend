-- Migration: Add geometry columns for parking area creation wizard
-- Date: 2025-01-29
-- Description: Adds polygon geometry storage to parking_slots and parking_area tables

BEGIN;

-- Add polygon geometry column to parking_slots
-- Stores 4-corner coordinates for slot boundary detection
-- Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
ALTER TABLE parking_slots
ADD COLUMN IF NOT EXISTS polygon_points JSONB;

-- Add geographic boundary and video source to parking_area
ALTER TABLE parking_area
ADD COLUMN IF NOT EXISTS boundary_polygon JSONB,
ADD COLUMN IF NOT EXISTS video_source VARCHAR(500),
ADD COLUMN IF NOT EXISTS video_source_type VARCHAR(20) DEFAULT 'file',
ADD COLUMN IF NOT EXISTS reference_frame_path VARCHAR(500);

-- Comments for documentation
COMMENT ON COLUMN parking_slots.polygon_points IS 'Slot polygon coordinates for CV detection: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]';
COMMENT ON COLUMN parking_area.boundary_polygon IS 'Geographic boundary for map display: [[lat1,lng1], [lat2,lng2], ...]';
COMMENT ON COLUMN parking_area.video_source IS 'File path or RTSP URL for video source';
COMMENT ON COLUMN parking_area.video_source_type IS 'Type of video source: file or rtsp';
COMMENT ON COLUMN parking_area.reference_frame_path IS 'Path to extracted reference frame for slot editing';

COMMIT;
