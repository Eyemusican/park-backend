-- Smart Parking System - Camera Management Schema
-- Migration for RTSP camera support

-- Camera configuration table
CREATE TABLE IF NOT EXISTS cameras (
    camera_id SERIAL PRIMARY KEY,
    camera_name VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(500) NOT NULL,
    parking_id INTEGER REFERENCES parking_area(parking_id) ON DELETE SET NULL,
    username VARCHAR(100),
    password_encrypted VARCHAR(255),
    buffer_size INTEGER DEFAULT 1,
    timeout_seconds INTEGER DEFAULT 10,
    retry_interval_seconds INTEGER DEFAULT 5,
    max_retries INTEGER DEFAULT 3,
    is_active BOOLEAN DEFAULT true,
    status VARCHAR(50) DEFAULT 'DISCONNECTED',
    last_connected_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Camera health logs for monitoring
CREATE TABLE IF NOT EXISTS camera_health_logs (
    log_id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(camera_id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    fps DECIMAL(5,2),
    frame_latency_ms INTEGER,
    error_message TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_cameras_parking_id ON cameras(parking_id);
CREATE INDEX IF NOT EXISTS idx_cameras_is_active ON cameras(is_active);
CREATE INDEX IF NOT EXISTS idx_cameras_status ON cameras(status);
CREATE INDEX IF NOT EXISTS idx_camera_health_logs_camera_id ON camera_health_logs(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_health_logs_logged_at ON camera_health_logs(logged_at);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_cameras_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cameras_updated_at_trigger ON cameras;
CREATE TRIGGER cameras_updated_at_trigger
    BEFORE UPDATE ON cameras
    FOR EACH ROW
    EXECUTE FUNCTION update_cameras_updated_at();

-- Add comment for documentation
COMMENT ON TABLE cameras IS 'RTSP camera configuration for parking lot monitoring';
COMMENT ON TABLE camera_health_logs IS 'Health metrics history for camera monitoring';
