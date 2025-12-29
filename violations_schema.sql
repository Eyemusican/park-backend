-- Add violations table to the database schema

-- Table for parking violations
CREATE TABLE IF NOT EXISTS parking_violations (
    violation_id VARCHAR(50) PRIMARY KEY,
    slot_id INT,
    vehicle_id VARCHAR(50),
    license_plate VARCHAR(50),
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    duration_minutes DECIMAL(10,2),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    parking_area VARCHAR(255),
    additional_data TEXT
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_violations_status ON parking_violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON parking_violations(severity);
CREATE INDEX IF NOT EXISTS idx_violations_detected_at ON parking_violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_violations_slot_id ON parking_violations(slot_id);
