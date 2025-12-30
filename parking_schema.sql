-- Table for parking areas
CREATE TABLE parking_area (
    parking_id SERIAL PRIMARY KEY,
    parking_name VARCHAR(255) NOT NULL,
    slot_count numeric NULL,
    boundary_polygon JSONB,
    video_source VARCHAR(500),
    video_source_type VARCHAR(20) DEFAULT 'file',
    reference_frame_path VARCHAR(500)
);

-- Table for physical slots assigned to each area
CREATE TABLE parking_slots (
    slot_id SERIAL PRIMARY KEY,
    parking_id INT NOT NULL,
    slot_number INT NOT NULL,
    polygon_points JSONB,
    FOREIGN KEY (parking_id) REFERENCES parking_area(parking_id),
    UNIQUE(parking_id, slot_number)
);

-- Table for parking events
CREATE TABLE parking_events (
    event_id SERIAL PRIMARY KEY,
    slot_id INT NOT NULL,
    arrival_time TIMESTAMP NOT NULL,
    departure_time TIMESTAMP,
    parked_time INT,
    car_type VARCHAR(50),
    color VARCHAR(30),
    number_plate VARCHAR(20),
    FOREIGN KEY (slot_id) REFERENCES parking_slots(slot_id)
);
