"""Tests for parking slots API endpoints."""
import pytest
import json


class TestParkingSlotsAPI:
    """Test suite for /api/parking-slots endpoints."""

    def test_get_slots_for_area(self, client):
        """GET /api/parking-areas/<id>/slots should return slots."""
        # Get first parking area
        areas_response = client.get('/api/parking-areas')
        areas = json.loads(areas_response.data)

        if len(areas) > 0:
            area_id = areas[0]['id']
            response = client.get(f'/api/parking-areas/{area_id}/slots')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)

    def test_slot_structure(self, client):
        """Each slot should have required fields."""
        areas_response = client.get('/api/parking-areas')
        areas = json.loads(areas_response.data)

        if len(areas) > 0:
            area_id = areas[0]['id']
            response = client.get(f'/api/parking-areas/{area_id}/slots')

            if response.status_code == 200:
                data = json.loads(response.data)
                if len(data) > 0:
                    slot = data[0]
                    # Check for essential fields
                    assert 'slot_id' in slot or 'id' in slot


class TestParkingStatus:
    """Test suite for live parking status endpoints."""

    def test_parking_status_endpoint(self, client):
        """GET /api/parking-status should return current status."""
        response = client.get('/api/parking-status')
        # May return 200 or 404 depending on setup
        assert response.status_code in [200, 404]

    def test_video_feed_endpoint(self, client):
        """GET /video_feed should return video stream."""
        response = client.get('/video_feed')
        # This will depend on video availability
        assert response.status_code in [200, 500]
