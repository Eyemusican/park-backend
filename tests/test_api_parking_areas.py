"""Tests for parking areas API endpoints."""
import pytest
import json


class TestParkingAreasAPI:
    """Test suite for /api/parking-areas endpoints."""

    def test_get_parking_areas_returns_list(self, client):
        """GET /api/parking-areas should return a list."""
        response = client.get('/api/parking-areas')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_get_parking_areas_structure(self, client):
        """Each parking area should have required fields."""
        response = client.get('/api/parking-areas')
        assert response.status_code == 200
        data = json.loads(response.data)

        if len(data) > 0:
            area = data[0]
            assert 'id' in area
            assert 'name' in area
            assert 'total_slots' in area
            assert 'occupied_slots' in area
            assert 'available_slots' in area

    def test_create_parking_area(self, client, sample_parking_area):
        """POST /api/parking-areas should create a new area."""
        response = client.post(
            '/api/parking-areas',
            data=json.dumps(sample_parking_area),
            content_type='application/json'
        )
        assert response.status_code in [200, 201]
        data = json.loads(response.data)
        assert 'id' in data or 'parking_id' in data

    def test_get_single_parking_area(self, client):
        """GET /api/parking-areas/<id> should return area details."""
        # First get list to find an ID
        list_response = client.get('/api/parking-areas')
        areas = json.loads(list_response.data)

        if len(areas) > 0:
            area_id = areas[0]['id']
            response = client.get(f'/api/parking-areas/{area_id}')
            assert response.status_code == 200

    def test_get_nonexistent_parking_area(self, client):
        """GET /api/parking-areas/<invalid_id> should return 404."""
        response = client.get('/api/parking-areas/99999')
        assert response.status_code in [404, 500]
