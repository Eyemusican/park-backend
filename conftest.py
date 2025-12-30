"""Pytest configuration and fixtures for parking system tests."""
import pytest
import os
import sys

# Add park-backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_parking_area():
    """Sample parking area data for tests."""
    return {
        "name": "Test Parking Area",
        "slot_count": 10
    }
