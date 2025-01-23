from fastapi.testclient import TestClient
from src.wikipedia.api import app

# Client that can be used to send requests to the API
client = TestClient(app)

# API integration test 
def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the MNIST model inference API!"}

# Run in terminal: pytest tests/test_api.py 