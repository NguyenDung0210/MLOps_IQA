from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)

def test_predict_endpoint():
    with open("tests/test.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    assert response.status_code == 200
    assert "quality" in response.json()