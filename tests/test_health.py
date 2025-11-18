from fastapi.testclient import TestClient
from serving.routers.health import health_router
from fastapi import FastAPI


app = FastAPI()
app.include_router(health_router)


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
