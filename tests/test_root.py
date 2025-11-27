from fastapi.testclient import TestClient
from serving.routers.root import root_router
from fastapi import FastAPI


app = FastAPI()
app.include_router(root_router)


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the IQA Web App" in response.text