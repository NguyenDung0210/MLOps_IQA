from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import serving.routers.predict as predict_module


@pytest.fixture
def mock_model():
    mock_model = MagicMock()
    import torch
    mock_model.return_value = torch.tensor([[0.5]])
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None
    return mock_model


@pytest.fixture
def client(mock_model):
    # patch model in predict module
    with patch("serving.routers.predict.load_model", return_value=mock_model):
        predict_module.model = mock_model
        app = FastAPI()
        app.include_router(predict_module.predict_router)
        client = TestClient(app)
        yield client


def test_predict_endpoint(client):
    with open("tests/test.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    assert response.status_code == 200
    assert "quality" in response.json()
    assert response.json()["quality"] == 50.0
