from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_api_status():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Fashion MNIST Model API is running"}


def test_predict_rejects_invalid_pixel_count():
    response = client.post(
        "/predict",
        json={"pixels": [0.0] * 783},
    )

    assert response.status_code == 422


def test_predict_rejects_pixels_outside_allowed_range():
    response = client.post(
        "/predict",
        json={"pixels": [-1.0] + [0.0] * 783},
    )

    assert response.status_code == 422
