from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evaluate_endpoint() -> None:
    response = client.post(
        "/evaluate",
        json={
            "response": "Paris is the capital of France.",
            "evidence": [{"id": "geo", "text": "The capital of France is Paris."}],
        },
    )

    assert response.status_code == 200
    assert response.json()["claims_supported"] == 1


def test_rejects_empty_evidence() -> None:
    response = client.post("/evaluate", json={"response": "A claim.", "evidence": []})

    assert response.status_code == 422
