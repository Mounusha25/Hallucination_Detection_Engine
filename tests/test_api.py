from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from src.experiments import record_run

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


def test_dashboard_is_available() -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "Experiment history" in response.text


def test_run_history_and_comparison_api(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "experiments.db"
    monkeypatch.setenv("HALLU_DB_PATH", str(database))
    report = {
        "dataset": "benchmark.jsonl",
        "mode": "baseline",
        "summary": {
            "cases": 1,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "p50_latency_ms": 5.0,
        },
        "results": [],
    }
    first = record_run(report, name="first", threshold=0.8, database=database)
    second = record_run(report, name="second", threshold=0.8, database=database)

    history = client.get("/runs")
    detail = client.get(f"/runs/{first['id']}")
    comparison = client.get(
        "/compare", params=[("run_id", first["id"]), ("run_id", second["id"])]
    )

    assert history.status_code == 200
    assert len(history.json()) == 2
    assert detail.json()["name"] == "first"
    assert comparison.status_code == 200
    assert comparison.json()["baseline_run_id"] == first["id"]


def test_missing_experiment_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HALLU_DB_PATH", str(tmp_path / "experiments.db"))

    assert client.get("/runs/missing").status_code == 404


def test_rubric_endpoint_validates_duplicate_dimensions() -> None:
    response = client.post(
        "/evaluate/rubric",
        json={
            "question": "What is the capital of France?",
            "response": "Paris.",
            "reference_answer": "Paris is the capital of France.",
            "evidence": [{"id": "geo", "text": "France's capital is Paris."}],
            "dimensions": ["correctness", "correctness"],
        },
    )

    assert response.status_code == 422
