from pathlib import Path

import pytest

from src.experiments import compare_runs, get_run, list_runs, record_run


def sample_report(accuracy: float, *, mode: str = "baseline") -> dict:
    return {
        "dataset": "data/test.jsonl",
        "mode": mode,
        "summary": {
            "cases": 2,
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
            "p50_latency_ms": 12.5,
        },
        "results": [{"id": "case-1", "correct": True}],
    }


def test_record_list_and_get_run(tmp_path: Path) -> None:
    database = tmp_path / "experiments.db"
    created = record_run(sample_report(1.0), name="baseline-v1", threshold=0.8, database=database)

    assert created["name"] == "baseline-v1"
    assert created["results"][0]["id"] == "case-1"
    assert list_runs(database=database)[0]["id"] == created["id"]
    assert "results" not in list_runs(database=database)[0]
    assert get_run(created["id"], database=database)["summary"]["accuracy"] == 1.0


def test_compare_runs_calculates_deltas(tmp_path: Path) -> None:
    database = tmp_path / "experiments.db"
    first = record_run(sample_report(0.5), name="first", threshold=0.8, database=database)
    second = record_run(sample_report(1.0), name="second", threshold=0.8, database=database)

    comparison = compare_runs([first["id"], second["id"]], database=database)

    assert comparison["deltas_from_baseline"][second["id"]]["accuracy"] == 0.5


def test_missing_run_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        get_run("missing", database=tmp_path / "experiments.db")
