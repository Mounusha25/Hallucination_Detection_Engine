from pathlib import Path

from src.evaluate import run_benchmark


def test_sample_benchmark_meets_quality_floor() -> None:
    report = run_benchmark(Path("data/benchmark.jsonl"))

    assert report["summary"]["cases"] == 6
    assert report["summary"]["accuracy"] >= 0.8
