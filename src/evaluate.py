"""Run a versioned JSONL benchmark and report binary detection metrics."""

import argparse
import json
import statistics
from pathlib import Path

from src.engine import evaluate
from src.experiments import DEFAULT_DATABASE, record_run
from src.models.schemas import EvaluationRequest


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_benchmark(path: Path, threshold: float = 0.50) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        result = evaluate(EvaluationRequest.model_validate(row["request"]))
        predicted = "hallucinated" if result.hallucination_score >= threshold else "supported"
        results.append(
            {
                "id": row["id"],
                "expected": row["expected_label"],
                "predicted": predicted,
                "score": result.hallucination_score,
                "latency_ms": result.latency_ms,
                "correct": predicted == row["expected_label"],
            }
        )

    true_positive = sum(r["predicted"] == r["expected"] == "hallucinated" for r in results)
    false_positive = sum(
        r["predicted"] == "hallucinated" and r["expected"] == "supported" for r in results
    )
    false_negative = sum(
        r["predicted"] == "supported" and r["expected"] == "hallucinated" for r in results
    )
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    modes = {row.get("request", {}).get("mode", "baseline") for row in rows}
    return {
        "dataset": str(path),
        "mode": modes.pop() if len(modes) == 1 else "mixed",
        "summary": {
            "cases": len(results),
            "accuracy": round(statistics.mean(r["correct"] for r in results), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in results), 2),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark.jsonl"))
    parser.add_argument("--fail-under", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name", help="Experiment name; defaults to the dataset filename")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--no-record", action="store_true", help="Do not save this run to SQLite")
    args = parser.parse_args()
    report = run_benchmark(args.dataset)
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_record:
        run = record_run(
            report,
            name=args.name or args.dataset.stem,
            threshold=args.fail_under,
            database=args.database,
        )
        print(f"Recorded experiment {run['id']} in {args.database}")
    return int(report["summary"]["accuracy"] < args.fail_under)


if __name__ == "__main__":
    raise SystemExit(main())
