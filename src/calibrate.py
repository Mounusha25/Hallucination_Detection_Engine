"""Calibrate the rubric judge against versioned human labels."""

import argparse
import json
import statistics
from pathlib import Path

from src.experiments import DEFAULT_DATABASE, record_run
from src.models.schemas import RubricEvaluationRequest
from src.providers.base import RubricProvider
from src.rubrics import evaluate_rubric


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_calibration(
    path: Path,
    *,
    provider: RubricProvider | None = None,
    tolerance: float = 0.25,
    pass_threshold: float = 0.70,
) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results = []
    pairs: list[tuple[float, float, str]] = []
    latencies = []
    for row in rows:
        request = RubricEvaluationRequest.model_validate(row["request"])
        result = evaluate_rubric(request, provider=provider)
        predicted = {item.dimension: item.score for item in result.dimensions}
        human = row["human_scores"]
        dimension_results = {}
        for dimension in request.dimensions:
            judge_score = predicted[dimension]
            human_score = float(human[dimension])
            absolute_error = abs(judge_score - human_score)
            pairs.append((judge_score, human_score, dimension))
            dimension_results[dimension] = {
                "judge_score": judge_score,
                "human_score": human_score,
                "absolute_error": round(absolute_error, 4),
                "agrees": absolute_error <= tolerance,
            }
        latencies.append(result.latency_ms)
        results.append(
            {
                "id": row["id"],
                "overall_score": result.overall_score,
                "latency_ms": result.latency_ms,
                "dimensions": dimension_results,
            }
        )

    true_positive = sum(
        predicted >= pass_threshold and human >= pass_threshold for predicted, human, _ in pairs
    )
    false_positive = sum(
        predicted >= pass_threshold and human < pass_threshold for predicted, human, _ in pairs
    )
    false_negative = sum(
        predicted < pass_threshold and human >= pass_threshold for predicted, human, _ in pairs
    )
    true_negative = sum(
        predicted < pass_threshold and human < pass_threshold for predicted, human, _ in pairs
    )
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(true_positive + true_negative, len(pairs))
    absolute_errors = [abs(predicted - human) for predicted, human, _ in pairs]
    per_dimension = {}
    for dimension in rows[0]["human_scores"]:
        selected = [
            abs(predicted - human)
            for predicted, human, name in pairs
            if name == dimension
        ]
        per_dimension[dimension] = {
            "mae": round(statistics.mean(selected), 4),
            "agreement_rate": round(statistics.mean(error <= tolerance for error in selected), 4),
        }
    return {
        "dataset": str(path),
        "mode": "rubric",
        "summary": {
            "cases": len(results),
            "ratings": len(pairs),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mae": round(statistics.mean(absolute_errors), 4),
            "agreement_rate": round(
                statistics.mean(error <= tolerance for error in absolute_errors), 4
            ),
            "p50_latency_ms": round(statistics.median(latencies), 2),
            "confusion_matrix": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
            },
            "per_dimension": per_dimension,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/rubric_calibration.jsonl"))
    parser.add_argument("--name", default="rubric-calibration-v1")
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--pass-threshold", type=float, default=0.70)
    parser.add_argument("--fail-under", type=float, default=0.60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    report = run_calibration(
        args.dataset,
        tolerance=args.tolerance,
        pass_threshold=args.pass_threshold,
    )
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_record:
        run = record_run(
            report,
            name=args.name,
            threshold=args.fail_under,
            database=args.database,
        )
        print(f"Recorded calibration {run['id']} in {args.database}")
    return int(report["summary"]["agreement_rate"] < args.fail_under)


if __name__ == "__main__":
    raise SystemExit(main())
