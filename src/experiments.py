"""SQLite-backed experiment history for benchmark runs."""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATABASE = Path("data/experiments.db")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            dataset TEXT NOT NULL,
            mode TEXT NOT NULL,
            threshold REAL NOT NULL,
            summary_json TEXT NOT NULL,
            results_json TEXT NOT NULL
        )
        """
    )
    return connection


def record_run(
    report: dict[str, Any],
    *,
    name: str,
    threshold: float,
    database: Path = DEFAULT_DATABASE,
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(UTC).isoformat()
    mode = str(report.get("mode", "baseline"))
    with _connect(database) as connection:
        connection.execute(
            """
            INSERT INTO experiment_runs
                (id, name, created_at, dataset, mode, threshold, summary_json, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                created_at,
                str(report["dataset"]),
                mode,
                threshold,
                json.dumps(report["summary"]),
                json.dumps(report["results"]),
            ),
        )
    return get_run(run_id, database=database)


def _deserialize(row: sqlite3.Row, *, include_results: bool) -> dict[str, Any]:
    run = {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "dataset": row["dataset"],
        "mode": row["mode"],
        "threshold": row["threshold"],
        "summary": json.loads(row["summary_json"]),
    }
    if include_results:
        run["results"] = json.loads(row["results_json"])
    return run


def list_runs(*, limit: int = 50, database: Path = DEFAULT_DATABASE) -> list[dict[str, Any]]:
    with _connect(database) as connection:
        rows = connection.execute(
            "SELECT * FROM experiment_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_deserialize(row, include_results=False) for row in rows]


def get_run(run_id: str, *, database: Path = DEFAULT_DATABASE) -> dict[str, Any]:
    with _connect(database) as connection:
        row = connection.execute(
            "SELECT * FROM experiment_runs WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise KeyError(run_id)
    return _deserialize(row, include_results=True)


def compare_runs(run_ids: list[str], *, database: Path = DEFAULT_DATABASE) -> dict[str, Any]:
    if len(run_ids) < 2:
        raise ValueError("at least two run IDs are required")
    runs = [get_run(run_id, database=database) for run_id in run_ids]
    baseline = runs[0]["summary"]
    metrics = ("accuracy", "precision", "recall", "f1", "p50_latency_ms")
    return {
        "baseline_run_id": runs[0]["id"],
        "runs": runs,
        "deltas_from_baseline": {
            run["id"]: {
                metric: round(run["summary"][metric] - baseline[metric], 4)
                for metric in metrics
            }
            for run in runs[1:]
        },
    }
