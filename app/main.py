"""FastAPI service for claim-level hallucination evaluation and experiments."""

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.engine import evaluate
from src.experiments import compare_runs, get_run, list_runs
from src.models.schemas import (
    EvaluationRequest,
    EvaluationResult,
    RubricEvaluationRequest,
    RubricEvaluationResult,
)
from src.providers.base import ProviderError
from src.rubrics import evaluate_rubric

app = FastAPI(
    title="Hallucination Detection Engine",
    version="0.4.0",
    description="Evidence-grounded, claim-level evaluation for LLM outputs.",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "hallucination-detection-engine",
        "version": "0.4.0",
        "modes": {
            "baseline": {"external_models": False},
            "hybrid": {
                "embeddings": "ollama:bge-m3",
                "nli": "ollama:qwen3:4b-instruct",
            },
            "rubric": {
                "judge": "ollama:qwen3:4b-instruct",
                "dimensions": [
                    "correctness",
                    "relevance",
                    "completeness",
                    "faithfulness",
                ],
            },
        },
    }


def _database_path() -> Path:
    return Path(os.environ.get("HALLU_DB_PATH", "data/experiments.db"))


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return (Path(__file__).with_name("dashboard.html")).read_text()


@app.get("/runs")
def experiment_runs(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return list_runs(limit=limit, database=_database_path())


@app.get("/runs/{run_id}")
def experiment_run(run_id: str) -> dict:
    try:
        return get_run(run_id, database=_database_path())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment run not found") from exc


@app.get("/compare")
def compare_experiment_runs(run_id: Annotated[list[str], Query(min_length=2)]) -> dict:
    try:
        return compare_runs(run_id, database=_database_path())
    except KeyError as exc:
        detail = f"Experiment run {exc.args[0]} not found"
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate_response(request: EvaluationRequest) -> EvaluationResult:
    try:
        return evaluate(request)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/evaluate/rubric", response_model=RubricEvaluationResult)
def evaluate_response_with_rubric(request: RubricEvaluationRequest) -> RubricEvaluationResult:
    try:
        return evaluate_rubric(request)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/evaluate/batch", response_model=list[EvaluationResult])
def evaluate_batch(requests: list[EvaluationRequest]) -> list[EvaluationResult]:
    try:
        return [evaluate(request) for request in requests]
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
