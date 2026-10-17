"""FastAPI service for claim-level hallucination evaluation."""

from fastapi import FastAPI

from src.engine import evaluate
from src.models.schemas import EvaluationRequest, EvaluationResult

app = FastAPI(
    title="Hallucination Detection Engine",
    version="0.1.0",
    description="Evidence-grounded, claim-level evaluation for LLM outputs.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "hallucination-detection-engine", "version": "0.1.0"}


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate_response(request: EvaluationRequest) -> EvaluationResult:
    return evaluate(request)


@app.post("/evaluate/batch", response_model=list[EvaluationResult])
def evaluate_batch(requests: list[EvaluationRequest]) -> list[EvaluationResult]:
    return [evaluate(request) for request in requests]
