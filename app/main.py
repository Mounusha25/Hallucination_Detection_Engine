"""FastAPI service for claim-level hallucination evaluation."""

from fastapi import FastAPI, HTTPException

from src.engine import evaluate
from src.models.schemas import EvaluationRequest, EvaluationResult
from src.providers.base import ProviderError

app = FastAPI(
    title="Hallucination Detection Engine",
    version="0.2.0",
    description="Evidence-grounded, claim-level evaluation for LLM outputs.",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "hallucination-detection-engine",
        "version": "0.2.0",
        "modes": {
            "baseline": {"external_models": False},
            "hybrid": {
                "embeddings": "ollama:bge-m3",
                "nli": "ollama:qwen3:4b-instruct",
            },
        },
    }


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate_response(request: EvaluationRequest) -> EvaluationResult:
    try:
        return evaluate(request)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/evaluate/batch", response_model=list[EvaluationResult])
def evaluate_batch(requests: list[EvaluationRequest]) -> list[EvaluationResult]:
    try:
        return [evaluate(request) for request in requests]
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
