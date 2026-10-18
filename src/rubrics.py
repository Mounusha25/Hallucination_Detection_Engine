"""Multi-dimension response evaluation using a calibrated rubric judge."""

import time

from src.models.schemas import (
    RubricDimensionResult,
    RubricEvaluationRequest,
    RubricEvaluationResult,
)
from src.providers.base import RubricProvider
from src.providers.ollama import OllamaRubricProvider


def evaluate_rubric(
    request: RubricEvaluationRequest,
    *,
    provider: RubricProvider | None = None,
) -> RubricEvaluationResult:
    started = time.perf_counter()
    provider = provider or OllamaRubricProvider()
    scores = provider.judge(
        question=request.question,
        response=request.response,
        reference_answer=request.reference_answer,
        evidence=[document.text for document in request.evidence],
        dimensions=request.dimensions,
    )
    dimensions = [
        RubricDimensionResult(
            dimension=item.dimension,
            score=round(item.score, 4),
            passed=item.score >= request.pass_threshold,
            rationale=item.rationale,
        )
        for item in scores
    ]
    overall_score = sum(item.score for item in dimensions) / len(dimensions)
    return RubricEvaluationResult(
        model=f"{provider.name}:{provider.model}",
        overall_score=round(overall_score, 4),
        passed=all(item.passed for item in dimensions),
        pass_threshold=request.pass_threshold,
        dimensions=dimensions,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
