"""End-to-end evidence-grounded hallucination evaluator."""

import time

from src.models.schemas import (
    ClaimEvaluation,
    EvaluationRequest,
    EvaluationResult,
    EvidenceMatch,
)
from src.retrieval.hybrid import BM25Retriever
from src.scoring.groundedness import is_contradiction, support_score
from src.utils.text import split_claims


def _verdict(score: float) -> str:
    if score < 0.30:
        return "low_risk"
    if score < 0.60:
        return "medium_risk"
    return "high_risk"


def evaluate(request: EvaluationRequest) -> EvaluationResult:
    started = time.perf_counter()
    retriever = BM25Retriever(request.evidence)
    claim_results: list[ClaimEvaluation] = []

    for claim in split_claims(request.response):
        retrieved = retriever.search(claim, request.top_k)
        matches: list[EvidenceMatch] = []
        contradiction = False
        best_support = 0.0
        for candidate in retrieved:
            score = support_score(claim, candidate.document.text)
            contradiction = contradiction or is_contradiction(claim, candidate.document.text)
            best_support = max(best_support, score)
            matches.append(
                EvidenceMatch(
                    document_id=candidate.document.id,
                    source=candidate.document.source,
                    retrieval_score=round(candidate.score, 4),
                    support_score=round(score, 4),
                    excerpt=candidate.document.text[:500],
                )
            )

        supported = best_support >= request.support_threshold and not contradiction
        if contradiction:
            risk = max(0.90, 1 - best_support)
        elif not supported:
            risk = max(0.60, 1 - best_support)
        else:
            risk = 1 - best_support
        claim_results.append(
            ClaimEvaluation(
                claim=claim,
                supported=supported,
                contradicted=contradiction,
                hallucination_score=round(max(0.0, min(1.0, risk)), 4),
                evidence=matches,
            )
        )

    total = len(claim_results)
    supported_count = sum(result.supported for result in claim_results)
    contradicted_count = sum(result.contradicted for result in claim_results)
    hallucination_score = sum(result.hallucination_score for result in claim_results) / total
    groundedness = 1 - hallucination_score
    coverage = supported_count / total
    latency_ms = (time.perf_counter() - started) * 1000

    return EvaluationResult(
        verdict=_verdict(hallucination_score),
        hallucination_score=round(hallucination_score, 4),
        groundedness=round(groundedness, 4),
        coverage=round(coverage, 4),
        claims_total=total,
        claims_supported=supported_count,
        claims_contradicted=contradicted_count,
        claims=claim_results,
        latency_ms=round(latency_ms, 2),
    )
