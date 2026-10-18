"""End-to-end evidence-grounded hallucination evaluator."""

import time

from src.models.schemas import (
    ClaimEvaluation,
    EvaluationRequest,
    EvaluationResult,
    EvidenceMatch,
)
from src.providers.base import EmbeddingProvider, NLIProvider
from src.providers.ollama import OllamaEmbeddingProvider, OllamaNLIProvider
from src.retrieval.hybrid import BM25Retriever, HybridRetriever
from src.scoring.groundedness import is_contradiction, support_score
from src.utils.text import split_claims


def _verdict(score: float, threshold: float) -> str:
    if score < threshold * 0.60:
        return "low_risk"
    if score < threshold:
        return "medium_risk"
    return "high_risk"


def evaluate(
    request: EvaluationRequest,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    nli_provider: NLIProvider | None = None,
) -> EvaluationResult:
    started = time.perf_counter()
    if request.mode == "hybrid":
        embedding_provider = embedding_provider or OllamaEmbeddingProvider()
        nli_provider = nli_provider or OllamaNLIProvider()
        retriever = HybridRetriever(
            request.evidence,
            embedding_provider,
            lexical_weight=request.lexical_weight,
            semantic_weight=request.semantic_weight,
        )
        models = {
            "retrieval": f"bm25+{embedding_provider.name}:{embedding_provider.model}",
            "nli": f"{nli_provider.name}:{nli_provider.model}",
        }
    else:
        retriever = BM25Retriever(request.evidence)
        models = {"retrieval": "bm25", "nli": "explicit-negation-rules"}
    claim_results: list[ClaimEvaluation] = []

    for claim in split_claims(request.response):
        retrieved = retriever.search(claim, request.top_k)
        matches: list[EvidenceMatch] = []
        contradiction = False
        best_support = 0.0
        for index, candidate in enumerate(retrieved):
            score = support_score(claim, candidate.document.text)
            if index == 0:
                contradiction = is_contradiction(claim, candidate.document.text)
            best_support = max(best_support, score)
            matches.append(
                EvidenceMatch(
                    document_id=candidate.document.id,
                    source=candidate.document.source,
                    retrieval_score=round(candidate.score, 4),
                    lexical_score=round(candidate.lexical_score, 4),
                    semantic_score=(
                        round(candidate.semantic_score, 4)
                        if candidate.semantic_score is not None
                        else None
                    ),
                    support_score=round(score, 4),
                    excerpt=candidate.document.text[:500],
                )
            )

        nli_result = None
        if request.mode == "hybrid" and retrieved and nli_provider is not None:
            nli_result = nli_provider.classify(retrieved[0].document.text, claim)
            if nli_result.label == "entailment":
                best_support = max(best_support, nli_result.confidence)
            elif nli_result.label == "contradiction":
                contradiction = nli_result.confidence >= 0.50

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
                nli_label=nli_result.label if nli_result else None,
                nli_confidence=round(nli_result.confidence, 4) if nli_result else None,
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
        mode=request.mode,
        models=models,
        verdict=_verdict(hallucination_score, request.hallucination_threshold),
        hallucination_score=round(hallucination_score, 4),
        groundedness=round(groundedness, 4),
        coverage=round(coverage, 4),
        claims_total=total,
        claims_supported=supported_count,
        claims_contradicted=contradicted_count,
        claims=claim_results,
        latency_ms=round(latency_ms, 2),
    )
