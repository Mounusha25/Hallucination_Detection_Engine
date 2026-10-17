from src.engine import evaluate
from src.models.schemas import EvaluationRequest, EvidenceDocument


def _request(response: str, evidence: str) -> EvaluationRequest:
    return EvaluationRequest(
        response=response,
        evidence=[EvidenceDocument(id="doc-1", text=evidence)],
    )


def test_supported_claim_has_low_risk() -> None:
    result = evaluate(
        _request(
            "Paris is the capital of France.",
            "France is in Europe and its capital city is Paris.",
        )
    )

    assert result.verdict == "low_risk"
    assert result.claims_supported == 1
    assert result.groundedness > 0.7


def test_unsupported_claim_has_high_risk() -> None:
    result = evaluate(
        _request(
            "Ada Lovelace invented the internet in 1843.",
            "Ada Lovelace wrote notes on the Analytical Engine in 1843.",
        )
    )

    assert result.verdict in {"medium_risk", "high_risk"}
    assert result.claims_supported == 0


def test_explicit_negation_conflict_is_contradiction() -> None:
    result = evaluate(
        _request(
            "The medication does not cause drowsiness.",
            "The medication causes drowsiness in some patients.",
        )
    )

    assert result.claims_contradicted == 1
    assert result.hallucination_score >= 0.9


def test_multiple_claims_are_scored_independently() -> None:
    request = EvaluationRequest(
        response="Paris is the capital of France. Sydney is the capital of Australia.",
        evidence=[
            EvidenceDocument(id="fr", text="The capital of France is Paris."),
            EvidenceDocument(id="au", text="The capital of Australia is Canberra."),
        ],
    )

    result = evaluate(request)

    assert result.claims_total == 2
    assert result.claims_supported == 1
