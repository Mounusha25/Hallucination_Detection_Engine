from src.engine import evaluate
from src.models.schemas import EvaluationRequest, EvidenceDocument
from src.providers.base import NLIResult


class FakeEmbeddings:
    name = "fake"
    model = "embedding-test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class ContradictionNLI:
    name = "fake"
    model = "nli-test"

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return NLIResult(label="contradiction", confidence=0.98)


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


def test_hybrid_mode_uses_nli_contradiction() -> None:
    request = EvaluationRequest(
        mode="hybrid",
        response="Sydney is the capital of Australia.",
        evidence=[EvidenceDocument(id="au", text="Canberra is Australia's capital.")],
    )

    result = evaluate(
        request,
        embedding_provider=FakeEmbeddings(),
        nli_provider=ContradictionNLI(),
    )

    assert result.mode == "hybrid"
    assert result.claims_contradicted == 1
    assert result.claims[0].nli_label == "contradiction"
    assert result.hallucination_score >= 0.9
