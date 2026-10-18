import json

from src.providers.ollama import (
    OllamaEmbeddingProvider,
    OllamaNLIProvider,
    OllamaRubricProvider,
)


class FakeClient:
    def __init__(self, responses: list[dict]):
        self.responses = responses

    def post(self, path: str, payload: dict) -> dict:
        return self.responses.pop(0)


def test_embedding_provider_validates_shape() -> None:
    provider = OllamaEmbeddingProvider(client=FakeClient([{"embeddings": [[1.0], [2.0]]}]))

    assert provider.embed(["one", "two"]) == [[1.0], [2.0]]


def test_nli_provider_parses_structured_result() -> None:
    provider = OllamaNLIProvider(
        client=FakeClient(
            [
                {
                    "message": {
                        "content": json.dumps(
                            {"label": "entailment", "confidence": 0.93}
                        )
                    }
                }
            ]
        )
    )

    result = provider.classify("Paris is France's capital.", "Paris is the capital of France.")

    assert result.label == "entailment"
    assert result.confidence == 0.93


def test_rubric_provider_parses_scores_in_requested_order() -> None:
    provider = OllamaRubricProvider(
        client=FakeClient(
            [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "relevance": {
                                        "score": 0.8,
                                        "rationale": "Directly addresses the question.",
                                    },
                                "correctness": {
                                        "score": 1.0,
                                        "rationale": "Matches the reference.",
                                    },
                            }
                        )
                    }
                }
            ]
        )
    )

    result = provider.judge(
        question="Question?",
        response="Answer.",
        reference_answer="Answer.",
        evidence=["Answer."],
        dimensions=["correctness", "relevance"],
    )

    assert [item.dimension for item in result] == ["correctness", "relevance"]
    assert result[0].score == 1.0
