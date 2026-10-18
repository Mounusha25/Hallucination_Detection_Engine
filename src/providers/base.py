"""Provider protocols keep the engine testable and model-agnostic."""

from dataclasses import dataclass
from typing import Literal, Protocol


class ProviderError(RuntimeError):
    """Raised when an optional external model provider cannot respond."""


class EmbeddingProvider(Protocol):
    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class NLIResult:
    label: Literal["entailment", "contradiction", "neutral"]
    confidence: float


class NLIProvider(Protocol):
    name: str
    model: str

    def classify(self, premise: str, hypothesis: str) -> NLIResult: ...


RubricDimension = Literal["correctness", "relevance", "completeness", "faithfulness"]


@dataclass(frozen=True)
class RubricScore:
    dimension: RubricDimension
    score: float
    rationale: str


class RubricProvider(Protocol):
    name: str
    model: str

    def judge(
        self,
        *,
        question: str,
        response: str,
        reference_answer: str,
        evidence: list[str],
        dimensions: list[RubricDimension],
    ) -> list[RubricScore]: ...
