"""Validated request and result contracts for the evaluator."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.providers.base import RubricDimension


class EvidenceDocument(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=50_000)
    source: str | None = Field(default=None, max_length=2_000)

    @field_validator("id", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class EvaluationRequest(BaseModel):
    response: str = Field(min_length=1, max_length=20_000)
    evidence: list[EvidenceDocument] = Field(min_length=1, max_length=100)
    mode: Literal["baseline", "hybrid"] = "baseline"
    support_threshold: float = Field(default=0.70, ge=0, le=1)
    hallucination_threshold: float = Field(default=0.50, ge=0, le=1)
    top_k: int = Field(default=3, ge=1, le=20)
    lexical_weight: float = Field(default=0.40, ge=0, le=1)
    semantic_weight: float = Field(default=0.60, ge=0, le=1)

    @field_validator("response")
    @classmethod
    def strip_response(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("response must not be blank")
        return value


class EvidenceMatch(BaseModel):
    document_id: str
    source: str | None = None
    retrieval_score: float = Field(ge=0, le=1)
    lexical_score: float = Field(ge=0, le=1)
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    support_score: float = Field(ge=0, le=1)
    excerpt: str


class ClaimEvaluation(BaseModel):
    claim: str
    supported: bool
    contradicted: bool
    nli_label: Literal["entailment", "contradiction", "neutral"] | None = None
    nli_confidence: float | None = Field(default=None, ge=0, le=1)
    hallucination_score: float = Field(ge=0, le=1)
    evidence: list[EvidenceMatch]


class EvaluationResult(BaseModel):
    mode: Literal["baseline", "hybrid"]
    models: dict[str, str]
    verdict: Literal["low_risk", "medium_risk", "high_risk"]
    hallucination_score: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    claims_total: int = Field(ge=1)
    claims_supported: int = Field(ge=0)
    claims_contradicted: int = Field(ge=0)
    claims: list[ClaimEvaluation]
    latency_ms: float = Field(ge=0)


class RubricEvaluationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    response: str = Field(min_length=1, max_length=20_000)
    reference_answer: str = Field(min_length=1, max_length=20_000)
    evidence: list[EvidenceDocument] = Field(min_length=1, max_length=100)
    dimensions: list[RubricDimension] = Field(
        default_factory=lambda: [
            "correctness",
            "relevance",
            "completeness",
            "faithfulness",
        ],
        min_length=1,
    )
    pass_threshold: float = Field(default=0.70, ge=0, le=1)

    @field_validator("question", "response", "reference_answer")
    @classmethod
    def strip_rubric_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_unique(
        cls, value: list[RubricDimension]
    ) -> list[RubricDimension]:
        if len(value) != len(set(value)):
            raise ValueError("dimensions must be unique")
        return value


class RubricDimensionResult(BaseModel):
    dimension: RubricDimension
    score: float = Field(ge=0, le=1)
    passed: bool
    rationale: str


class RubricEvaluationResult(BaseModel):
    model: str
    overall_score: float = Field(ge=0, le=1)
    passed: bool
    pass_threshold: float = Field(ge=0, le=1)
    dimensions: list[RubricDimensionResult]
    latency_ms: float = Field(ge=0)
