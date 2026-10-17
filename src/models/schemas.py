"""Validated request and result contracts for the evaluator."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
    support_threshold: float = Field(default=0.70, ge=0, le=1)
    hallucination_threshold: float = Field(default=0.50, ge=0, le=1)
    top_k: int = Field(default=3, ge=1, le=20)

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
    support_score: float = Field(ge=0, le=1)
    excerpt: str


class ClaimEvaluation(BaseModel):
    claim: str
    supported: bool
    contradicted: bool
    hallucination_score: float = Field(ge=0, le=1)
    evidence: list[EvidenceMatch]


class EvaluationResult(BaseModel):
    verdict: Literal["low_risk", "medium_risk", "high_risk"]
    hallucination_score: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    claims_total: int = Field(ge=1)
    claims_supported: int = Field(ge=0)
    claims_contradicted: int = Field(ge=0)
    claims: list[ClaimEvaluation]
    latency_ms: float = Field(ge=0)
