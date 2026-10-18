from pathlib import Path

from src.calibrate import run_calibration
from src.models.schemas import EvidenceDocument, RubricEvaluationRequest
from src.providers.base import RubricDimension, RubricScore
from src.rubrics import evaluate_rubric


class FixedRubricProvider:
    name = "fake"
    model = "rubric-test"

    def __init__(self, scores: dict[RubricDimension, float] | None = None):
        self.scores = scores or {
            "correctness": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "faithfulness": 1.0,
        }

    def judge(
        self,
        *,
        question: str,
        response: str,
        reference_answer: str,
        evidence: list[str],
        dimensions: list[RubricDimension],
    ) -> list[RubricScore]:
        return [
            RubricScore(dimension=dimension, score=self.scores[dimension], rationale="test")
            for dimension in dimensions
        ]


def test_rubric_evaluation_returns_dimension_scores() -> None:
    request = RubricEvaluationRequest(
        question="What is the capital of France?",
        response="Paris is the capital of France.",
        reference_answer="Paris is the capital of France.",
        evidence=[EvidenceDocument(id="geo", text="France's capital is Paris.")],
    )

    result = evaluate_rubric(request, provider=FixedRubricProvider())

    assert result.overall_score == 1.0
    assert result.passed is True
    assert len(result.dimensions) == 4


def test_rubric_requires_every_dimension_to_pass() -> None:
    provider = FixedRubricProvider(
        {
            "correctness": 1.0,
            "relevance": 1.0,
            "completeness": 0.5,
            "faithfulness": 1.0,
        }
    )
    request = RubricEvaluationRequest(
        question="Name two symptoms.",
        response="Fever.",
        reference_answer="Fever and cough.",
        evidence=[EvidenceDocument(id="health", text="Symptoms include fever and cough.")],
    )

    result = evaluate_rubric(request, provider=provider)

    assert result.overall_score == 0.875
    assert result.passed is False


def test_calibration_metrics_match_human_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        '{"id":"one","request":{"question":"Q?","response":"A.",'
        '"reference_answer":"A.","evidence":[{"id":"e","text":"A."}]},'
        '"human_scores":{"correctness":1,"relevance":1,"completeness":1,'
        '"faithfulness":1}}\n'
    )

    report = run_calibration(dataset, provider=FixedRubricProvider())

    assert report["summary"]["agreement_rate"] == 1.0
    assert report["summary"]["mae"] == 0.0
    assert report["summary"]["confusion_matrix"]["true_positive"] == 4
