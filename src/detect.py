"""CLI for evaluating one generated response against supplied evidence."""

import argparse
import json
from pathlib import Path

from src.engine import evaluate
from src.models.schemas import EvaluationRequest, EvidenceDocument


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, help="LLM response to evaluate")
    parser.add_argument("--evidence", action="append", default=[], help="Evidence text; repeatable")
    parser.add_argument("--evidence-file", type=Path, help="JSON list of evidence documents")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid"],
        default="baseline",
        help="Hybrid uses local Ollama embeddings and NLI judging.",
    )
    args = parser.parse_args()

    documents = [
        EvidenceDocument(id=f"inline-{index}", text=text)
        for index, text in enumerate(args.evidence, start=1)
    ]
    if args.evidence_file:
        documents.extend(
            EvidenceDocument.model_validate(item)
            for item in json.loads(args.evidence_file.read_text())
        )
    if not documents:
        parser.error("provide --evidence or --evidence-file")

    result = evaluate(
        EvaluationRequest(response=args.response, evidence=documents, mode=args.mode)
    )
    print(result.model_dump_json(indent=2))
    return int(result.hallucination_score >= 0.50)


if __name__ == "__main__":
    raise SystemExit(main())
