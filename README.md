# Hallucination Detection Engine

An evidence-grounded evaluation service for LLM outputs. It breaks a response
into claims, retrieves the most relevant evidence for each claim, calculates
support and contradiction signals, and returns an explainable hallucination
risk report.

Unlike a demo that returns one opaque score, this project exposes the evidence,
claim-level decisions, latency, groundedness, and coverage needed to debug an
AI system and enforce quality gates in CI.

## Current milestone

The repository contains a working deterministic baseline:

- validated FastAPI request and response contracts;
- sentence-level claim extraction;
- in-memory BM25 evidence retrieval;
- transparent token-support and explicit-negation contradiction metrics;
- per-claim evidence attribution;
- single and batch evaluation endpoints;
- CLI evaluation;
- a versioned JSONL benchmark and regression runner;
- automated tests, linting, and GitHub Actions CI.

The baseline deliberately runs without downloading a large model. Optional
embedding and NLI adapters are the next milestone and are separated into the
`ml` dependency group.

## Architecture

```text
LLM response + trusted evidence
              |
              v
      Claim segmentation
              |
              v
       BM25 retrieval
              |
              v
  Support + contradiction scoring
              |
              v
Claim evidence, risk, coverage, latency
              |
       +------+------+
       |             |
   FastAPI         CLI / CI
```

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

## API example

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Sydney is the capital of Australia.",
    "evidence": [
      {
        "id": "atlas-1",
        "text": "Canberra is the capital city of Australia.",
        "source": "reference atlas"
      }
    ]
  }'
```

The response includes:

```json
{
  "verdict": "high_risk",
  "hallucination_score": 0.6,
  "groundedness": 0.4,
  "coverage": 0.0,
  "claims_total": 1,
  "claims_supported": 0,
  "claims_contradicted": 0,
  "claims": [
    {
      "claim": "Sydney is the capital of Australia.",
      "supported": false,
      "contradicted": false,
      "hallucination_score": 0.6,
      "evidence": [
        {
          "document_id": "atlas-1",
          "source": "reference atlas",
          "retrieval_score": 1.0,
          "support_score": 0.6,
          "excerpt": "Canberra is the capital city of Australia."
        }
      ]
    }
  ]
}
```

Scores are continuous and the complete response also contains ranked evidence
matches and request latency.

## CLI

Evaluate one response:

```bash
python -m src.detect \
  --response "Paris is the capital of France." \
  --evidence "France is a country in Europe and its capital is Paris."
```

Run the versioned benchmark:

```bash
python -m src.evaluate \
  --dataset data/benchmark.jsonl \
  --fail-under 0.80 \
  --output results/latest.json
```

The command reports accuracy, precision, recall, F1, and median latency. It
returns a non-zero exit status when accuracy drops below the quality threshold,
making it suitable for deployment gates.

## Development

```bash
ruff check .
pytest
python -m src.evaluate --fail-under 0.80
```

Install optional local ML dependencies with:

```bash
pip install -e ".[dev,ml]"
```

They are not required for the baseline and can be large.

## Repository structure

```text
app/main.py                 FastAPI service
src/engine.py               Evaluation orchestration
src/detect.py               Single-response CLI
src/evaluate.py             Benchmark and regression runner
src/models/schemas.py       Validated contracts
src/retrieval/hybrid.py     BM25 evidence retrieval
src/scoring/groundedness.py Explainable scoring
src/utils/text.py           Claim and token utilities
data/benchmark.jsonl        Versioned baseline dataset
tests/                      Unit and API tests
.github/workflows/ci.yml    Automated quality gate
```

## Roadmap

1. Sentence-transformer semantic retrieval and score fusion.
2. DeBERTa NLI entailment/contradiction adapter.
3. Prompt/model experiment tracking in SQLite.
4. LLM-as-judge rubric with calibration against human labels.
5. RAG retrieval metrics, citation validation, and tool-call evaluation.
6. Web dashboard for run comparison and failure analysis.
7. FoodPilot integration as the first external evaluated application.

## Scope and limitations

The current scorer is intentionally an explainable lexical baseline. It is
useful for regression testing and obvious factual support/contradiction cases,
but paraphrases and subtle reasoning require the planned semantic and NLI
adapters. The README describes implemented behavior separately from roadmap
work so benchmark claims remain reproducible.
