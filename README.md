# Hallucination Detection Engine

An evidence-grounded evaluation service for LLM outputs. It breaks a response
into claims, retrieves the most relevant evidence for each claim, calculates
support and contradiction signals, and returns an explainable hallucination
risk report.

Unlike a demo that returns one opaque score, this project exposes the evidence,
claim-level decisions, latency, groundedness, and coverage needed to debug an
AI system and enforce quality gates in CI.

## Current milestone

The repository contains two working evaluation modes:

- `baseline`: deterministic BM25 retrieval plus transparent lexical and
  explicit-negation scoring; no external model is required.
- `hybrid`: weighted BM25 + `bge-m3` embedding retrieval followed by a
  structured NLI-style judge using `qwen3:4b-instruct` through local Ollama.

Both modes provide:

- validated FastAPI request and response contracts;
- sentence-level claim extraction;
- in-memory evidence retrieval with per-signal scores;
- transparent token-support and explicit-negation contradiction metrics;
- per-claim evidence attribution;
- single and batch evaluation endpoints;
- CLI evaluation;
- a versioned JSONL benchmark and regression runner;
- automated tests, linting, and GitHub Actions CI.

The baseline deliberately runs without downloading a large model and remains
the default CI quality gate. Hybrid evaluation is an explicit opt-in so a model
server outage never silently changes deterministic results.

## Architecture

```text
LLM response + trusted evidence
              |
              v
      Claim segmentation
              |
              v
 BM25 or hybrid dense retrieval
              |
              v
Support + NLI contradiction scoring
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

For hybrid mode, also install [Ollama](https://ollama.com/) and pull the local
models:

```bash
ollama pull bge-m3
ollama pull qwen3:4b-instruct
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
    "mode": "hybrid",
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

Use semantic retrieval and NLI judging:

```bash
python -m src.detect \
  --mode hybrid \
  --response "A doctor treats illnesses." \
  --evidence "Whales live in oceans." \
  --evidence "A physician treats illnesses and cares for sick people."
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

Run the separate local-model benchmark after starting Ollama:

```bash
python -m src.evaluate \
  --dataset data/hybrid_benchmark.jsonl \
  --fail-under 0.80 \
  --output results/hybrid-latest.json
```

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
src/providers/ollama.py     Local embedding and NLI provider
src/utils/text.py           Claim and token utilities
data/benchmark.jsonl        Deterministic baseline dataset
data/hybrid_benchmark.jsonl Ollama-backed semantic/NLI dataset
tests/                      Unit and API tests
.github/workflows/ci.yml    Automated quality gate
```

## Roadmap

1. Dedicated DeBERTa NLI and sentence-transformer provider adapters.
2. Prompt/model experiment tracking in SQLite.
3. LLM-as-judge rubric with calibration against human labels.
4. RAG retrieval metrics, citation validation, and tool-call evaluation.
5. Web dashboard for run comparison and failure analysis.
6. FoodPilot integration as the first external evaluated application.

## Scope and limitations

The baseline scorer is intentionally lexical and explainable. Hybrid mode adds
semantic retrieval and a local LLM acting under a strict three-label NLI schema;
it is not presented as a fine-tuned NLI classifier. A dedicated classifier and
larger human-labeled benchmarks are planned. The README separates implemented
behavior from roadmap work so quality claims remain reproducible.
