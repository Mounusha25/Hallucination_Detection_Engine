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

Milestone 3 adds a persistent experiment layer:

- automatic SQLite tracking for benchmark runs;
- named experiments with dataset, mode, threshold, metrics, and case-level results;
- run history, detail, and comparison APIs;
- metric deltas against a selected baseline run;
- a responsive browser dashboard for quality and latency monitoring.

Milestone 4 adds calibrated, multi-dimensional judging:

- correctness, relevance, completeness, and faithfulness rubrics;
- strict structured output from a local `qwen3:4b-instruct` judge;
- versioned human labels for repeatable calibration;
- mean absolute error and tolerance-based agreement by dimension;
- pass/fail confusion matrices, precision, recall, and F1;
- case-level judge-versus-human diagnostics saved as experiment runs.

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

Evaluate response quality across a rubric:

```bash
curl -X POST http://127.0.0.1:8000/evaluate/rubric \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of Australia?",
    "response": "Canberra is the capital of Australia.",
    "reference_answer": "Canberra is the capital of Australia.",
    "evidence": [{"id": "atlas-1", "text": "Canberra is Australia’s capital city."}]
  }'
```

The result contains a normalized score, pass/fail decision, and concise
rationale for each dimension plus an overall score. The model runs locally
through Ollama; no response data is sent to an external API.

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
  --name baseline-v1 \
  --fail-under 0.80 \
  --output results/latest.json
```

The command reports accuracy, precision, recall, F1, and median latency, then
records the experiment in `data/experiments.db`. Pass `--no-record` for an
ephemeral CI run. It returns a non-zero exit status when accuracy drops below
the quality threshold, making it suitable for deployment gates.

Run the separate local-model benchmark after starting Ollama:

```bash
python -m src.evaluate \
  --dataset data/hybrid_benchmark.jsonl \
  --name hybrid-qwen3-bge-m3 \
  --fail-under 0.80 \
  --output results/hybrid-latest.json
```

Calibrate the rubric judge against the versioned human labels:

```bash
python -m src.calibrate \
  --dataset data/rubric_calibration.jsonl \
  --name rubric-qwen3-v1 \
  --tolerance 0.25 \
  --fail-under 0.60 \
  --output results/rubric-calibration.json
```

This reports mean absolute error, agreement within the selected tolerance,
per-dimension calibration, a confusion matrix, classification metrics, and
median latency. It exits non-zero when agreement falls below the quality gate.

## Experiment dashboard

After recording one or more benchmarks and starting the API, open
<http://127.0.0.1:8000/dashboard>. The dashboard displays the latest quality
and latency metrics and a searchable history source through these APIs:

```text
GET /runs?limit=50
GET /runs/{run_id}
GET /compare?run_id={baseline_id}&run_id={candidate_id}
```

The first ID passed to `/compare` is treated as the baseline. The response
contains accuracy, precision, recall, F1, and latency deltas for every candidate.
Set `HALLU_DB_PATH` to use a database outside the default `data/experiments.db`.

## Development

```bash
ruff check .
pytest
python -m src.evaluate --fail-under 0.80 --no-record
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
src/calibrate.py            Human-label calibration runner
src/experiments.py          SQLite experiment tracking and comparisons
src/rubrics.py              Multi-dimension rubric orchestration
src/models/schemas.py       Validated contracts
src/retrieval/hybrid.py     BM25 evidence retrieval
src/scoring/groundedness.py Explainable scoring
src/providers/ollama.py     Local embedding and NLI provider
src/utils/text.py           Claim and token utilities
data/benchmark.jsonl        Deterministic baseline dataset
data/hybrid_benchmark.jsonl Ollama-backed semantic/NLI dataset
data/rubric_calibration.jsonl Versioned human rubric labels
tests/                      Unit and API tests
app/dashboard.html          Browser experiment dashboard
.github/workflows/ci.yml    Automated quality gate
```

## Roadmap

1. Dedicated DeBERTa NLI and sentence-transformer provider adapters.
2. RAG retrieval metrics, citation validation, and tool-call evaluation.
3. FoodPilot integration as the first external evaluated application.

## Scope and limitations

The baseline scorer is intentionally lexical and explainable. Hybrid mode adds
semantic retrieval and a local LLM acting under a strict three-label NLI schema;
it is not presented as a fine-tuned NLI classifier. A dedicated classifier and
larger human-labeled benchmarks are planned. The README separates implemented
behavior from roadmap work so quality claims remain reproducible.
