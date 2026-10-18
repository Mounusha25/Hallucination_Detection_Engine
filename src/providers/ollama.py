"""Ollama adapters for local embeddings and NLI-style judging."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.providers.base import NLIResult, ProviderError


class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str | None = None, timeout: float = 90.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.timeout = timeout

    def post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProviderError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc


class OllamaEmbeddingProvider:
    name = "ollama"

    def __init__(self, model: str = "bge-m3", client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self.client.post("/api/embed", {"model": self.model, "input": texts})
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ProviderError("Ollama returned an invalid embedding response")
        return embeddings


class OllamaNLIProvider:
    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        client: OllamaClient | None = None,
    ):
        self.model = model
        self.client = client or OllamaClient()

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        prompt = (
            "Classify the relationship between the evidence premise and claim hypothesis. "
            "Use entailment only when the evidence supports the complete claim, contradiction "
            "when it conflicts with the claim, and neutral when evidence is insufficient. "
            "Return JSON with label and confidence from 0 to 1.\n\n"
            f"Premise: {premise}\nHypothesis: {hypothesis}"
        )
        payload = self.client.post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict natural-language-inference classifier.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "format": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["entailment", "contradiction", "neutral"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["label", "confidence"],
                },
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 80},
            },
        )
        try:
            content = payload["message"]["content"]
            result = json.loads(content)
            label = result["label"]
            confidence = float(result["confidence"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError("Ollama returned an invalid NLI response") from exc
        if label not in {"entailment", "contradiction", "neutral"}:
            raise ProviderError(f"Ollama returned an unknown NLI label: {label}")
        return NLIResult(label=label, confidence=max(0.0, min(1.0, confidence)))
