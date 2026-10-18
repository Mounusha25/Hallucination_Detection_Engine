"""BM25 baseline and dense/lexical hybrid evidence retrieval."""

import math
from collections import Counter
from dataclasses import dataclass

from src.models.schemas import EvidenceDocument
from src.providers.base import EmbeddingProvider
from src.utils.text import tokenize


@dataclass(frozen=True)
class RetrievedDocument:
    document: EvidenceDocument
    score: float
    lexical_score: float
    semantic_score: float | None = None


class BM25Retriever:
    """Small in-memory BM25 index suitable for per-request evidence packs."""

    def __init__(self, documents: list[EvidenceDocument], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("at least one evidence document is required")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document.text) for document in documents]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens) or 1.0
        self.document_frequency = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def _idf(self, term: str) -> float:
        count = self.document_frequency.get(term, 0)
        total = len(self.documents)
        return math.log(1 + (total - count + 0.5) / (count + 0.5))

    def _score(self, query_tokens: list[str], index: int) -> float:
        length = len(self.tokens[index])
        counts = self.term_counts[index]
        score = 0.0
        for term in set(query_tokens):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self.avg_length
            )
            score += self._idf(term) * frequency * (self.k1 + 1) / denominator
        return score

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        query_tokens = tokenize(query)
        raw = [(self._score(query_tokens, index), index) for index in range(len(self.documents))]
        raw.sort(reverse=True)
        selected = raw[: min(top_k, len(raw))]
        maximum = selected[0][0] if selected else 0.0
        return [
            RetrievedDocument(
                document=self.documents[index],
                score=(score / maximum if maximum > 0 else 0.0),
                lexical_score=(score / maximum if maximum > 0 else 0.0),
            )
            for score, index in selected
        ]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class HybridRetriever:
    """Fuse BM25 with dense cosine similarity for paraphrase-aware retrieval."""

    def __init__(
        self,
        documents: list[EvidenceDocument],
        embedding_provider: EmbeddingProvider,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        if lexical_weight < 0 or semantic_weight < 0 or lexical_weight + semantic_weight <= 0:
            raise ValueError("retrieval weights must be non-negative with a positive sum")
        total = lexical_weight + semantic_weight
        self.lexical_weight = lexical_weight / total
        self.semantic_weight = semantic_weight / total
        self.documents = documents
        self.embedding_provider = embedding_provider
        self.bm25 = BM25Retriever(documents)
        self.document_embeddings = embedding_provider.embed(
            [document.text for document in documents]
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        lexical = {
            result.document.id: result.lexical_score
            for result in self.bm25.search(query, len(self.documents))
        }
        query_embedding = self.embedding_provider.embed([query])[0]
        results = []
        for document, vector in zip(self.documents, self.document_embeddings, strict=True):
            # Embedding cosine can be negative. Rescale to a stable 0..1 metric.
            semantic = (_cosine(query_embedding, vector) + 1) / 2
            lexical_score = lexical.get(document.id, 0.0)
            fused = self.lexical_weight * lexical_score + self.semantic_weight * semantic
            results.append(
                RetrievedDocument(
                    document=document,
                    score=fused,
                    lexical_score=lexical_score,
                    semantic_score=semantic,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[: min(top_k, len(results))]
