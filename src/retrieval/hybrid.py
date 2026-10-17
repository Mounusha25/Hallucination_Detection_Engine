"""Dependency-free BM25 retrieval used by the baseline evaluator."""

import math
from collections import Counter
from dataclasses import dataclass

from src.models.schemas import EvidenceDocument
from src.utils.text import tokenize


@dataclass(frozen=True)
class RetrievedDocument:
    document: EvidenceDocument
    score: float


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
            )
            for score, index in selected
        ]
