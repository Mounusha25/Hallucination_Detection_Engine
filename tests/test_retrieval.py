from src.models.schemas import EvidenceDocument
from src.retrieval.hybrid import BM25Retriever, HybridRetriever


class FakeEmbeddings:
    name = "fake"
    model = "semantic-test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if any(word in text.lower() for word in ("doctor", "physician"))
            else [0.0, 1.0]
            for text in texts
        ]


def test_bm25_returns_relevant_document_first() -> None:
    retriever = BM25Retriever(
        [
            EvidenceDocument(id="space", text="Apollo 11 landed on the Moon in 1969."),
            EvidenceDocument(id="food", text="A taco is a popular Mexican dish."),
        ]
    )

    results = retriever.search("When did Apollo 11 land on the Moon?", top_k=2)

    assert results[0].document.id == "space"
    assert results[0].score == 1.0


def test_hybrid_retrieval_finds_semantic_paraphrase() -> None:
    retriever = HybridRetriever(
        [
            EvidenceDocument(id="medical", text="A physician treats illnesses."),
            EvidenceDocument(id="ocean", text="Whales live in the ocean."),
        ],
        FakeEmbeddings(),
    )

    results = retriever.search("A doctor cares for sick people.")

    assert results[0].document.id == "medical"
    assert results[0].semantic_score == 1.0
