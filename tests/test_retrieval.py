from src.models.schemas import EvidenceDocument
from src.retrieval.hybrid import BM25Retriever


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
