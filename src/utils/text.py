"""Small deterministic text utilities shared by retrieval and scoring."""

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "will", "with",
}
NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def _normalize_token(token: str) -> str:
    token = NUMBER_WORDS.get(token, token)
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is", "us")):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = [
        _normalize_token(token.lower().replace("’", "'"))
        for token in TOKEN_RE.findall(text)
    ]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def split_claims(text: str) -> list[str]:
    """Split a response into non-empty sentence-level claims."""
    pieces = SENTENCE_BOUNDARY_RE.split(text.strip())
    return [piece.strip() for piece in pieces if tokenize(piece)]


def token_f1(left: str, right: str) -> float:
    left_tokens = Counter(tokenize(left))
    right_tokens = Counter(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_tokens.values())
    recall = overlap / sum(right_tokens.values())
    return 2 * precision * recall / (precision + recall)


def claim_coverage(claim: str, evidence: str) -> float:
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & set(tokenize(evidence))) / len(claim_tokens)


def has_negation(text: str) -> bool:
    return bool(set(tokenize(text, remove_stopwords=False)) & NEGATIONS)
