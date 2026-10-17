"""Transparent claim/evidence scoring for the deterministic baseline."""

from src.utils.text import claim_coverage, has_negation, token_f1


def support_score(claim: str, evidence: str) -> float:
    """Combine symmetric token F1 with claim-token coverage."""
    score = max(token_f1(claim, evidence), 0.9 * claim_coverage(claim, evidence))
    return max(0.0, min(1.0, score))


def is_contradiction(claim: str, evidence: str, minimum_coverage: float = 0.55) -> bool:
    """Flag explicit polarity conflicts when the statements share their subject."""
    return (
        claim_coverage(claim, evidence) >= minimum_coverage
        and has_negation(claim) != has_negation(evidence)
    )
