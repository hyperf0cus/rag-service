"""
Retrieval evaluation metrics.

All functions are pure (no I/O) so they can be unit-tested trivially.

Terminology
-----------
relevant_ids : set of ground-truth IDs (chunk_ids or doc_ids).
retrieved_ids: ordered list of retrieved IDs (highest score first).
"""
from __future__ import annotations


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Fraction of relevant items found in the top-k retrieved results.

    Defined as  |relevant ∩ top_k| / |relevant|.
    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Fraction of top-k retrieved results that are relevant.

    Defined as  |relevant ∩ top_k| / k.
    Returns 0.0 when k == 0.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / k


def reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """
    Reciprocal of the rank of the first relevant item (1-indexed).

    Returns 0.0 when no relevant item appears in retrieved_ids.
    """
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    pairs: list[tuple[set[str], list[str]]],
) -> float:
    """
    Mean Reciprocal Rank across multiple (relevant, retrieved) pairs.

    Returns 0.0 for an empty list.
    """
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(rel, ret) for rel, ret in pairs) / len(pairs)


def hit_rate_at_k(
    pairs: list[tuple[set[str], list[str]]],
    k: int,
) -> float:
    """
    Fraction of queries where at least one relevant item appears in top-k.
    """
    if not pairs:
        return 0.0
    hits = sum(1 for rel, ret in pairs if any(r in rel for r in ret[:k]))
    return hits / len(pairs)
