"""
Query Builder (online pipeline — Step 3)
=========================================
Converts a CVProfile into three retrieval-ready representations:
  - dense_query  : natural language string for embedding / ANN search
  - bm25_query   : space-separated keywords for sparse BM25 search
  - filters      : ChromaDB metadata pre-filter dict
"""
from __future__ import annotations

from app.models import CVProfile, RetrievalQuery

_SENIORITY_ORDER = ["junior", "mid", "senior", "lead"]


def build_dense_query(profile: CVProfile) -> str:
    """
    Natural language summary of the candidate for semantic embedding.
    e.g. "4 years experience, Backend Engineer, mid, Python, FastAPI, fintech"
    """
    parts: list[str] = [f"{profile.total_years_experience:.1f} years experience"]

    if profile.roles:
        parts.append(profile.roles[0].title)

    parts.append(profile.seniority)

    if profile.tech_stack:
        parts.extend(profile.tech_stack[:8])

    if profile.domains:
        parts.extend(profile.domains)

    return ", ".join(parts)


def build_bm25_query(profile: CVProfile) -> str:
    """
    Flat space-separated keyword string for BM25 sparse retrieval.
    e.g. "Python FastAPI PostgreSQL Docker problem solving backend engineer"
    """
    terms: list[str] = list(profile.tech_stack)
    terms.extend(profile.skills[:6])

    if profile.roles:
        terms.append(profile.roles[0].title)

    return " ".join(terms)


def build_filters(profile: CVProfile) -> dict:
    """
    ChromaDB metadata pre-filter. Expands seniority ±1 level so candidates
    aren't excluded from near-level roles.

    junior → ["junior", "mid"]
    mid    → ["junior", "mid", "senior"]
    senior → ["mid", "senior", "lead"]
    lead   → ["senior", "lead"]
    """
    idx = _SENIORITY_ORDER.index(profile.seniority)
    allowed = _SENIORITY_ORDER[max(0, idx - 1) : idx + 2]
    return {"seniority": {"$in": allowed}}


def build_rerank_text(profile: CVProfile) -> str:
    """
    Structured CV summary for the cross-encoder reranker.
    Richer than dense_query but short enough to fit within the 512-token budget
    alongside the JD text.
    """
    lines: list[str] = [
        f"{profile.seniority.capitalize()} {profile.roles[0].title if profile.roles else 'Professional'}"
        f" | {profile.total_years_experience:.1f} years"
        + (f" | {', '.join(profile.domains)}" if profile.domains else ""),
    ]

    if profile.tech_stack:
        lines.append(f"Tech: {', '.join(profile.tech_stack[:10])}")

    if profile.skills:
        lines.append(f"Skills: {', '.join(profile.skills[:8])}")

    if profile.roles:
        r = profile.roles[0]
        lines.append(f"Recent: {r.title} at {r.company} ({r.years:.0f} yr) — {r.description[:120]}")

    return "\n".join(lines)


def build(profile: CVProfile) -> RetrievalQuery:
    """Return all three query representations in one object."""
    return RetrievalQuery(
        dense_query=build_dense_query(profile),
        bm25_query=build_bm25_query(profile),
        filters=build_filters(profile),
    )
