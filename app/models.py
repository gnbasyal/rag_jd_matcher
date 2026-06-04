from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ── JD ingestion models ──────────────────────────────────────────────────────

class JDMetadata(BaseModel):
    """Structured metadata extracted from a raw JD by the LLM tagger."""
    title: str
    company: str
    seniority: Literal["junior", "mid", "senior", "lead"]
    required_skills: list[str]
    tech_stack: list[str]


class JDChunk(BaseModel):
    jd_id: str
    chunk_id: str
    chunk_type: Literal["summary", "responsibilities", "requirements", "benefits"]
    text: str
    metadata: dict


class JDCandidate(BaseModel):
    jd_id: str
    title: str
    company: str
    full_text: str
    retrieval_score: float
    rerank_score: float | None = None


# ── CV models ────────────────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    title: str
    company: str
    years: float
    description: str


class Education(BaseModel):
    degree: str
    institution: str
    year: int | None = None


class CVProfile(BaseModel):
    name: str
    total_years_experience: float
    seniority: Literal["junior", "mid", "senior", "lead"]
    skills: list[str]
    tech_stack: list[str]
    roles: list[WorkExperience]
    education: list[Education]
    domains: list[str]
    raw_text: str


# ── Query builder models ─────────────────────────────────────────────────────

class RetrievalQuery(BaseModel):
    dense_query: str
    bm25_query: str
    filters: dict


# ── Match / explanation models ───────────────────────────────────────────────

class MatchExplanation(BaseModel):
    summary: str
    matching_signals: list[str]
    potential_gaps: list[str]
    seniority_fit: Literal["under", "match", "over"]


class MatchResult(BaseModel):
    rank: int
    jd_id: str
    title: str
    company: str
    rerank_score: float
    explanation: MatchExplanation
