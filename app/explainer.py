"""
LLM Match Explainer (online pipeline — Step 6)
================================================
For each of the top-5 reranked JDs, generates a structured MatchExplanation
via an LLM call. All 5 calls run concurrently with asyncio.gather.
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.prompts import ChatPromptTemplate

from app.models import JDCandidate, CVProfile, MatchExplanation, MatchResult

# ── Rank → tone mapping ───────────────────────────────────────────────────────

_TONE = {
    1: "highly relevant",
    2: "relevant",
    3: "moderately relevant",
    4: "somewhat relevant",
    5: "somewhat relevant",
}

# ── Rank → signal/gap count ranges (rank-1 value, rank-5 value) ──────────────
# Edit these two constants to tune the counts across all ranks.

_SIGNAL_RANGE: tuple[int, int] = (6, 3)
_GAP_RANGE:    tuple[int, int] = (2, 5)


def _counts(rank: int, n_ranks: int = 5) -> tuple[int, int]:
    t = (rank - 1) / (n_ranks - 1)
    n_sig = round(_SIGNAL_RANGE[0] + t * (_SIGNAL_RANGE[1] - _SIGNAL_RANGE[0]))
    n_gap = round(_GAP_RANGE[0]    + t * (_GAP_RANGE[1]    - _GAP_RANGE[0]))
    return n_sig, n_gap

# ── Prompt ────────────────────────────────────────────────────────────────────

_EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert recruitment analyst. Your job is to explain how well a job description "
        "matches a candidate's profile.\n\n"
        "Rules:\n"
        "- summary: A paragraph of exactly 4-5 sentences. Open with: "
        "'This job is {tone} for the candidate because...'. "
        "The remaining sentences must cover: specific skill/tech alignment, domain fit, "
        "experience level match, and any notable gaps or strengths. Be concrete, not generic.\n"
        "- matching_signals: Return exactly {n_signals} signals — the most important overlaps between the CV and JD. "
        "Reference actual skill names, years of experience, domain terms, and education. "
        "Each signal should be a single precise statement (e.g. '5 years Python matches 4-year requirement').\n"
        "- potential_gaps: Return exactly {n_gaps} gaps — the most important JD requirements the candidate's profile does not clearly satisfy. "
        "If the JD has fewer real gaps than {n_gaps}, return as many as exist (do not fabricate).\n"
        "- seniority_fit: 'under' if the candidate is below the JD's expected level, "
        "'match' if aligned, 'over' if the candidate is overqualified.",
    ),
    (
        "human",
        "Candidate profile (rank {rank}/5, rerank score {rerank_score}):\n"
        "{cv_profile_json}\n\n"
        "Job description:\n"
        "{jd_text}",
    ),
])

# ── Functions ─────────────────────────────────────────────────────────────────

async def explain(cv_profile: CVProfile, jd: JDCandidate, rank: int, llm) -> MatchResult:
    """Generate a structured explanation for one CV–JD pair."""
    chain = _EXPLAINER_PROMPT | llm.with_structured_output(MatchExplanation)
    cv_json = json.dumps(
        cv_profile.model_dump(exclude={"raw_text"}),
        indent=2,
        default=str,
    )
    n_signals, n_gaps = _counts(rank)
    explanation: MatchExplanation = await chain.ainvoke({
        "tone": _TONE.get(rank, "somewhat relevant"),
        "rank": rank,
        "rerank_score": f"{jd.rerank_score:.2%}" if jd.rerank_score is not None else "N/A",
        "cv_profile_json": cv_json,
        "jd_text": jd.full_text,
        "n_signals": n_signals,
        "n_gaps": n_gaps,
    })
    return MatchResult(
        rank=rank,
        jd_id=jd.jd_id,
        title=jd.title,
        company=jd.company,
        rerank_score=jd.rerank_score or 0.0,
        explanation=explanation,
    )


async def explain_all(cv_profile: CVProfile, jds: list[JDCandidate], llm) -> list[MatchResult]:
    """Run all explanation calls concurrently and return results in rank order."""
    return list(await asyncio.gather(*[
        explain(cv_profile, jd, rank=i + 1, llm=llm)
        for i, jd in enumerate(jds)
    ]))
