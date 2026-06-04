"""
Cross-Encoder Reranker (online pipeline — Step 5)
===================================================
Scores each (CV, JD) pair jointly and returns the top-k by relevance.
"""
from __future__ import annotations

import math

from sentence_transformers.cross_encoder import CrossEncoder

from app.models import JDCandidate

# ── Module-level init (runs once on import) ───────────────────────────────────

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Main function ─────────────────────────────────────────────────────────────

def rerank(
    cv_text: str,
    candidates: list[JDCandidate],
    top_k: int = 5,
) -> list[JDCandidate]:
    """
    Score each candidate against the CV using a cross-encoder and return top-k.

    The model reads both texts jointly:
        [CLS] cv_text [SEP] jd_full_text [SEP]

    Top-k selected by raw logit, then softmax applied so scores sum to 1.0.
    A single batched forward pass covers all candidates.
    """
    if not candidates:
        return []

    pairs = [(cv_text, c.full_text) for c in candidates]
    logits = _model.predict(pairs)  # np.ndarray, shape (len(candidates),)

    # Sort by raw logit, take top-k, then apply softmax so scores sum to 1
    ranked = sorted(zip(candidates, logits), key=lambda x: x[1], reverse=True)[:top_k]
    top_candidates, top_logits = zip(*ranked)

    exp_scores = [math.exp(float(s)) for s in top_logits]
    total = sum(exp_scores)
    for candidate, exp_s in zip(top_candidates, exp_scores):
        candidate.rerank_score = round(exp_s / total, 4)

    return list(top_candidates)
