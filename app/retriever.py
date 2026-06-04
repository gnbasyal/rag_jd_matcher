"""
Hybrid Retriever (online pipeline — Step 4)
============================================
Dense (Chroma ANN) + BM25 → RRF merge → top-20 JDCandidates
"""
from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.models import JDCandidate, RetrievalQuery

# ── Module-level init (runs once on import) ───────────────────────────────────

_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
_vectorstore = Chroma(
    collection_name=settings.jd_collection_name,
    embedding_function=_embeddings,
    persist_directory=settings.chroma_persist_dir,
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_corpus(filters: dict) -> list[Document]:
    """Fetch all ChromaDB docs matching the seniority filter (no embedding needed)."""
    result = _vectorstore._collection.get(
        where=filters,
        include=["documents", "metadatas"],
    )
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]


def _reconstruct_jds(docs: list[Document]) -> dict[str, dict]:
    """
    Group chunks by jd_id and concatenate into full JD text.
    Returns {jd_id: {title, company, seniority, full_text}}.
    """
    jds: dict[str, dict] = {}
    for doc in docs:
        meta = doc.metadata
        jd_id = meta["jd_id"]
        if jd_id not in jds:
            jds[jd_id] = {
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "seniority": meta.get("seniority", ""),
                "chunks": [],
            }
        jds[jd_id]["chunks"].append(doc.page_content)

    for data in jds.values():
        data["full_text"] = "\n".join(data.pop("chunks"))

    return jds


_HitRow = tuple[str, str, str, float]  # (jd_id, title, company, score)


def _dedup(hits: list[_HitRow]) -> list[_HitRow]:
    """Keep the highest-scoring chunk per jd_id."""
    best: dict[str, _HitRow] = {}
    for row in hits:
        jd_id = row[0]
        if jd_id not in best or row[3] > best[jd_id][3]:
            best[jd_id] = row
    return list(best.values())


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(
    query: RetrievalQuery, k: int = 20
) -> tuple[list[dict], list[dict], list[JDCandidate]]:
    """
    Hybrid retrieval over the JD vector store.

    Returns:
        dense_hits  — top results from Chroma ANN, ranked by cosine similarity
        bm25_hits   — top results from BM25 keyword search, ranked by TF-IDF score
        merged      — RRF-merged top-k JDCandidates (60% dense / 40% BM25)
    """
    fetch_k = k * 2  # over-fetch before dedup

    # ── Dense retrieval ───────────────────────────────────────────────────────
    dense_raw = _vectorstore.similarity_search_with_relevance_scores(
        query.dense_query, k=fetch_k, filter=query.filters
    )
    dense_rows: list[_HitRow] = [
        (
            doc.metadata["jd_id"],
            doc.metadata.get("title", ""),
            doc.metadata.get("company", ""),
            score,
        )
        for doc, score in dense_raw
    ]
    dense_rows = sorted(_dedup(dense_rows), key=lambda r: r[3], reverse=True)

    dense_hits = [
        {"rank": i + 1, "jd_id": r[0], "title": r[1], "company": r[2], "score": round(r[3], 4)}
        for i, r in enumerate(dense_rows)
    ]

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    corpus = _fetch_corpus(query.filters)
    jd_map = _reconstruct_jds(corpus)

    if corpus:
        bm25 = BM25Retriever.from_documents(corpus, k=fetch_k)
        bm25_docs = bm25.invoke(query.bm25_query)
        # BM25Retriever doesn't expose raw scores — use rank-based proxy
        bm25_rows: list[_HitRow] = _dedup([
            (
                doc.metadata["jd_id"],
                doc.metadata.get("title", ""),
                doc.metadata.get("company", ""),
                0.0,
            )
            for doc in bm25_docs
        ])
        # Assign descending rank-based scores after dedup
        bm25_rows = [
            (jd_id, title, company, round(1.0 - i / max(len(bm25_rows), 1), 4))
            for i, (jd_id, title, company, _) in enumerate(bm25_rows)
        ]
    else:
        bm25_rows = []

    bm25_hits = [
        {"rank": i + 1, "jd_id": r[0], "title": r[1], "company": r[2], "score": r[3]}
        for i, r in enumerate(bm25_rows)
    ]

    # ── RRF merge (60% dense / 40% BM25) ─────────────────────────────────────
    dense_rank = {r[0]: i + 1 for i, r in enumerate(dense_rows)}
    bm25_rank  = {r[0]: i + 1 for i, r in enumerate(bm25_rows)}

    all_ids = set(dense_rank) | set(bm25_rank)
    RRF_K   = 60
    fallback = len(all_ids) + RRF_K  # penalty rank for docs missing from one retriever

    rrf: dict[str, float] = {
        jd_id: (
            0.6 * (1 / (dense_rank.get(jd_id, fallback) + RRF_K))
            + 0.4 * (1 / (bm25_rank.get(jd_id, fallback)  + RRF_K))
        )
        for jd_id in all_ids
    }

    merged_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]

    merged: list[JDCandidate] = [
        JDCandidate(
            jd_id=jd_id,
            title=jd_map.get(jd_id, {}).get("title", ""),
            company=jd_map.get(jd_id, {}).get("company", ""),
            full_text=jd_map.get(jd_id, {}).get("full_text", ""),
            retrieval_score=round(rrf[jd_id], 6),
        )
        for jd_id in merged_ids
    ]

    return dense_hits, bm25_hits, merged
