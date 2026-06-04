"""
JD Ingestion Pipeline (offline)
================================
Load → Parse → Chunk → Tag metadata → Embed → Index into ChromaDB
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.llm import build_llm
from app.models import JDChunk, JDMetadata


# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_jd_files(source_dir: str) -> list[tuple[str, str]]:
    """
    Load all .txt JD files from source_dir.
    Returns list of (jd_id, raw_text) where jd_id is the filename stem prefix
    (e.g. 'JD03' from 'JD03_devops_engineer.txt').
    """
    results = []
    for path in sorted(Path(source_dir).glob("*.txt")):
        jd_id = path.stem.split("_")[0].upper()
        results.append((jd_id, path.read_text(encoding="utf-8")))
    return results


def parse_jd(raw_text: str) -> str:
    """Normalise whitespace; no other transformation needed for plain-text JDs."""
    text = raw_text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



_METADATA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an HR data analyst. Extract structured metadata from the job description below.\n"
        "Rules:\n"
        "- seniority: 'junior' (0-2 yr), 'mid' (2-5 yr), 'senior' (5-10 yr), 'lead' (10+ yr or people-management).\n"
        "- tech_stack: only hard technical tools, languages, frameworks, and platforms explicitly mentioned.\n"
        "  Leave empty [] for non-technical roles.\n"
        "- required_skills: ALL skills and competencies required (soft + hard, excluding tech_stack duplication).\n"
        "- company: use '[Company]' if not explicitly named.",
    ),
    ("human", "{jd_text}"),
])


def tag_metadata(jd_id: str, text: str, llm) -> JDMetadata:
    """Call LLM to extract structured metadata from a JD."""
    chain = _METADATA_PROMPT | llm.with_structured_output(JDMetadata)
    return chain.invoke({"jd_text": text})


def chunk_jd(text: str, jd_id: str, metadata: JDMetadata) -> list[JDChunk]:
    """Split a JD into fixed-size chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    raw_chunks = splitter.split_text(text)

    base_meta = {
        "jd_id": jd_id,
        "title": metadata.title,
        "company": metadata.company,
        "seniority": metadata.seniority,
        # ChromaDB requires flat scalar metadata — serialise lists as comma strings
        "required_skills": ", ".join(metadata.required_skills),
        "tech_stack": ", ".join(metadata.tech_stack),
        "date_added": datetime.now(timezone.utc).isoformat(),
        "chunk_type": "summary",
    }

    return [
        JDChunk(
            jd_id=jd_id,
            chunk_id=f"{jd_id}_chunk_{i}_{uuid.uuid4().hex[:8]}",
            chunk_type="summary",
            text=chunk,
            metadata=base_meta,
        )
        for i, chunk in enumerate(raw_chunks)
    ]


def embed_and_index(chunks: list[JDChunk], reset: bool = False) -> None:
    """Embed chunks and upsert into ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if reset:
        # Wipe existing collection before re-indexing
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        try:
            client.delete_collection(settings.jd_collection_name)
            print("  Cleared existing collection.")
        except Exception:
            pass

    vectorstore = Chroma(
        collection_name=settings.jd_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    documents = [
        Document(page_content=chunk.text, metadata=chunk.metadata)
        for chunk in chunks
    ]
    ids = [chunk.chunk_id for chunk in chunks]

    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"  Indexed {len(documents)} chunks into '{settings.jd_collection_name}'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ingestion(
    api_key: str,
    provider: str = "openai",
    source_dir: str | None = None,
    reset: bool = True,
) -> None:
    """Full offline ingestion pipeline."""
    llm = build_llm(api_key=api_key, provider=provider)
    src = source_dir or settings.jd_source_dir
    print(f"Loading JDs from: {src}")
    jd_files = load_jd_files(src)
    print(f"Found {len(jd_files)} JDs.\n")

    all_chunks: list[JDChunk] = []

    for jd_id, raw_text in jd_files:
        print(f"[{jd_id}] Tagging metadata...")
        text = parse_jd(raw_text)
        metadata = tag_metadata(jd_id, text, llm=llm)
        chunks = chunk_jd(text, jd_id, metadata)
        print(
            f"  → {metadata.title} | {metadata.company} | {metadata.seniority} | "
            f"{len(chunks)} chunks"
        )
        all_chunks.extend(chunks)

    print(f"\nEmbedding {len(all_chunks)} total chunks...")
    embed_and_index(all_chunks, reset=reset)
    print("\nIngestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run JD ingestion pipeline")
    parser.add_argument("--api-key", required=True, help="LLM API key")
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--source", default=None, help="Path to JD source directory")
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Append to existing collection instead of resetting",
    )
    args = parser.parse_args()
    run_ingestion(
        api_key=args.api_key,
        provider=args.provider,
        source_dir=args.source,
        reset=not args.no_reset,
    )
