# CV–JD Matching System: Technical Specification

**Version:** 1.0  
**Status:** Draft  
**Last updated:** April 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [System Overview](#3-system-overview)
4. [Technical Workflow](#4-technical-workflow)
5. [Module Specifications](#5-module-specifications)
6. [Data Models](#6-data-models)
7. [Tech Stack](#7-tech-stack)
8. [Retrieval Strategy](#8-retrieval-strategy)
9. [Cross-Encoder Reranking](#9-cross-encoder-reranking)
10. [LLM Explanation Layer](#10-llm-explanation-layer)
11. [Future Enhancements](#11-future-enhancements)
12. [Open Questions](#12-open-questions)

---

## 1. Problem Statement

Recruiters and candidates face a mutual discovery problem. A database of job descriptions (JDs) exists, and when a candidate submits a CV, the goal is to surface the most relevant JDs from that database — along with a clear, human-readable explanation of *why* each JD is a good match.

Simple keyword search fails here. A CV that says "backend engineering" should match a JD that says "server-side development." A candidate with "3 years Python" should rank higher for a role requiring "2–4 years Python" than one requiring "10+ years Java." These are semantic and contextual judgements, not keyword hits.

This system solves that using a Retrieval-Augmented Generation (RAG) pipeline: semantic retrieval narrows the JD database to a shortlist of candidates, a cross-encoder reranker precisely scores each shortlisted JD against the CV, and an LLM generates a structured explanation for each match.

### Core user flow

> A user uploads their CV. The system returns the top 5 matching job descriptions from the database, each with a detailed explanation of what makes it a good match.

---

## 2. Goals and Non-Goals

### Goals

- Accept a CV in PDF, DOCX, or plain text format
- Parse the CV into a structured candidate profile
- Retrieve the top-matching JDs from a maintained database
- Rerank candidates using a cross-encoder for precision
- Generate a structured, human-readable match explanation per JD
- Maintain a JD database that can be updated independently of the query pipeline

### Non-Goals (v1)

- Reverse search (JD → ranked CVs) — planned for v2
- Fine-tuning the cross-encoder on domain-specific data — planned for v2
- User authentication, multi-tenancy, or role-based access
- Real-time JD scraping or ATS integration
- CV editing or improvement suggestions

---

## 3. System Overview

The system consists of two independent pipelines:

### 3.1 Offline pipeline — JD ingestion

Runs once at setup, then periodically as new JDs are added. Parses, chunks, embeds, and indexes all job descriptions into a vector store.

```
JD documents → Parser → Chunker → Metadata tagger → Embedder → Vector store
```

### 3.2 Online pipeline — CV query

Runs on every CV submission. Parses the CV, builds a retrieval query, fetches candidate JDs, reranks them, and generates explanations.

```
CV upload → CV parser → Query builder → Hybrid retriever → Reranker → LLM explainer → Ranked results
```

These two pipelines share the vector store as the integration point. They are otherwise fully decoupled — the JD database can be updated without touching the query logic.

---

## 4. Technical Workflow

### Step 1 — JD ingestion (offline)

1. Raw JD files (plain `.txt`) are loaded from a source directory. *(v1 scope: TXT only. PDF/HTML support and Docling/unstructured integration deferred to v2.)*
2. Text is normalised (excess whitespace collapsed); no semantic transformation is applied.
3. The text is split into fixed-size chunks using `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)`. *(v1 scope: section-aware splitting deferred to v2.)*
4. Each chunk is tagged with LLM-extracted metadata: `jd_id`, `title`, `company`, `seniority`, `required_skills`, `tech_stack`, `chunk_type`, `date_added`.
5. Chunks are embedded using a sentence transformer model and upserted into the vector store.
6. Ingestion is triggered via CLI (`python -m app.jd_ingestion --api-key ... --provider ...`), not via an API endpoint.

### Step 2 — CV upload and parsing (online)

1. User uploads a CV file (PDF, DOCX, or `.txt`).
2. Raw text is extracted using `PyMuPDF` (PDF) or `python-docx` (DOCX).
3. Extracted text is sent to an LLM with a structured output prompt.
4. The LLM returns a `CVProfile` object: name, years of experience, skills list, work history, education, domain, seniority level.

### Step 3 — Query building

From the structured `CVProfile`, two query representations are constructed:

- **Dense query string:** A natural language summary of the candidate, e.g. `"4 years backend engineering, Python, FastAPI, PostgreSQL, Docker, fintech, mid-level"`. This is embedded for vector similarity search.
- **BM25 keyword string:** A flat bag-of-skills string, e.g. `"Python FastAPI PostgreSQL Docker AWS backend engineer"`. Used for sparse keyword search.
- **Metadata filters:** Hard constraints passed to the vector store, e.g. `seniority IN [mid, senior]`, to pre-filter before ANN search.

### Step 4 — Hybrid retrieval

Two retrievers run in parallel and their results are merged:

- **Dense retriever:** Embeds the query string and performs approximate nearest-neighbour (ANN) search against the vector store. Captures semantic similarity.
- **BM25 retriever:** Performs keyword frequency matching against the JD corpus. Captures exact skill/technology matches.

Results are merged using a weighted ensemble (default: 60% dense / 40% sparse). The merged list is deduplicated and the top 20 JDs are returned.

### Step 5 — Cross-encoder reranking

The top 20 JD candidates are scored by a cross-encoder model. Each JD is paired with the full CV text as:

```
[CV text] [SEP] [JD text]
```

The cross-encoder reads both documents jointly — tokens in the CV can attend to tokens in the JD and vice versa. It outputs a single relevance score per pair. The 20 pairs are scored and sorted descending. The top 5 are returned.

This stage is slower than retrieval but runs on only 20 candidates, making it practical. It is significantly more precise than vector similarity because it performs joint contextual reasoning over both documents.

### Step 6 — LLM match explanation

For each of the top 5 reranked JDs, an LLM call is made with a structured prompt. The prompt provides the parsed `CVProfile` and the JD text, and asks for a structured `MatchResult`. The 5 calls are executed in parallel using `asyncio.gather`.

### Step 7 — Response

The orchestrator returns a list of 5 `MatchResult` objects, ordered by match quality, ready to be serialised and returned via the API.

---

## 5. Module Specifications

### Module 1 — `jd_ingestion.py`

**Responsibility:** Offline pipeline. Load, parse, chunk, embed, and index all JDs.

**Key functions:**
- `load_jd_files(source_dir: str) -> list[str]` — loads raw JD documents from disk or API
- `parse_jd(raw_text: str) -> str` — cleans and normalises raw JD text
- `chunk_jd(text: str, jd_id: str) -> list[JDChunk]` — splits into section-aware chunks with metadata
- `embed_and_index(chunks: list[JDChunk]) -> None` — embeds and upserts to vector store

**Chunking strategy (v1):** `RecursiveCharacterTextSplitter` with `chunk_size=512, chunk_overlap=64`. Section-aware splitting (by heading patterns such as "Responsibilities", "Requirements") is deferred to v2. Each chunk carries its parent `jd_id` so full JD text can be reconstructed at retrieval time.

**Metadata schema per chunk:**

| Field | Type | Description |
|---|---|---|
| `jd_id` | `str` | Unique identifier for the parent JD |
| `title` | `str` | Job title |
| `company` | `str` | Company name |
| `seniority` | `str` | junior / mid / senior / lead |
| `required_skills` | `list[str]` | Extracted required skills |
| `tech_stack` | `list[str]` | Technologies explicitly mentioned |
| `chunk_type` | `str` | summary / responsibilities / requirements / benefits |
| `date_added` | `str` | ISO 8601 timestamp |

---

### Module 2 — `cv_parser.py`

**Responsibility:** Extract text from a CV file and return a structured `CVProfile`.

**Key functions:**
- `extract_text(file_bytes: bytes, filename: str) -> str` — extracts raw text from PDF, DOCX, or TXT
- `parse_cv(raw_text: str, llm) -> CVProfile` — calls LLM with structured output prompt; the LLM instance is caller-supplied (API key is not read from environment)

**LLM prompt contract:** The prompt instructs the LLM to return a JSON object matching the `CVProfile` schema. Uses `with_structured_output()` in LangChain with a Pydantic model to enforce the schema. Includes a retry chain for malformed outputs.

**Libraries:** `PyMuPDF` for PDF, `python-docx` for DOCX, LangChain for structured LLM output.

---

### Module 3 — `query_builder.py`

**Responsibility:** Convert a `CVProfile` into retrieval-ready query representations.

**Key functions:**
- `build_dense_query(profile: CVProfile) -> str` — constructs a natural language summary string for embedding
- `build_bm25_query(profile: CVProfile) -> str` — constructs a keyword string for sparse search
- `build_filters(profile: CVProfile) -> dict` — constructs metadata filter dict for vector store pre-filtering

**Design note:** This module is kept separate from the retriever so query construction strategy can be iterated independently. A/B testing different query formulations requires only changes here.

---

### Module 4 — `retriever.py`

**Responsibility:** Perform hybrid retrieval and return top-20 candidate JDs.

**Key functions:**
- `retrieve(query: RetrievalQuery, k: int = 20) -> list[JDCandidate]` — runs ensemble retrieval and returns merged, deduplicated results

**Implementation:** Uses LangChain's `EnsembleRetriever` wrapping a `Chroma` (dev) or `Weaviate` (prod) vector retriever and a `BM25Retriever`. Ensemble weights are configurable (default `[0.6, 0.4]`). Metadata filters from the query builder are passed to the vector retriever's `search_kwargs`.

**This module owns only retrieval.** No scoring, no LLM calls, no business logic.

---

### Module 5 — `reranker.py`

**Responsibility:** Precisely score and rerank the top-20 retrieved JDs against the CV, returning top-5.

**Key functions:**
- `rerank(cv_text: str, candidates: list[JDCandidate], top_k: int = 5) -> list[JDCandidate]` — scores all pairs and returns sorted top-k

**Implementation:** Uses `sentence-transformers` `CrossEncoder` class. Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Input pairs are formatted as `(cv_text, jd_full_text)`. Scores are returned as floats; candidates are sorted descending and top-k are returned with scores attached.

**Note on model choice:** The default model was trained on web search relevance (MS MARCO), not HR data. It will generalise reasonably well for v1. Fine-tuning on labelled CV-JD pairs is planned for v2 and will significantly improve domain-specific accuracy.

---

### Module 6 — `explainer.py`

**Responsibility:** For each top-5 JD, call an LLM to generate a structured match explanation.

**Key functions:**
- `explain(cv_profile: CVProfile, jd: JDCandidate) -> MatchResult` — generates explanation for one pair
- `explain_all(cv_profile: CVProfile, jds: list[JDCandidate]) -> list[MatchResult]` — runs all 5 calls in parallel

**Prompt contract:** The prompt provides the structured `CVProfile` (not raw CV text) and the full JD text. It instructs the LLM to respond with a JSON object matching the `MatchExplanation` schema. Running on structured profile data rather than raw CV text reduces hallucination and produces more consistent output.

**Parallelism:** `explain_all` uses `asyncio.gather` to run all 5 LLM calls concurrently. This is the highest-latency step; parallelism cuts wall-clock time by ~5x.

---

### Module 7 — `orchestrator.py`

**Responsibility:** Top-level entry point that wires all modules together.

**Key functions:**
- `match_cv_to_jds(cv_path: str, top_k: int = 5) -> list[MatchResult]`

**Flow:**
```python
async def match_cv_to_jds(cv_path: str, top_k: int = 5) -> list[MatchResult]:
    raw_text = cv_parser.extract_text(cv_path)
    profile  = await cv_parser.parse_cv(raw_text)
    query    = query_builder.build(profile)
    candidates = retriever.retrieve(query, k=20)
    ranked   = reranker.rerank(raw_text, candidates, top_k=top_k)
    results  = await explainer.explain_all(profile, ranked)
    return results
```

**This is the only module the API layer calls.** All other modules are internal.

---

### Module 8 — `api/routes.py`

**Responsibility:** Thin FastAPI wrapper. Accepts CV upload, returns match results as JSON.

**Endpoints (v1 as implemented):**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check |
| `GET` | `/jds` | Lists JD filenames from the configured source directory |
| `GET` | `/jds/download` | Streams the JD directory as a zip file |
| `POST` | `/parse-cv` | Parses CV only, returns `CVProfile` |
| `POST` | `/match` | Full pipeline — returns profile + top-5 matches with explanations |

**Note:** The originally specced `POST /jd/ingest` endpoint was not implemented. JD ingestion is triggered via CLI only (`python -m app.jd_ingestion`).

**Request (`/match`, `/parse-cv`):** `multipart/form-data` with `cv_file` (PDF, DOCX, or TXT), `api_key` (required — user-supplied at runtime, never stored), and `llm_provider` (`openai` or `anthropic`).  
**Response:** `application/json` — full match payload including profile, retrieval debug info, and ranked results.

**No business logic lives here.** Validation, error handling, and response serialisation only.

---

## 6. Data Models

### `CVProfile`
```python
class CVProfile(BaseModel):
    name: str
    total_years_experience: float
    seniority: Literal["junior", "mid", "senior", "lead"]
    skills: list[str]
    tech_stack: list[str]
    roles: list[WorkExperience]
    education: list[Education]
    domains: list[str]          # e.g. ["fintech", "e-commerce"]
    raw_text: str               # original extracted text, used by reranker
```

### `WorkExperience`
```python
class WorkExperience(BaseModel):
    title: str
    company: str
    years: float
    description: str
```

### `JDChunk`
```python
class JDChunk(BaseModel):
    jd_id: str
    chunk_id: str
    chunk_type: str
    text: str
    metadata: dict
```

### `JDCandidate`
```python
class JDCandidate(BaseModel):
    jd_id: str
    title: str
    company: str
    full_text: str
    retrieval_score: float      # from hybrid retrieval
    rerank_score: float | None  # populated after reranking
```

### `MatchExplanation`
```python
class MatchExplanation(BaseModel):
    summary: str                        # 2-sentence overall assessment
    matching_signals: list[str]         # e.g. ["5yr Python matches 4yr requirement"]
    potential_gaps: list[str]           # e.g. ["No Kubernetes experience mentioned"]
    seniority_fit: Literal["under", "match", "over"]
```

### `MatchResult`
```python
class MatchResult(BaseModel):
    rank: int
    jd_id: str
    title: str
    company: str
    rerank_score: float
    explanation: MatchExplanation
```

---

## 7. Tech Stack

| Layer | Library / Tool | Notes |
|---|---|---|
| Orchestration | LangChain | Chains, retrievers, structured output |
| LLM | OpenAI GPT-4o or Claude 3.5 Sonnet | Configurable via env var |
| Embeddings | `text-embedding-3-large` or `BAAI/bge-large-en-v1.5` | Swap via config |
| Vector store (dev) | ChromaDB | In-process, zero setup |
| Vector store (prod) | Weaviate or Pinecone | Managed, supports metadata filters |
| Sparse retrieval | `rank_bm25` | BM25 over JD corpus |
| Reranker | `sentence-transformers` CrossEncoder | `ms-marco-MiniLM-L-6-v2` default |
| PDF parsing | `PyMuPDF` (`fitz`) | |
| DOCX parsing | `python-docx` | |
| API | FastAPI + Uvicorn | |
| Async | `asyncio` | For parallel LLM calls in explainer |
| Validation | Pydantic v2 | All data models |

---

## 8. Retrieval Strategy

### Why hybrid retrieval

Neither dense nor sparse retrieval alone is sufficient for this domain:

- **Dense (vector) retrieval** handles semantic similarity well — "server-side Python development" matches "backend engineering" — but can miss exact keyword matches, especially for specific technology names, certifications, and version numbers.
- **Sparse (BM25) retrieval** handles exact matches well but misses semantic equivalence and fails on vocabulary mismatch.

Combining both in a weighted ensemble captures the strengths of each.

### Ensemble weights

Default: `0.6` dense / `0.4` sparse. These are tunable. In domains with highly specific technology requirements (e.g. embedded systems, niche frameworks), increase the sparse weight. In domains with more generalist roles, increase the dense weight.

### Pre-filtering

Metadata filters are applied inside the vector store before ANN search, not after. This avoids retrieving irrelevant documents just to discard them post-hoc. Filters are constructed by the query builder from the candidate's `seniority` field and `domains` list.

---

## 9. Cross-Encoder Reranking

### How it works

A cross-encoder is a BERT-style transformer that takes a concatenated pair of texts as input:

```
[CLS] <CV text> [SEP] <JD text> [SEP]
```

All tokens in both documents can attend to all other tokens. The `[CLS]` token's final hidden state is passed through a linear classification head to produce a single relevance score. Unlike bi-encoders (used in retrieval), there is no independent encoding — the model reasons over both documents jointly.

### Why it's more accurate than vector similarity

Bi-encoders compress each document into a fixed-size vector before any comparison occurs. Information relevant to the specific pairing is inevitably lost. Cross-encoders have full context at scoring time: they can detect that "5 years required" mismatches "2 years experience", that skill overlaps are specific rather than generic, and that domain terminology is consistent.

### Why it's only used for reranking

Cross-encoders require a forward pass per pair. Running one against a 10,000-JD database on every query is computationally infeasible. The two-stage design — retrieve broadly with a fast bi-encoder, rerank precisely with a slow cross-encoder — gives the best of both.

### Default model and fine-tuning plan

**v1:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (trained on web search relevance). Will generalise reasonably to CV-JD matching but is not domain-optimised.

**v2:** Fine-tune on labelled CV-JD pairs. Training data format:
```json
{"cv": "...", "jd": "...", "label": 1}
{"cv": "...", "jd": "...", "label": 0}
```
Even 5,000–10,000 pairs will produce a meaningful improvement. Data can be generated synthetically via LLM or collected from recruiter feedback over time.

---

## 10. LLM Explanation Layer

### Purpose

The reranker produces a score, not an explanation. The LLM explanation layer converts the score into a structured, human-readable rationale — the "why" behind each match.

### Prompt design principles

- Provide the structured `CVProfile` object, not raw CV text. Structured input produces more consistent and factual output.
- Ask for a fixed JSON schema via `with_structured_output()`. Do not rely on free-form text parsing.
- Separate concerns: the reranker ranks, the LLM explains. The LLM does not re-score.

### Prompt template (abbreviated)

```
You are a recruitment analyst.

Given the following candidate profile:
{cv_profile_json}

And the following job description:
{jd_text}

Explain why this job description is or is not a strong match for this candidate.
Respond ONLY with a JSON object matching this schema:
{
  "summary": "<2 sentence overall assessment>",
  "matching_signals": ["<specific signal 1>", ...],
  "potential_gaps": ["<gap 1>", ...],
  "seniority_fit": "under" | "match" | "over"
}
```

### Parallelism

All 5 explanation calls are fired concurrently with `asyncio.gather`. Since each call is independent, there is no correctness reason to serialise them. This reduces the explanation step from ~15s to ~3–4s.

---

## 11. Future Enhancements

### v2 features

| Feature | Description |
|---|---|
| Reverse search | Given a JD, return ranked CVs from a candidate database |
| Cross-encoder fine-tuning | Train on labelled CV-JD pairs for domain-specific accuracy |
| Match score breakdown | Structured scores per dimension: skills, experience, domain, seniority |
| Skill gap report | Per-match list of required skills the candidate is missing |
| CV improvement suggestions | Given a target JD, suggest CV rewording to improve match score |
| JD normalisation | LLM-based structured extraction of JD fields before indexing |

### v3 features

| Feature | Description |
|---|---|
| Feedback loop | Users label matches as relevant/irrelevant; used to retrain reranker |
| Batch CV processing | Accept folder of CVs, output ranked matrix against JD database |
| JD staleness detection | Flag JDs not updated in >90 days |
| ATS integration | Pull JDs directly from Greenhouse, Lever, or Workday APIs |

---

## 12. Open Questions

| # | Question | Impact | Owner |
|---|---|---|---|
| 1 | Which vector store for production — Weaviate or Pinecone? | Infra, cost | TBD |
| 2 | What is the expected size of the JD database? Affects index choice. | Performance | TBD |
| 3 | Should the CV parser run entirely via LLM, or use rule-based extraction first? | Accuracy, cost | TBD |
| 4 | What LLM provider for the explainer — OpenAI or Anthropic? | Cost, latency | TBD |
| 5 | How will labelled CV-JD pairs be collected for v2 fine-tuning? | v2 accuracy | TBD |
| 6 | Is there a need to support multiple languages in CVs or JDs? | Scope | TBD |
