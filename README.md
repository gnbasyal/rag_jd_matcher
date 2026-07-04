# CV–JD Matcher

A RAG-based system that matches a candidate's CV against a database of job descriptions and returns the top 5 most relevant jobs with structured, LLM-generated explanations.

Upload a CV (PDF, DOCX, or TXT) and the system parses it into a structured profile, runs hybrid retrieval (dense semantic search + BM25 keyword matching) over the job database, reranks the top candidates with a cross-encoder, and generates parallel LLM explanations for each match — covering skill alignment, domain fit, potential gaps, and seniority fit.

For a full technical breakdown of every component and the end-to-end workflow, see [docs/technical_overview.md](docs/technical_overview.md).

---

## Tech Stack

- **API & server:** Python 3.11+, FastAPI, Uvicorn
- **LLM integration:** LangChain, OpenAI (`gpt-4.1`) or Anthropic (`claude-sonnet-4-6`)
- **Vector store:** ChromaDB with `all-MiniLM-L6-v2` embeddings (local, no API key)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **Sparse retrieval:** BM25 via `rank-bm25`
- **CV parsing:** PyMuPDF (PDF), python-docx (DOCX)

---

## Requirements

- Python 3.11+
- An **OpenAI** or **Anthropic** API key — entered in the UI at runtime, never stored on disk
- ~500 MB free disk space for the local embedding and reranker models (downloaded automatically on first run)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/gnbasyal/rag_jd_matcher.git
cd rag_jd_matcher

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the project root with the following content (no API keys needed here):

```env
OPENAI_MODEL=gpt-4.1
ANTHROPIC_MODEL=claude-sonnet-4-6
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
JD_COLLECTION_NAME=job_descriptions
JD_SOURCE_DIR=./test_jds
```

### 4. Ingest job descriptions

This step runs the offline pipeline — it reads the JD files, tags them with LLM-extracted metadata, chunks and embeds them, and stores everything in ChromaDB. Run it once before starting the app, and again whenever JDs are added or changed.

```bash
# OpenAI
python -m app.jd_ingestion --api-key sk-... --provider openai

# Anthropic
python -m app.jd_ingestion --api-key sk-ant-... --provider anthropic
```

The 30 sample JDs in `test_jds/` are processed by default. Use `--source ./my_jds` to point to a different directory, and `--no-reset` to append to the existing collection instead of rebuilding it.

---

## Running the App

```bash
python main.py
```

Open `http://localhost:9000` in your browser.

---

## Using the App

### Step 1 — Enter your API key

Select your LLM provider (OpenAI or Anthropic) and paste your API key. The key is sent directly to the provider on each request and is never stored by the application. The submit button stays disabled until both a file and a key are provided.

### Step 2 — Upload your CV

Drag and drop a CV onto the upload area, or click to browse. Accepted formats: **PDF**, **DOCX**, **TXT**.

### Step 3 — Click "Match CV to Jobs"

Processing takes approximately 10–15 seconds. The status bar shows when the request is in progress.

### Step 4 — Review the results

Two columns appear side by side:

**Left — Candidate Profile**
Your CV parsed into structured cards: name, seniority level, total years of experience, industry domains, skills, tech stack, work history, and education. Review this to verify the CV was parsed correctly before acting on the job matches.

Between the upload area and the submit button there is a small info bar with two buttons:

- **View list of jobs** — opens a popup listing all job description filenames in the configured JD directory
- **Download JDs** — downloads the entire JD directory as a zip file

**Right — Matched Jobs**
Five job cards ranked by relevance. Each card shows the job title, company, and a match score (the five scores sum to 100%). Click any card to expand it:

- **Summary** — a 4–5 sentence explanation of why this job matches the candidate profile
- **Matching signals** — specific overlaps between the CV and the JD (e.g. *"5 years Python matches 4-year requirement"*)
- **Potential gaps** — JD requirements the CV does not clearly satisfy
- **Seniority fit** — one of three badges:
  - 🟢 **Match** — candidate's level aligns with the role
  - 🟡 **Under** — candidate is below the expected level
  - 🔵 **Over** — candidate is overqualified

### Error messages

| Message | Cause |
|---|---|
| *"API key is required."* | Key field was left empty |
| *"Invalid API key. Please check your key and try again."* | Key was rejected by the LLM provider |
| Other error text | CV parsing or pipeline failure — check the terminal running `main.py` for details |

---

## Adding More Job Descriptions

Place new `.txt` files in `test_jds/` (or another directory) and re-run ingestion with `--no-reset` to append without wiping the existing index:

```bash
python -m app.jd_ingestion --api-key sk-... --no-reset --source ./my_new_jds
```
