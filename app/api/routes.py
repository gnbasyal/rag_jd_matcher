import io
import zipfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.cv_parser import extract_text, parse_cv
from app.llm import build_llm
from app import explainer, query_builder, retriever, reranker

app = FastAPI(title="CV–JD Matcher")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jds")
def list_jds():
    src = Path(settings.jd_source_dir)
    files = sorted(p.name for p in src.glob("*.txt"))
    return {"files": files}


@app.get("/jds/download")
def download_jds():
    src = Path(settings.jd_source_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.glob("*.txt")):
            zf.write(path, arcname=path.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=job_descriptions.zip"},
    )


# ── Error helpers ─────────────────────────────────────────────────────────────

def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("incorrect api key", "invalid api key", "authentication", "401", "invalid x-api-key"))


def _friendly(exc: Exception) -> str:
    if _is_auth_error(exc):
        return "Invalid API key. Please check your key and try again."
    return str(exc)


def _status(exc: Exception) -> int:
    return 401 if _is_auth_error(exc) else 422


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/parse-cv")
async def parse_cv_endpoint(
    cv_file: UploadFile,
    api_key: str = Form(""),
    llm_provider: str = Form("openai"),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")

    file_bytes = await cv_file.read()

    try:
        raw_text = extract_text(file_bytes, cv_file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm = build_llm(api_key=api_key.strip(), provider=llm_provider)

    try:
        profile = parse_cv(raw_text, llm=llm)
    except Exception as e:
        raise HTTPException(status_code=_status(e), detail=_friendly(e))

    return profile.model_dump()


@app.post("/match")
async def match_endpoint(
    cv_file: UploadFile,
    api_key: str = Form(""),
    llm_provider: str = Form("openai"),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")

    file_bytes = await cv_file.read()

    try:
        raw_text = extract_text(file_bytes, cv_file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm = build_llm(api_key=api_key.strip(), provider=llm_provider)

    try:
        profile = parse_cv(raw_text, llm=llm)
    except Exception as e:
        raise HTTPException(status_code=_status(e), detail=_friendly(e))

    query = query_builder.build(profile)
    dense_hits, bm25_hits, merged = retriever.retrieve(query)
    top5 = reranker.rerank(query_builder.build_rerank_text(profile), merged)

    try:
        results = await explainer.explain_all(profile, top5, llm=llm)
    except Exception as e:
        raise HTTPException(status_code=_status(e), detail=_friendly(e))

    return {
        "profile": profile.model_dump(exclude={"raw_text"}),
        "candidate": {
            "name": profile.name,
            "seniority": profile.seniority,
            "years_experience": profile.total_years_experience,
            "domains": profile.domains,
        },
        "queries": {
            "dense": query.dense_query,
            "bm25": query.bm25_query,
            "filters": query.filters,
        },
        "retrieval": {
            "dense_hits": dense_hits,
            "bm25_hits": bm25_hits,
            "merged": [c.model_dump(exclude={"full_text"}) for c in merged],
        },
        "reranked": [r.model_dump() for r in results],
    }
