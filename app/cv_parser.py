"""
CV Parser (online pipeline — Step 2)
======================================
extract_text()  — pull raw text from PDF / DOCX / TXT bytes
parse_cv()      — LLM call → structured CVProfile
"""
from __future__ import annotations

from datetime import date
from io import BytesIO

from langchain_core.prompts import ChatPromptTemplate

from app.models import CVProfile


_SUPPORTED = {".pdf", ".docx", ".txt"}

_CV_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert CV analyst. Extract a structured profile from the CV text below.\n"
        "Today's date is {current_date}.\n"
        "Rules:\n"
        "- name: candidate's full name.\n"
        "- total_years_experience: sum of years across all roles (float). "
        "For roles still listed as 'present', 'current', or with no end date, "
        "calculate duration using today's date ({current_date}).\n"
        "- seniority: 'junior' (0-2 yr), 'mid' (2-5 yr), 'senior' (5-10 yr), "
        "'lead' (10+ yr or people-management title).\n"
        "- skills: all competencies — soft and hard — excluding items already in tech_stack.\n"
        "- tech_stack: hard technical tools, languages, frameworks, and platforms only.\n"
        "- roles: list of work experiences with title, company, years (float), and a brief description.\n"
        "- education: list of degrees with degree name, institution, and year (if stated).\n"
        "- domains: industry domains the candidate has worked in (e.g. fintech, healthcare, e-commerce).\n"
        "- raw_text: copy the full CV text verbatim into this field.\n"
        "If a field cannot be determined, use an empty list [] or 0.0 for numbers.",
    ),
    ("human", "{cv_text}"),
])

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT bytes."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in _SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Accepted: {', '.join(_SUPPORTED)}")

    if suffix == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()

    if suffix == ".docx":
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    # .txt
    return file_bytes.decode("utf-8", errors="replace").strip()


def parse_cv(raw_text: str, llm) -> CVProfile:
    """Send raw CV text to LLM and return a structured CVProfile."""
    chain = _CV_PROMPT | llm.with_structured_output(CVProfile)
    return chain.invoke({"cv_text": raw_text, "current_date": date.today().isoformat()})
