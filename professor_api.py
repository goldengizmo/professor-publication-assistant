"""
professor_api.py  —  Phase 5 Step 4
FastAPI backend for the professor publication assistant.

Endpoints:
  GET  /           -> index.html (frontend)
  GET  /health     -> {"status": "ok"}
  GET  /status     -> corpus stats (chunks, publications, years, model)
  GET  /publications -> deduplicated publication list from corpus
  GET  /filters    -> available years and file types for UI dropdowns
  POST /ask              -> hybrid retrieve + Claude answer
  POST /upload_masterlist -> save uploaded .docx for future pipeline ingestion

Startup: initialises Retriever (Pinecone + Cohere + BM25) once.
         Anthropic client is optional — omitting ANTHROPIC_API_KEY
         returns retrieved chunks without a synthesised answer.
"""

import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from professor_retrieval import Retriever

load_dotenv()

# ── globals ────────────────────────────────────────────────────────────────────

_retriever: Optional[Retriever] = None
_anthropic = None

_SYSTEM_PROMPT = (
    "You are a research assistant with access to excerpts from a professor's "
    "published papers. Answer the user's question using only the provided context. "
    "Be concise, cite specific findings, and indicate when the context is insufficient."
)

_FRONTEND    = Path(__file__).parent / "index.html"
_UPLOADS_DIR = Path(__file__).parent / "uploads"


# ── lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _anthropic
    _retriever = Retriever()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        import anthropic as _ant
        _anthropic = _ant.Anthropic(api_key=anthropic_key)
        print("Anthropic client ready")
    else:
        print("WARNING: ANTHROPIC_API_KEY not set — /ask will return sources only")
    yield


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Professor Publication Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:       str
    publication_id: Optional[int] = None
    title:          Optional[str] = None   # resolved to publication_id via corpus lookup
    doi:            Optional[str] = None
    year_from:      Optional[int] = None
    year_to:        Optional[int] = None
    author:         Optional[str] = None
    top_k:          int           = 8


# ── internal helpers ───────────────────────────────────────────────────────────

def _rewrite_query(question: str) -> str:
    resp = _anthropic.messages.create(
        model="claude-haiku-4-5",
        max_tokens=120,
        messages=[{
            "role": "user",
            "content": (
                "Rewrite the following question as a concise search query optimised "
                "for retrieving relevant academic text chunks. Output only the query, "
                "no explanation.\n\nQuestion: " + question
            ),
        }],
    )
    return resp.content[0].text.strip()


def _generate_answer(question: str, results) -> str:
    context = "\n\n---\n\n".join(
        f"[{r.title} ({r.year})]\n{r.chunk_text}"
        for r in results
    )
    resp = _anthropic.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return resp.content[0].text


def _resolve_title(title: str) -> Optional[int]:
    """Find the first publication_id whose title contains the given substring."""
    needle = title.lower()
    for chunk in _retriever._corpus:
        if needle in chunk.get("title", "").lower():
            return chunk["publication_id"]
    return None


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(str(_FRONTEND))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    return JSONResponse(content=_retriever.corpus_stats())


@app.get("/publications")
async def publications():
    seen: dict[int, dict] = {}
    for chunk in _retriever._corpus:
        pid = chunk["publication_id"]
        if pid not in seen:
            seen[pid] = {
                "publication_id": pid,
                "title":          chunk["title"],
                "authors":        chunk["authors"],
                "year":           chunk["year"],
                "doi":            chunk["doi"],
                "pmid":           chunk["pmid"],
                "pmcid":          chunk["pmcid"],
                "file_type":      chunk["file_type"],
            }
    pubs = sorted(seen.values(), key=lambda x: x["publication_id"])
    return JSONResponse(content={"publications": pubs, "total": len(pubs)})


@app.get("/filters")
async def filters():
    years  = sorted({c["year"]      for c in _retriever._corpus if c.get("year")})
    ftypes = sorted({c["file_type"] for c in _retriever._corpus if c.get("file_type")})
    return JSONResponse(content={"years": years, "file_types": ftypes})


@app.post("/upload_masterlist")
async def upload_masterlist(file: UploadFile = File(...)):
    """
    Accept a .docx publication master list and save it to disk.
    Future: trigger parse -> enrich -> download -> vectorize pipeline.
    """
    if not file.filename.lower().endswith(".docx"):
        return JSONResponse(
            {"error": "Only .docx files are accepted."},
            status_code=400,
        )
    try:
        _UPLOADS_DIR.mkdir(exist_ok=True)
        dest = _UPLOADS_DIR / file.filename
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        size_kb = dest.stat().st_size // 1024
        return JSONResponse({
            "message":  f'"{file.filename}" uploaded successfully ({size_kb} KB). Archive build queued.',
            "filename": file.filename,
            "saved_to": str(dest),
            "status":   "pending",
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/ask")
async def ask(body: AskRequest):
    try:
        if not body.question.strip():
            return JSONResponse({"error": "question is required"}, status_code=400)

        # Resolve title -> publication_id if needed
        pub_id = body.publication_id
        if pub_id is None and body.title:
            pub_id = _resolve_title(body.title)

        # Query rewrite (optional)
        search_query = body.question
        if _anthropic:
            try:
                search_query = _rewrite_query(body.question)
            except Exception:
                pass  # fall back to original question

        # Retrieve
        results = _retriever.retrieve(
            question        = search_query,
            top_k           = body.top_k,
            publication_id  = pub_id,
            year_from       = body.year_from,
            year_to         = body.year_to,
            doi             = body.doi,
            author_contains = body.author,
        )

        # Synthesise answer
        if not results:
            answer = "No relevant passages found for this query and filters."
        elif _anthropic:
            answer = _generate_answer(body.question, results)
        else:
            answer = (
                "Retrieved sources are shown below. "
                "Add ANTHROPIC_API_KEY to .env for a synthesised answer."
            )

        return JSONResponse(content={
            "answer":     answer,
            "sources":    [asdict(r) for r in results],
            "query_used": search_query,
        })

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
