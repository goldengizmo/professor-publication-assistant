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
  POST /insights         -> match question to analytics template + pre-computed data
  POST /upload_masterlist -> save uploaded .docx for future pipeline ingestion

Startup: initialises Retriever (Pinecone + Cohere + BM25) once.
         Anthropic client is optional — omitting ANTHROPIC_API_KEY
         returns retrieved chunks without a synthesised answer.
"""

import os
import shutil
from collections import Counter
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
_insights_data: dict = {}
_professor_name: str = ""

_SYSTEM_PROMPT = (
    "You are a research assistant helping a professor explore their own body of work. "
    "Answer questions as if speaking to the professor about their own work — use 'you' and 'your'. "
    "Use only the provided context excerpts to answer. Be concise and cite specific findings. "
    "Never say you lack access to their publications or publication record; "
    "if the context doesn't contain the answer, say the available excerpts don't cover that detail."
)

_CLASSIFY_PROMPT = (
    "Classify this question into exactly one category. "
    "Only classify as a chart category if the question is asking for an overview, pattern, or trend across ALL the professor's work. "
    "Return 'none' for ANY question asking about specific findings, results, drugs, diseases, trials, or papers.\n\n"
    "Categories:\n"
    "  timeline      - trends or counts across years (e.g. 'publications over time', 'output by decade')\n"
    "  collaborators - co-authorship patterns (e.g. 'who have I worked with', 'research partners', 'which researchers')\n"
    "  themes        - research topics across corpus (e.g. 'what do I research', 'main topics', 'diseases I study')\n"
    "  geography     - locations in the work (e.g. 'where is my work focused', 'what countries', 'parts of the world')\n"
    "  none          - specific content, findings, results, or anything not clearly an overview\n\n"
    "Reply with exactly one word: timeline, collaborators, themes, geography, or none.\n"
    "Question: "
)

_FRONTEND    = Path(__file__).parent / "index.html"
_UPLOADS_DIR = Path(__file__).parent / "uploads"


# ── insights pre-computation ───────────────────────────────────────────────────

def _build_insights_data(corpus: list) -> dict:
    # Deduplicate: keep first chunk per publication
    seen: dict[int, dict] = {}
    for chunk in corpus:
        pid = chunk["publication_id"]
        if pid not in seen:
            seen[pid] = chunk
    pubs = list(seen.values())

    # 1. Timeline: publications per year
    year_counts: Counter = Counter()
    for p in pubs:
        yr = p.get("year")
        if yr:
            year_counts[str(yr)] += 1
    timeline = dict(sorted(year_counts.items()))

    # 2. Top 15 collaborators — detect professor by most frequent last name
    author_counts: Counter = Counter()
    last_name_counts: Counter = Counter()
    for p in pubs:
        for name in [a.strip() for a in (p.get("authors") or "").split(",") if a.strip()]:
            author_counts[name] += 1
            last_name_counts[name.split()[0]] += 1
    professor_last_name = last_name_counts.most_common(1)[0][0] if last_name_counts else ""
    filtered_counts = Counter({a: author_counts[a] for a in author_counts
                               if professor_last_name.lower() not in a.lower()})
    collaborators = [
        {"name": name, "count": count, "is_professor": False}
        for name, count in filtered_counts.most_common(15)
    ]

    # 3. Research themes: keyword scan of title + first 400 chars of first chunk
    theme_map = {
        "HIV/AIDS":              ["hiv", "aids", "antiretroviral"],
        "Tuberculosis":          ["tuberculosis", " tb ", "mycobacterium"],
        "Infectious Disease":    ["infection", "infectious", "pathogen"],
        "Immunology":            ["immune", "immunity", "immunodeficiency"],
        "Clinical Trials":       ["trial", "randomized", "randomised", "placebo"],
        "Epidemiology":          ["epidemiology", "cohort", "prevalence", "incidence"],
        "Treatment Outcomes":    ["treatment outcome", "viral suppression", "adherence"],
        "Coinfection":           ["coinfection", "hepatitis", "malaria"],
        "Mortality & Survival":  ["mortality", "survival", "fatality"],
        "Inflammation":          ["inflammation", "inflammatory", "cytokine"],
        "Aging & Comorbidities": ["aging", "ageing", "elderly", "comorbid"],
        "Neurology":             ["neurolog", "cognitive", "dementia"],
    }
    theme_counts: Counter = Counter()
    for p in pubs:
        text = ((p.get("title") or "") + " " + (p.get("chunk_text") or "")[:400]).lower()
        for theme, keywords in theme_map.items():
            if any(kw in text for kw in keywords):
                theme_counts[theme] += 1
    themes = [
        {"theme": t, "count": c}
        for t, c in sorted(theme_counts.items(), key=lambda x: -x[1])
        if c > 0
    ]

    # 4. Geographic focus: keyword scan of title + first 400 chars of first chunk
    geo_map = {
        "Sub-Saharan Africa": ["africa", "kenya", "uganda", "tanzania", "zimbabwe",
                               "ethiopia", "malawi", "zambia", "south africa", "nigeria",
                               "mozambique", "rwanda", "botswana"],
        "United States":      ["united states", " u.s.", "usa", "american"],
        "Asia":               ["asia", "india", "china", "thailand", "vietnam",
                               "cambodia", "myanmar", "indonesia"],
        "Europe":             ["europe", "united kingdom", "france", "germany"],
        "Latin America":      ["latin america", "brazil", "peru", "haiti", "mexico"],
        "Global/Multi-site":  ["global", "multinational", "multi-site", "multisite",
                               "international", "multicountry"],
    }
    geo_counts: Counter = Counter()
    for p in pubs:
        text = ((p.get("title") or "") + " " + (p.get("chunk_text") or "")[:400]).lower()
        for region, keywords in geo_map.items():
            if any(kw in text for kw in keywords):
                geo_counts[region] += 1
    geography = [
        {"region": r, "count": c}
        for r, c in sorted(geo_counts.items(), key=lambda x: -x[1])
        if c > 0
    ]

    return (
        {
            "timeline":      timeline,
            "collaborators": collaborators,
            "themes":        themes,
            "geography":     geography,
        },
        professor_last_name,
    )


# ── lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _anthropic, _insights_data, _professor_name
    _retriever = Retriever()
    _insights_data, _professor_name = _build_insights_data(_retriever._corpus)
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


class InsightsRequest(BaseModel):
    question: str


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


def _classify_insight_question(question: str) -> str:
    q = question.lower().strip().strip('"').strip("'")

    # Specific-content questions belong to RAG, not chart templates.
    # Check these before the LLM and keyword fallback.
    _rag_phrases = [
        "what did", "what were", "what was", "what has", "what have",
        "how did", "how does", "how has",
        "explain", "describe", "summarize", "summarise",
        "tell me about", "what do you know about",
        "what is the result", "what are the result",
        "what is the finding", "what are the finding",
    ]
    if any(p in q for p in _rag_phrases):
        print(f"[classify] RAG pre-screen match -> none | q={question!r}")
        return "none"

    if _anthropic:
        try:
            resp = _anthropic.messages.create(
                model="claude-haiku-4-5",
                max_tokens=10,
                messages=[{"role": "user", "content": _CLASSIFY_PROMPT + question}],
            )
            word = resp.content[0].text.strip().lower().split()[0]
            if word.startswith("collabor"):
                print(f"[classify] LLM -> collaborators | q={question!r}")
                return "collaborators"
            if word.startswith("geograph"):
                print(f"[classify] LLM -> geography | q={question!r}")
                return "geography"
            if word.startswith("theme"):
                print(f"[classify] LLM -> themes | q={question!r}")
                return "themes"
            if word.startswith("time"):
                print(f"[classify] LLM -> timeline | q={question!r}")
                return "timeline"
            print(f"[classify] LLM -> none (word={word!r}) | q={question!r}")
            return "none"
        except Exception:
            pass  # fall through to keyword fallback

    # Keyword fallback: only match on strong analytical signals.
    if any(w in q for w in ["how many", "over time", "over the years", "over the decades",
                             "trend", "annual", "output", "changed over", "decades", "productivity"]):
        print(f"[classify] keyword -> timeline | q={question!r}")
        return "timeline"
    if any(w in q for w in ["collaborat", "co-author", "coauthor", "who have i worked", "colleague"]):
        print(f"[classify] keyword -> collaborators | q={question!r}")
        return "collaborators"
    if any(w in q for w in ["where", "country", "countries", "geographic", "geography",
                             "location", "region", "place", "international", "global"]):
        print(f"[classify] keyword -> geography | q={question!r}")
        return "geography"
    if any(w in q for w in ["which topics", "what topics", "research themes", "research areas",
                             "main themes", "main topics", "what subjects"]):
        print(f"[classify] keyword -> themes | q={question!r}")
        return "themes"
    print(f"[classify] no match -> none | q={question!r}")
    return "none"


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(str(_FRONTEND))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/authors")
async def debug_authors():
    from collections import Counter
    author_counts = Counter()
    for chunk in _retriever._corpus:
        for name in [a.strip() for a in (chunk.get("authors") or "").split(",") if a.strip()]:
            author_counts[name] += 1
    return {"top_30": author_counts.most_common(30)}


@app.get("/status")
async def status():
    stats = _retriever.corpus_stats()
    stats["professor_name"] = _professor_name
    return JSONResponse(content=stats)


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


@app.post("/insights")
async def insights(body: InsightsRequest):
    if not body.question.strip():
        return JSONResponse({"error": "question is required"}, status_code=400)

    template = _classify_insight_question(body.question)

    if template not in _insights_data:
        return JSONResponse({
            "template": "none",
            "message":  "Try asking about publication trends, collaborators, research themes, or geographic focus",
            "data":     None,
        })

    return JSONResponse({
        "template": template,
        "message":  None,
        "data":     _insights_data[template],
    })
