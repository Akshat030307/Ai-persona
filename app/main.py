"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Registers all routers:
  /voice   — Vapi webhook handler
  /chat    — Chat API
  /health  — top-level health check
  /        — Chat frontend UI (served from frontend/index.html)

Run locally:
    uvicorn app.main:app --reload --port 8000

For production:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

CANDIDATE_NAME = os.getenv("CANDIDATE_NAME", "Candidate")
VAPI_PHONE     = os.getenv("VAPI_PHONE_NUMBER_DISPLAY", "")  # e.g. +1 (239) 663 4085


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vector store on startup so first request isn't slow."""
    logger.info("Starting AI Persona server...")
    try:
        from app.rag.ingest import get_vector_store
        vs    = get_vector_store()
        count = vs._collection.count()
        logger.info(f"Vector store loaded — {count} chunks indexed")
    except Exception as e:
        logger.warning(f"Vector store not ready: {e}. Run ingestion first.")
    yield
    logger.info("Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Persona — Candidate Representative",
    description=(
        "AI persona system for candidate screening. "
        "Handles voice calls (via Vapi webhook) and chat sessions, "
        "with RAG-grounded Q&A and real Google Calendar booking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.voice.vapi_webhook import router as voice_router
from app.chat.router        import router as chat_router

app.include_router(voice_router)
app.include_router(chat_router)


# ── Frontend — serve index.html with env vars injected ───────────────────────
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the chat UI with CANDIDATE_NAME and phone number
    injected from environment variables — no hardcoding needed.
    """
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h2>Frontend not found. Place index.html in /frontend/</h2>", status_code=404)

    html = FRONTEND_PATH.read_text(encoding="utf-8")

    # Inject real values from .env into the JS config block
    html = html.replace(
        'const CAND_NAME   = "Your Name";',
        f'const CAND_NAME   = "{CANDIDATE_NAME}";'
    )
    html = html.replace(
        'const PHONE_NUM   = "+1 (555) 000-0000";',
        f'const PHONE_NUM   = "{VAPI_PHONE}";'
    )

    return HTMLResponse(html)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api")
async def api_info():
    return {
        "candidate":  CANDIDATE_NAME,
        "voice":      "/voice/webhook  (POST — Vapi webhook)",
        "chat":       "/chat/message   (POST — chat API)",
        "docs":       "/docs           (Swagger UI)",
        "frontend":   "/               (Chat UI)",
    }
