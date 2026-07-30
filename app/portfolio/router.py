"""
portfolio/router.py
─────────────────────────────────────────────────────────────────────────────
Pure metadata API for the portfolio UI's Projects section. Reads the same
data/projects/*.md files used by RAG ingestion (app/rag/ingest.py) — single
source of truth, no LLM/RAG call involved here.
"""

from fastapi import APIRouter

from app.rag.projects_loader import load_project_files, drive_share_to_embed

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/projects")
async def list_projects():
    projects = load_project_files()
    return [
        {
            "slug":            p.slug,
            "title":           p.title,
            "summary":         p.summary,
            "tags":            p.tags,
            "stack":           p.stack,
            "github_url":      p.github_url,
            "video_embed_url": drive_share_to_embed(p.drive_video_url),
        }
        for p in projects
    ]
