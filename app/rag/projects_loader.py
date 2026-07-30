"""
rag/projects_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads curated project write-ups from data/projects/*.md (YAML frontmatter +
markdown body). This is the single source of truth for both:
  - RAG ingestion (app/rag/ingest.py)      → grounds chat/voice answers
  - the portfolio API (app/portfolio/router.py) → feeds the Projects UI section

Each file looks like:

    ---
    title: ELIRA
    slug: elira
    order: 1
    tags: [RAG, LangChain, FastAPI]
    stack: [Python, LangChain, ChromaDB, OpenAI]
    github_url: https://github.com/Akshat030307/ELIRA
    drive_video_url: https://drive.google.com/file/d/XXXX/view?usp=sharing
    summary: One-line blurb for the project card.
    ---
    Full markdown description used both for RAG grounding and as the
    "read more" body on the project card.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PROJECTS_DIR = Path(__file__).parent.parent.parent / "data" / "projects"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

# Matches Google Drive share links, e.g.
#   https://drive.google.com/file/d/1AbCDeFGhIJkLmNoPQRstuVWxyz/view?usp=sharing
DRIVE_FILE_ID_RE = re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")


# ── Model ─────────────────────────────────────────────────────────────────────
class ProjectMeta(BaseModel):
    title:            str
    slug:             str
    order:            int = 99
    tags:             List[str] = []
    stack:            List[str] = []
    github_url:       Optional[str] = None
    drive_video_url:  Optional[str] = None
    summary:          str = ""
    body:             str = ""


# ── Google Drive Embed Helper ─────────────────────────────────────────────────
def drive_share_to_embed(url: Optional[str]) -> Optional[str]:
    """
    Convert a Google Drive share link into an embeddable /preview URL.
    Returns None if the URL isn't a recognizable Drive file link.
    File must be shared as "Anyone with the link" (Viewer) for the embed to load.
    """
    if not url:
        return None
    match = DRIVE_FILE_ID_RE.search(url)
    if not match:
        return None
    file_id = match.group(1)
    return f"https://drive.google.com/file/d/{file_id}/preview"


# ── Frontmatter Parsing ────────────────────────────────────────────────────────
def _parse_project_file(path: Path) -> Optional[ProjectMeta]:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        logger.warning(f"Skipping {path.name}: no YAML frontmatter found.")
        return None

    frontmatter_raw, body = match.groups()
    meta = yaml.safe_load(frontmatter_raw) or {}

    meta.setdefault("slug", path.stem)
    meta["body"] = body.strip()

    try:
        return ProjectMeta(**meta)
    except Exception as e:
        logger.warning(f"Skipping {path.name}: invalid frontmatter — {e}")
        return None


def load_project_files() -> List[ProjectMeta]:
    """Load and sort every data/projects/*.md file into ProjectMeta objects."""
    if not PROJECTS_DIR.exists():
        logger.warning(f"Projects dir not found at {PROJECTS_DIR}. Skipping.")
        return []

    projects = []
    for path in sorted(PROJECTS_DIR.glob("*.md")):
        project = _parse_project_file(path)
        if project:
            projects.append(project)

    projects.sort(key=lambda p: p.order)
    logger.info(f"Loaded {len(projects)} project files from {PROJECTS_DIR}")
    return projects


# ── RAG Document Conversion ────────────────────────────────────────────────────
def to_documents(projects: List[ProjectMeta]) -> List[Document]:
    """Convert curated project write-ups into Documents for the vector store."""
    docs = []
    for p in projects:
        content = (
            f"# Project: {p.title}\n\n"
            f"Summary: {p.summary}\n"
            f"Tech Stack: {', '.join(p.stack) or 'Unknown'}\n"
            f"Tags: {', '.join(p.tags) or 'none'}\n"
            + (f"GitHub: {p.github_url}\n" if p.github_url else "")
            + (f"Demo video: {p.drive_video_url}\n" if p.drive_video_url else "")
            + f"\n{p.body}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":          "curated",
                "doc_type":        "project",
                "slug":            p.slug,
                "repo_name":       p.title,
                "github_url":      p.github_url or "",
                "drive_video_url": p.drive_video_url or "",
            },
        ))
    return docs


def load_projects() -> List[Document]:
    """Convenience wrapper used by app/rag/ingest.py."""
    return to_documents(load_project_files())
