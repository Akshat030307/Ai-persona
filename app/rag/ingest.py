"""
rag/ingest.py
─────────────────────────────────────────────────────────────────────────────
Ingestion pipeline:
  1. Load resume PDF  →  split into chunks
  2. Fetch GitHub repos via API → load README + repo metadata
  3. Embed all chunks with OpenAI embeddings
  4. Persist to Chroma vector store

Run once to build the knowledge base:
    python -m app.rag.ingest
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from github import Github, GithubException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
RESUME_PATH        = os.getenv("RESUME_PATH", "./data/resume/resume.pdf")
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME    = os.getenv("GITHUB_USERNAME")
COLLECTION_NAME    = "candidate_knowledge"

CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 64


# ── Splitter ──────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Resume Loader ─────────────────────────────────────────────────────────────
def load_resume() -> List[Document]:
    """Load and chunk the candidate's resume PDF."""
    path = Path(RESUME_PATH)
    if not path.exists():
        logger.warning(f"Resume not found at {path}. Skipping.")
        return []

    logger.info(f"Loading resume from {path}")
    loader = PyPDFLoader(str(path))
    pages  = loader.load()

    # Tag every chunk with source metadata
    for page in pages:
        page.metadata.update({
            "source":   "resume",
            "doc_type": "resume",
            "file":     path.name,
        })

    chunks = splitter.split_documents(pages)
    logger.info(f"Resume → {len(chunks)} chunks")
    return chunks


# ── GitHub Loader ─────────────────────────────────────────────────────────────
def load_github_repos() -> List[Document]:
    """
    For each public repo owned by GITHUB_USERNAME:
      - Fetch README.md content
      - Fetch repo description, topics, language, stars
      - Create a synthetic 'repo summary' document
    """
    if not GITHUB_TOKEN or not GITHUB_USERNAME:
        logger.warning("GITHUB_TOKEN or GITHUB_USERNAME not set. Skipping GitHub ingestion.")
        return []

    gh   = Github(GITHUB_TOKEN)
    user = gh.get_user(GITHUB_USERNAME)
    docs: List[Document] = []

    for repo in user.get_repos(type="public"):
        try:
            repo_docs = _load_single_repo(repo)
            docs.extend(repo_docs)
        except GithubException as e:
            logger.warning(f"Skipping repo {repo.name}: {e}")

    logger.info(f"GitHub → {len(docs)} chunks from {GITHUB_USERNAME}'s repos")
    return docs


def _load_single_repo(repo) -> List[Document]:
    docs = []

    # ── Repo Summary Doc ──────────────────────────────────────────────────────
    topics   = ", ".join(repo.get_topics()) or "none"
    summary  = (
        f"Repository: {repo.full_name}\n"
        f"Description: {repo.description or 'No description'}\n"
        f"Primary Language: {repo.language or 'Unknown'}\n"
        f"Topics/Tags: {topics}\n"
        f"Stars: {repo.stargazers_count}\n"
        f"URL: {repo.html_url}\n"
        f"Created: {repo.created_at.strftime('%B %Y')}\n"
        f"Last Updated: {repo.updated_at.strftime('%B %Y')}\n"
    )
    docs.append(Document(
        page_content=summary,
        metadata={
            "source":    "github",
            "doc_type":  "repo_summary",
            "repo_name": repo.name,
            "repo_url":  repo.html_url,
            "language":  repo.language or "Unknown",
        }
    ))

    # ── README ────────────────────────────────────────────────────────────────
    try:
        readme   = repo.get_readme()
        readme_text = readme.decoded_content.decode("utf-8", errors="ignore")
        readme_doc  = Document(
            page_content=f"# README: {repo.name}\n\n{readme_text}",
            metadata={
                "source":    "github",
                "doc_type":  "readme",
                "repo_name": repo.name,
                "repo_url":  repo.html_url,
                "file":      "README.md",
            }
        )
        chunks = splitter.split_documents([readme_doc])
        docs.extend(chunks)
        logger.info(f"  {repo.name}: README → {len(chunks)} chunks")
    except GithubException:
        logger.info(f"  {repo.name}: no README found")

    # ── Recent Commits (last 20) ───────────────────────────────────────────────
    try:
        commits = list(repo.get_commits()[:20])
        commit_text = f"# Recent Commit Messages: {repo.name}\n\n"
        for c in commits:
            msg   = c.commit.message.split("\n")[0][:120]  # first line only
            date  = c.commit.author.date.strftime("%Y-%m-%d")
            commit_text += f"- [{date}] {msg}\n"

        docs.append(Document(
            page_content=commit_text,
            metadata={
                "source":    "github",
                "doc_type":  "commits",
                "repo_name": repo.name,
                "repo_url":  repo.html_url,
            }
        ))
    except GithubException:
        pass

    return docs


# ── Build Vector Store ────────────────────────────────────────────────────────
def build_vector_store(docs: List[Document]) -> Chroma:
    """Embed all documents and persist to Chroma."""
    logger.info(f"Embedding {len(docs)} total chunks into Chroma...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info(f"Vector store saved to {CHROMA_PERSIST_DIR}")
    return vector_store


def get_vector_store() -> Chroma:
    """Load existing Chroma vector store (call after ingest)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
def run_ingestion():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("Starting knowledge base ingestion...")

    docs = []
    docs += load_resume()
    docs += load_github_repos()

    if not docs:
        logger.error("No documents loaded. Check RESUME_PATH and GITHUB_TOKEN.")
        return

    build_vector_store(docs)
    logger.info(f"Ingestion complete. {len(docs)} chunks indexed.")


if __name__ == "__main__":
    run_ingestion()
