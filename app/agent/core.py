"""
agent/core.py
─────────────────────────────────────────────────────────────────────────────
Shared LLM/agent building blocks used by both the chat API (app/chat/router.py)
and the voice webhook (app/voice/vapi_webhook.py). Previously each router
duplicated its own copy of this logic and reloaded the Chroma vector store
from disk on every single message — both are fixed here.

  - get_chat_llm()      → provider-switchable LLM (Groq by default, OpenAI fallback)
  - get_cached_vector_store() → Chroma loaded from disk once per process
  - get_rag_context()   → shared retrieval helper
  - build_tool_agent()  → shared tool-calling AgentExecutor factory
"""

import os
from functools import lru_cache
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from app.rag.ingest import get_vector_store

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ── LLM Factory ───────────────────────────────────────────────────────────────
def get_chat_llm(max_tokens: Optional[int] = None, streaming: bool = False) -> BaseChatModel:
    """
    Return the chat-completion LLM for persona responses, switchable via
    LLM_PROVIDER. Groq is the default (fast + far cheaper than GPT-4o for a
    RAG-grounded, tool-calling persona bot). Embeddings always stay on OpenAI —
    Groq has no embeddings API — see app/rag/ingest.py.
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            streaming=streaming,
            max_tokens=max_tokens,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        streaming=streaming,
        max_tokens=max_tokens,
    )


# ── Vector Store Cache ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_cached_vector_store():
    """Load Chroma from disk once per process instead of once per request."""
    return get_vector_store()


# ── Shared RAG Retrieval ──────────────────────────────────────────────────────
def get_rag_context(question: str, k: int = 6) -> Tuple[str, List[dict]]:
    vs   = get_cached_vector_store()
    docs = vs.similarity_search(question, k=k)

    if not docs:
        return "No relevant context found.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = [
        {
            "doc_type": doc.metadata.get("doc_type", "unknown"),
            "source":   doc.metadata.get("source", "unknown"),
            "repo":     doc.metadata.get("repo_name", ""),
            "snippet":  doc.page_content[:150],
        }
        for doc in docs
    ]
    return context, sources


# ── Shared Tool-Calling Agent Builder ─────────────────────────────────────────
def build_tool_agent(
    system_text: str,
    chat_history: list,
    tools: List[BaseTool],
    max_tokens: Optional[int] = None,
    max_iterations: int = 4,
    streaming: bool = False,
) -> AgentExecutor:
    """
    Build a stateless tool-calling AgentExecutor. Uses create_tool_calling_agent
    (provider-agnostic — works with any chat model implementing .bind_tools(),
    including ChatGroq and ChatOpenAI) rather than create_openai_tools_agent,
    which only works against OpenAI's tool-call response format.

    streaming=True enables token streaming on the underlying LLM — consumed via
    AgentExecutor.astream_events() by callers that want SSE (see app/chat/router.py's
    /chat/stream). The tool-decision turn still arrives as a single chunk (tool calls
    aren't token-streamable); only the final synthesis turn actually streams text.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm   = get_chat_llm(max_tokens=max_tokens, streaming=streaming)
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
    )
