"""
voice/vapi_webhook.py
─────────────────────────────────────────────────────────────────────────────
Vapi custom-llm integration.

Vapi sends requests to /voice/webhook/chat/completions in OpenAI format.
We respond in OpenAI streaming format with our RAG-grounded answers.
"""

import os
import json
import time
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from app.rag.ingest import get_vector_store
from app.calendar.tools import CALENDAR_TOOLS

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

CANDIDATE_NAME      = os.getenv("CANDIDATE_NAME", "the candidate")
CANDIDATE_ROLE      = os.getenv("CANDIDATE_ROLE_APPLYING", "AI Engineer at Scaler")
VAPI_WEBHOOK_SECRET = os.getenv("VAPI_WEBHOOK_SECRET", "")
MAX_HISTORY_TURNS   = 8

# ── Call Memory ───────────────────────────────────────────────────────────────
_call_history: dict = {}


def _get_history(call_id: str) -> list:
    return _call_history.get(call_id, [])


def _save_turn(call_id: str, human: str, ai: str):
    if call_id not in _call_history:
        _call_history[call_id] = []
    _call_history[call_id].append({"human": human, "ai": ai})
    if len(_call_history[call_id]) > MAX_HISTORY_TURNS:
        _call_history[call_id] = _call_history[call_id][-MAX_HISTORY_TURNS:]


def _build_lc_history(history: list) -> list:
    msgs = []
    for t in history:
        msgs.append(HumanMessage(content=t["human"]))
        msgs.append(AIMessage(content=t["ai"]))
    return msgs


# ── RAG ───────────────────────────────────────────────────────────────────────
def _get_rag_context(question: str, k: int = 4) -> str:
    vs   = get_vector_store()
    docs = vs.similarity_search(question, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs) or "No context found."


# ── Agent ─────────────────────────────────────────────────────────────────────
def _build_agent(rag_context: str, chat_history: list) -> AgentExecutor:
    system_text = (
        f"You are the AI representative of {CANDIDATE_NAME}, "
        f"speaking on a phone call with a recruiter evaluating them for {CANDIDATE_ROLE}.\n\n"
        "VOICE RULES:\n"
        "1. Keep ALL responses to 2-3 sentences maximum. This is a phone call.\n"
        "2. Speak naturally — no bullet points, no markdown, no asterisks.\n"
        f"3. Only state facts from the context below. If unsure say: "
        f"'I don't have that detail — {CANDIDATE_NAME} can follow up on that.'\n"
        "4. NEVER hallucinate skills, projects, or credentials.\n"
        "5. For scheduling: use check_availability, present 2-3 slots naturally.\n"
        "6. Once recruiter picks a slot and gives name + email, use book_meeting immediately.\n"
        "7. You remember the full conversation — refer back when relevant.\n\n"
        f"RETRIEVED CONTEXT:\n{rag_context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        streaming=False,
        max_tokens=180,
    )

    agent = create_openai_tools_agent(llm, CALENDAR_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=CALENDAR_TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )


def _trim_for_voice(text: str) -> str:
    """Strip markdown formatting for clean speech."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,3}\s+',     '',    text)
    text = re.sub(r'^- ',           '',    text, flags=re.MULTILINE)
    text = re.sub(r'\n+',           ' ',   text)
    return text.strip()


# ── OpenAI-compatible /chat/completions endpoint ──────────────────────────────
@router.post("/webhook/chat/completions")
async def vapi_chat_completions(request: Request):
    """
    Vapi calls this endpoint in OpenAI chat completions format.
    We parse the messages, run our RAG agent, and return in OpenAI format.
    """
    t0 = time.time()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages  = body.get("messages", [])
    call_id   = body.get("call", {}).get("id", "unknown")
    stream    = body.get("stream", False)

    # Extract the latest user message
    user_input = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_input = msg.get("content", "").strip()
            break

    logger.info(f"Voice turn | call: {call_id[:8]} | input: {user_input[:60]}")

    # First turn with no user input → return intro
    if not user_input:
        response_text = (
            f"Hi there! I'm the AI assistant representing {CANDIDATE_NAME}. "
            f"I'm here to answer your questions about their background and can also "
            f"schedule an interview. What would you like to know?"
        )
    else:
        try:
            rag_context  = _get_rag_context(user_input)
            history      = _get_history(call_id)
            chat_history = _build_lc_history(history)
            executor     = _build_agent(rag_context, chat_history)
            result       = await executor.ainvoke({
                "input":        user_input,
                "chat_history": chat_history,
            })
            response_text = _trim_for_voice(result.get("output", "Could you repeat that?"))
            _save_turn(call_id, user_input, response_text)

            t1 = time.time()
            logger.info(f"Voice response | call: {call_id[:8]} | latency: {t1-t0:.2f}s")

        except Exception as e:
            logger.error(f"Voice agent error: {e}", exc_info=True)
            response_text = "I had a brief technical issue. Could you repeat your question?"

    # Return in OpenAI chat completions format
    if stream:
        return StreamingResponse(
            _stream_openai_response(response_text),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id":      f"chatcmpl-{call_id[:8]}",
        "object":  "chat.completion",
        "model":   "gpt-4o",
        "choices": [{
            "index":         0,
            "message": {
                "role":    "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


async def _stream_openai_response(text: str) -> AsyncGenerator[str, None]:
    """Stream response in OpenAI SSE format."""
    chunk = {
        "id":      "chatcmpl-stream",
        "object":  "chat.completion.chunk",
        "model":   "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": text},
            "finish_reason": None,
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Send finish chunk
    finish_chunk = {
        "id":      "chatcmpl-stream",
        "object":  "chat.completion.chunk",
        "model":   "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── Event Webhook (status updates, end-of-call) ───────────────────────────────
@router.post("/webhook")
async def vapi_events(request: Request):
    """Handles Vapi event notifications (status, end-of-call, etc.)"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "ok"})

    message    = body.get("message", {})
    event_type = message.get("type", "")
    call_id    = message.get("call", {}).get("id", "unknown")

    if event_type == "end-of-call-report":
        duration = message.get("durationSeconds", 0)
        turns    = len(_call_history.get(call_id, []))
        logger.info(f"Call ended | {call_id[:8]} | duration: {duration}s | turns: {turns}")
        _call_history.pop(call_id, None)
    else:
        logger.info(f"Vapi event: {event_type} | call: {call_id[:8]}")

    return JSONResponse({"status": "ok"})