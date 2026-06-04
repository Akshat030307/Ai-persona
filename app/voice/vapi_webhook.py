"""
voice/vapi_webhook.py
─────────────────────────────────────────────────────────────────────────────
FastAPI router that handles Vapi webhook events.

How Vapi works with this server:
  1. Recruiter calls your Vapi phone number
  2. Vapi transcribes speech → POST /voice/webhook with {type: "assistant-request"}
  3. This server runs RAG + optionally calls calendar tools
  4. Returns {"response": "..."} → Vapi speaks the text via TTS

Vapi assistant config (set via API or dashboard):
  - serverUrl: https://your-domain.com/voice/webhook
  - firstMessage: set to "" (we return it dynamically)
  - model: server (not a built-in model — we are the model)

Reference: https://docs.vapi.ai/server-url
"""

import os
import logging
import json
from typing import Any, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from app.rag.ingest import get_vector_store
from app.calendar.tools import CALENDAR_TOOLS
from app.rag.chain import get_or_create_chain, clear_session

load_dotenv_called = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv_called = True
except ImportError:
    pass

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

CANDIDATE_NAME = os.getenv("CANDIDATE_NAME", "the candidate")
CANDIDATE_ROLE = os.getenv("CANDIDATE_ROLE_APPLYING", "AI Engineer at Scaler")

# ── Voice-Specific System Prompt ──────────────────────────────────────────────
VOICE_SYSTEM_PROMPT = f"""You are the AI representative of {CANDIDATE_NAME}, speaking on a phone call \
with a recruiter from Scaler who is evaluating {CANDIDATE_NAME} for the {CANDIDATE_ROLE} role.

VOICE CONVERSATION RULES:
1. Keep all responses SHORT — 2 to 3 sentences maximum per turn. This is a phone call.
2. Speak naturally, as if you are {CANDIDATE_NAME}'s knowledgeable colleague.
3. ONLY state facts grounded in what you know about {CANDIDATE_NAME}. If unsure, say so honestly.
4. NEVER invent details about projects, experience, or skills.
5. If asked about scheduling/availability, use the check_availability tool FIRST, then present 2-3 slots.
6. If the recruiter picks a slot and gives their name and email, use book_meeting to confirm immediately.
7. Do not say "based on the context" or refer to your tools — speak naturally.
8. On call start, introduce yourself warmly in 1-2 sentences.

RETRIEVED CONTEXT ABOUT {CANDIDATE_NAME}:
{{rag_context}}
"""

# ── In-memory voice session store ─────────────────────────────────────────────
# Stores AgentExecutor per call_id
_voice_sessions: dict[str, dict] = {}


def _get_rag_context(question: str) -> str:
    """Quick RAG retrieval for injecting context into voice agent prompt."""
    vs      = get_vector_store()
    docs    = vs.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return context or "No specific context found."


def _get_or_create_voice_agent(call_id: str, question: str) -> AgentExecutor:
    """Create or retrieve an AgentExecutor for a voice call session."""
    if call_id not in _voice_sessions:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=8,
        )

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            streaming=False,   # Vapi handles streaming; we return full response
            max_tokens=200,    # Keep voice responses short
        )

        # Inject fresh RAG context into the system prompt
        rag_ctx = _get_rag_context(question)
        system  = VOICE_SYSTEM_PROMPT.format(rag_context=rag_ctx)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder("chat_history"),
            ("human",  "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent   = create_openai_tools_agent(llm, CALENDAR_TOOLS, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=CALENDAR_TOOLS,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
        )

        _voice_sessions[call_id] = {
            "executor": executor,
            "memory":   memory,
        }
        logger.info(f"New voice session: {call_id}")

    return _voice_sessions[call_id]["executor"]


# ── Vapi Webhook Endpoint ─────────────────────────────────────────────────────
@router.post("/webhook")
async def vapi_webhook(request: Request):
    """
    Main webhook handler for all Vapi events.
    Vapi sends different event types; we handle:
      - assistant-request   (first message / turn)
      - function-call       (if configured for tool use via Vapi side)
      - end-of-call-report  (cleanup)
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = body.get("message", {}).get("type", "")
    call_id    = body.get("message", {}).get("call", {}).get("id", "unknown")

    logger.info(f"Vapi event: {event_type} | call: {call_id}")

    # ── 1. Assistant Request (main conversation turn) ──────────────────────────
    if event_type == "assistant-request":
        return await _handle_assistant_request(body, call_id)

    # ── 2. End of Call ─────────────────────────────────────────────────────────
    elif event_type == "end-of-call-report":
        _cleanup_voice_session(call_id, body)
        return JSONResponse({"status": "ok"})

    # ── 3. Status Updates (ignore) ─────────────────────────────────────────────
    elif event_type in ("status-update", "speech-update", "transcript"):
        return JSONResponse({"status": "ok"})

    else:
        logger.debug(f"Unhandled Vapi event type: {event_type}")
        return JSONResponse({"status": "ok"})


async def _handle_assistant_request(body: dict, call_id: str) -> JSONResponse:
    """Process a conversation turn and return the assistant's spoken response."""
    message_obj   = body.get("message", {})
    transcript    = message_obj.get("transcript", "")
    messages_list = message_obj.get("messages", [])

    # Get the latest human message
    if messages_list:
        human_msgs = [m for m in messages_list if m.get("role") == "user"]
        user_input = human_msgs[-1].get("content", "") if human_msgs else transcript
    else:
        user_input = transcript

    # First turn — return intro
    if not user_input.strip():
        intro = (
            f"Hi there! I'm the AI assistant representing {CANDIDATE_NAME}. "
            f"I'm here to help answer your questions about their background, skills, "
            f"and experience, and I can also schedule a follow-up interview. "
            f"What would you like to know?"
        )
        return JSONResponse({"response": intro})

    try:
        executor = _get_or_create_voice_agent(call_id, user_input)
        result   = await executor.ainvoke({"input": user_input})
        response = result.get("output", "I didn't quite catch that. Could you rephrase?")

        # Trim to keep voice responses concise
        response = _trim_voice_response(response)

        logger.info(f"Voice response [{call_id}]: {response[:100]}...")
        return JSONResponse({"response": response})

    except Exception as e:
        logger.error(f"Voice agent error [{call_id}]: {e}", exc_info=True)
        return JSONResponse({
            "response": "I'm having a brief technical issue. Could you repeat that?"
        })


def _trim_voice_response(text: str, max_sentences: int = 3) -> str:
    """Keep voice responses brief — max 3 sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    trimmed   = " ".join(sentences[:max_sentences])
    return trimmed


def _cleanup_voice_session(call_id: str, body: dict):
    """Log call summary and clean up session memory."""
    duration = body.get("message", {}).get("durationSeconds", 0)
    logger.info(f"Call ended [{call_id}] — duration: {duration}s")
    _voice_sessions.pop(call_id, None)


# ── Vapi Assistant Setup (call once to configure) ────────────────────────────
async def create_vapi_assistant(server_url: str) -> dict:
    """
    Create the Vapi assistant via API and return its ID.
    Call this once during deployment setup.

    Args:
        server_url: Your public server URL, e.g. "https://abc.ngrok.io"
    """
    import httpx
    vapi_key = os.getenv("VAPI_API_KEY")
    if not vapi_key:
        raise ValueError("VAPI_API_KEY not set")

    payload = {
        "name":              f"{CANDIDATE_NAME} - AI Persona",
        "model": {
            "provider":      "custom-llm",
            "url":           f"{server_url}/voice/webhook",
        },
        "voice": {
            "provider":      "11labs",
            "voiceId":       "rachel",   # Change to preferred ElevenLabs voice
        },
        "transcriber": {
            "provider":      "deepgram",
            "model":         "nova-2",
            "language":      "en",
        },
        "firstMessage":      "",         # We return this dynamically
        "endCallMessage":    f"Thank you for speaking with {CANDIDATE_NAME}'s AI assistant. Have a great day!",
        "serverUrl":         f"{server_url}/voice/webhook",
        "backgroundSound":   "off",
        "silenceTimeoutSeconds": 30,
        "maxDurationSeconds":    600,    # 10 min max call
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.vapi.ai/assistant",
            headers={"Authorization": f"Bearer {vapi_key}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Vapi assistant created: {data['id']}")
        return data
