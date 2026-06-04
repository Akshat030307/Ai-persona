# AI Persona — Candidate Representative System

> **Scaler AI Engineer Screening Assignment** — A fully autonomous AI persona that answers recruiter questions over voice and chat, and books real calendar interviews with no human in the loop.

**Live URLs**
- 💬 Chat: `https://web-production-bfcd0.up.railway.app`
- 📞 Voice: `+1 (239) 663 4085`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RECRUITER                                 │
│               calls phone number  /  opens chat URL                │
└──────────────────┬──────────────────────────┬───────────────────────┘
                   │                          │
          ┌────────▼────────┐       ┌─────────▼─────────┐
          │   VAPI           │       │   Chat UI          │
          │  Telephony       │       │  (index.html)      │
          │  Deepgram ASR    │       │  Railway static    │
          │  Azure TTS       │       └─────────┬──────────┘
          └────────┬─────────┘                 │
                   │ POST                      │ POST
                   │ /voice/webhook            │ /chat/message
                   │ /chat/completions         │
                   └──────────────┬────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │           FastAPI Server (Python)         │
              │                                           │
              │  ┌────────────────────────────────────┐  │
              │  │   LangChain AgentExecutor           │  │
              │  │   GPT-4o + OpenAI Tool Calling      │  │
              │  └──────────┬──────────────┬───────────┘  │
              │             │              │               │
              │   ┌─────────▼───┐  ┌───────▼───────────┐  │
              │   │  RAG Chain  │  │  Calendar Tools    │  │
              │   │  per-turn   │  │  check_availability│  │
              │   │  retrieval  │  │  book_meeting      │  │
              │   └─────────┬───┘  └───────┬────────────┘  │
              └─────────────┼──────────────┼───────────────┘
                            │              │
                 ┌──────────▼───┐  ┌───────▼──────────┐
                 │   ChromaDB   │  │  Google Calendar  │
                 │  (local,     │  │  API (freebusy,   │
                 │  persisted)  │  │  events.insert)   │
                 └──────────────┘  └──────────────────-┘
                       ▲
                       │  one-time ingestion
              ┌────────┴──────────────────┐
              │  Resume PDF (15 chunks)   │
              │  GitHub READMEs           │
              │  Repo summaries           │
              │  Commit history           │
              │  Total: 132 chunks        │
              └───────────────────────────┘
```

---

## Project Structure

```
ai-persona/
├── app/
│   ├── main.py                  # FastAPI app, CORS, frontend serving
│   ├── rag/
│   │   ├── ingest.py            # Resume + GitHub → Chroma ingestion
│   │   └── chain.py             # ConversationalRetrievalChain + sessions
│   ├── voice/
│   │   └── vapi_webhook.py      # Vapi custom-llm /chat/completions handler
│   ├── chat/
│   │   └── router.py            # Chat API — per-turn RAG + manual memory
│   └── calendar/
│       ├── gcal.py              # Google Calendar auth, slots, booking
│       └── tools.py             # LangChain Tools + voice email cleaner
├── frontend/
│   └── index.html               # Chat UI — dark editorial + 3D wireframe
├── scripts/
│   └── setup_vapi.py            # One-time Vapi assistant creation
├── tests/
│   └── eval_chat.py             # GPT-4o judge eval — 28 golden Q&A
├── data/
│   ├── resume/resume.pdf        # Candidate resume (not in git)
│   └── chroma_db/               # Persisted vector store (committed)
├── startup.sh                   # Railway startup — writes auth files from env
├── Procfile                     # web: bash startup.sh
├── runtime.txt                  # python-3.11.9
├── mise.toml                    # Disable Python attestation check
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Vapi account (free — $10 credits on signup)
- Google Cloud project with Calendar API enabled
- Resume as PDF

### Step 1 — Install
```bash
git clone https://github.com/Akshat030307/Ai-persona.git
cd Ai-persona
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Step 2 — Configure
```bash
cp .env.example .env
# Fill in all values — see .env.example for details
```

### Step 3 — Google Calendar OAuth (run once locally)
```bash
# Download credentials JSON from Google Cloud Console
# Save as data/google_credentials.json
python -m app.calendar.gcal --auth
# Opens browser → authorize → saves data/google_token.json
```

### Step 4 — Ingest knowledge base
```bash
# Place resume at data/resume/resume.pdf
python -m app.rag.ingest
# Output: Ingestion complete. 132 chunks indexed.
```

### Step 5 — Run locally
```bash
uvicorn app.main:app --reload --port 8000
# Chat UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Step 6 — Set up Vapi voice agent
```bash
# Need public URL first (ngrok or deployed)
python scripts/setup_vapi.py --server-url https://your-url.railway.app
# Output: phone number + assistant ID
```

### Step 7 — Deploy to Railway
```bash
git add .
git commit -m "deploy"
git push origin master
# Set all env vars in Railway dashboard → Variables
# Add GOOGLE_CREDENTIALS_B64, GOOGLE_TOKEN_B64, RESUME_PDF_B64 (base64 encoded)
```

Generate base64 values locally:
```bash
python -c "import base64; print(base64.b64encode(open('data/google_credentials.json','rb').read()).decode())"
python -c "import base64; print(base64.b64encode(open('data/google_token.json','rb').read()).decode())"
python -c "import base64; print(base64.b64encode(open('data/resume/resume.pdf','rb').read()).decode())"
```

---

## Key Design Decisions

**1. Per-turn RAG context refresh**
Context is retrieved fresh for every message rather than cached per session. This prevents stale retrieval — if turn 1 asks about calendar, turn 2 asking about ELIRA still gets the right GitHub chunks. Cost impact is negligible (~$0.0001/message).

**2. Vapi custom-llm via `/chat/completions`**
Vapi's `custom-llm` provider sends requests to `{serverUrl}/chat/completions` in OpenAI format, not a simple webhook. Our server implements this endpoint, parses the message array, runs the RAG agent, and returns an OpenAI-format response.

**3. Manual session memory over LangChain memory objects**
Rebuilding the AgentExecutor every turn (needed for fresh RAG context) caused double-saving with LangChain's `ConversationBufferWindowMemory`. Replaced with a plain Python dict of `{session_id: [{human, ai}, ...]}`, injected as `HumanMessage`/`AIMessage` objects directly into the prompt.

**4. Voice email cleaning**
Deepgram transcribes spoken emails as `"akshat at the rate of gmail dot com"` or `"a k s h a t 2 0 0 6"`. A `_clean_email()` function normalises these patterns before passing to the Calendar API.

**5. Chroma committed to repo**
Instead of re-running ingestion on every Railway deploy, the `chroma_db/` folder is committed to git. Cold start loads the vector store in ~200ms. Re-run `python -m app.rag.ingest` locally and push to update.

---

## Cost Breakdown

### Per Voice Call (5-minute interview)

| Component | Usage | Est. Cost |
|-----------|-------|-----------|
| OpenAI GPT-4o (~1.5K tokens × 6 turns) | ~9K tokens | $0.045 |
| OpenAI Embeddings (6 queries × ~300 tokens) | ~1.8K tokens | $0.0002 |
| Deepgram Nova-2 ASR (5 min audio) | 5 min | $0.022 |
| Azure TTS via Vapi | 5 min | $0.015 |
| Vapi platform fee | 5 min @ ~$0.05/min | $0.025 |
| **Total per call** | | **~$0.11** |

### Per Chat Session (10-message conversation)

| Component | Usage | Est. Cost |
|-----------|-------|-----------|
| OpenAI GPT-4o (~1K tokens × 10 turns) | ~10K tokens | $0.050 |
| OpenAI Embeddings (10 queries × ~300 tokens) | ~3K tokens | $0.0003 |
| Chroma (local) | — | $0.00 |
| Railway hosting | shared across sessions | ~$0.001 |
| **Total per session** | | **~$0.05** |

### Monthly estimate (100 calls + 500 chat sessions)
| | Cost |
|---|---|
| Voice (100 × $0.11) | $11 |
| Chat (500 × $0.05) | $25 |
| Railway (Hobby plan) | $5 |
| **Total** | **~$41/month** |

---

## Eval Results (Part C)

| Metric | Result |
|--------|--------|
| Voice first-response latency | ~400ms server-side / ~1.4s end-to-end |
| Transcription WER | < 15% (Deepgram Nova-2 on clear audio) |
| Booking success rate | 4/5 (80%) |
| Hallucination rate | 7.1% (2/28 questions) |
| Retrieval keyword hit rate | 71.4% (20/28 questions) |
| Injection resistance | 4/4 (100%) |
| Average quality score | 3.96 / 5.0 |

See `tests/eval_chat.py` for the full golden Q&A set and judge methodology.

---

## Environment Variables

See `.env.example` for all required variables. Key ones:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | GPT-4o + embeddings |
| `VAPI_API_KEY` | Vapi private key |
| `VAPI_PHONE_NUMBER_ID` | Vapi phone number UUID |
| `GITHUB_TOKEN` | For RAG ingestion of repos |
| `GOOGLE_CREDENTIALS_PATH` | OAuth2 credentials JSON |
| `GOOGLE_TOKEN_PATH` | OAuth2 token (auto-generated) |
| `CANDIDATE_NAME` | Injected into all prompts |
| `BOOKING_TIMEZONE` | e.g. `Asia/Kolkata` |
| `GOOGLE_CREDENTIALS_B64` | Base64 credentials for Railway |
| `GOOGLE_TOKEN_B64` | Base64 token for Railway |