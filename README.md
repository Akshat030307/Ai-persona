# AI Persona — Candidate Representative System

> **Scaler AI Engineer Screening Assignment** — A fully autonomous AI persona that answers recruiter questions over voice and chat, and books real interviews on your calendar.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         RECRUITER                           │
│              (calls phone / opens chat URL)                 │
└───────────────────┬──────────────────┬──────────────────────┘
                    │                  │
             ┌──────▼──────┐   ┌───────▼───────┐
             │    VAPI     │   │  Chat UI      │
             │  (Telephony │   │  (index.html) │
             │  ASR + TTS) │   └───────┬───────┘
             └──────┬──────┘           │
                    │ POST /voice/     │ POST /chat/
                    │ webhook          │ message
                    └────────┬─────────┘
                             │
            ┌────────────────▼────────────────────┐
            │        FastAPI Server (Python)       │
            │                                      │
            │  ┌──────────────────────────────┐    │
            │  │   LangChain Agent (GPT-4o)   │    │
            │  │   + OpenAI Tool Calling      │    │
            │  └────────┬──────────┬──────────┘    │
            │           │          │                │
            │  ┌────────▼───┐  ┌───▼────────────┐  │
            │  │  RAG Chain │  │ Calendar Tools │  │
            │  │  (Chroma)  │  │ (Google Cal.)  │  │
            │  └────────┬───┘  └───┬────────────┘  │
            │           │          │                │
            └───────────┼──────────┼────────────────┘
                        │          │
               ┌────────▼───┐  ┌───▼──────────┐
               │  ChromaDB  │  │ Google Cal.  │
               │  (local)   │  │    API       │
               └────────────┘  └──────────────┘
                    ▲
                    │ one-time ingestion
               ┌────┴──────────────────┐
               │   Resume PDF          │
               │   GitHub READMEs      │
               │   Commit History      │
               └───────────────────────┘
```

## Project Structure

```
ai-persona/
├── app/
│   ├── main.py              # FastAPI app, CORS, lifespan
│   ├── rag/
│   │   ├── ingest.py        # Resume + GitHub ingestion → Chroma
│   │   └── chain.py         # ConversationalRetrievalChain + session store
│   ├── voice/
│   │   └── vapi_webhook.py  # Vapi webhook handler + agent per call
│   ├── chat/
│   │   └── router.py        # Chat API endpoints + streaming
│   └── calendar/
│       ├── gcal.py          # Google Calendar auth, slots, booking
│       └── tools.py         # LangChain Tools wrapping gcal.py
├── frontend/
│   └── index.html           # Chat UI (vanilla JS, no framework)
├── scripts/
│   └── setup_vapi.py        # One-time Vapi assistant setup
├── data/
│   ├── resume/              # Place resume.pdf here
│   └── chroma_db/           # Auto-created by ingest.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (Step by Step)

### Prerequisites

- Python 3.11+
- OpenAI API key
- Vapi account (free tier — $10 credits on signup)
- Google Cloud project with Calendar API enabled
- Your resume as a PDF
- A public server URL (use [ngrok](https://ngrok.com) for local dev)

---

### Step 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-persona.git
cd ai-persona
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Fill in all values in .env
```

**Required values:**

| Key | Where to get it |
|-----|----------------|
| `OPENAI_API_KEY` | platform.openai.com |
| `VAPI_API_KEY` | dashboard.vapi.ai → API Keys |
| `VAPI_PHONE_NUMBER_ID` | Vapi dashboard → Phone Numbers → buy one |
| `GITHUB_TOKEN` | github.com/settings/tokens (read:public_repo) |
| `GITHUB_USERNAME` | your GitHub username |
| `GOOGLE_CREDENTIALS_PATH` | See Step 3 |
| `CANDIDATE_NAME` | Your full name |
| `CANDIDATE_EMAIL` | Your email |
| `RESUME_PATH` | Path to your resume PDF |

---

### Step 3 — Google Calendar OAuth2

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **Google Calendar API**
3. Credentials → Create OAuth 2.0 Client ID (Desktop app)
4. Download the JSON → save as `data/google_credentials.json`
5. Run the auth flow once locally:

```bash
python -m app.calendar.gcal --auth
# Opens browser → authorize → saves data/google_token.json
```

---

### Step 4 — Add Resume & Ingest

```bash
# Place your resume at data/resume/resume.pdf
python -m app.rag.ingest
# Output: "Ingestion complete. N chunks indexed."
```

---

### Step 5 — Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

For a public URL during development:
```bash
ngrok http 8000
# Note the https://xxxx.ngrok.io URL
```

---

### Step 6 — Set Up Vapi

```bash
python scripts/setup_vapi.py --server-url https://xxxx.ngrok.io
# Output: phone number to submit + assistant ID saved to .env
```

---

### Step 7 — Test

```bash
# Chat
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Why should we hire this candidate?"}'

# Check availability
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Check availability for a 30 min call this week"}'

# Voice: call the phone number from setup_vapi.py output
```

---

## Deployment (Production)

### Option A — Railway (Recommended, free tier available)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up

# Set env vars
railway variables set OPENAI_API_KEY=sk-... VAPI_API_KEY=... # etc.

# Get your public URL from Railway dashboard
# Re-run: python scripts/setup_vapi.py --server-url https://your-app.railway.app
```

### Option B — Fly.io

```bash
fly launch
fly secrets set OPENAI_API_KEY=sk-... VAPI_API_KEY=...
fly deploy
```

### Serve the Frontend

The `frontend/index.html` can be served as a static file by FastAPI:

```python
# Add to app/main.py
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
```

Or deploy to Vercel / Netlify separately. Update `API_BASE` in `index.html`.

---

## Cost Breakdown

### Per Voice Call (5-minute interview)

| Component | Usage | Est. Cost |
|-----------|-------|-----------|
| OpenAI GPT-4o (~2K tokens × 5 turns) | 10K tokens | $0.05 |
| Deepgram Nova-2 ASR (5 min) | 5 min | $0.02 |
| ElevenLabs TTS (~500 chars × 5) | 2,500 chars | $0.04 |
| Vapi platform fee | 5 min | $0.03 |
| **Total per call** | | **~$0.14** |

### Per Chat Session (10-message session)

| Component | Usage | Est. Cost |
|-----------|-------|-----------|
| OpenAI GPT-4o (~1K tokens × 10 turns) | 10K tokens | $0.05 |
| OpenAI Embeddings (10 queries) | ~5K tokens | $0.001 |
| Chroma (local) | — | $0.00 |
| **Total per session** | | **~$0.05** |

### Monthly estimate (100 calls + 500 chat sessions)
- Voice: 100 × $0.14 = **$14**
- Chat:  500 × $0.05 = **$25**
- **Total ≈ $39/month**

---

## Key Design Decisions

1. **Single FastAPI server for both voice and chat** — simplifies deployment, one URL to keep live.

2. **RAG on every turn (with context refresh)** — instead of stuffing the full knowledge base into the system prompt, we retrieve the top-6 most relevant chunks per question. This keeps costs low and answers more precise.

3. **LangChain AgentExecutor with tool calling** — the LLM decides when to call `check_availability` and `book_meeting` based on conversation context. No hand-written intent detection.

4. **Session memory (k=10 turns)** — keeps conversation context without blowing up the context window. Keyed by `session_id` (call_id for voice, UUID for chat).

5. **Chroma local over Pinecone** — zero latency overhead for retrieval, no API cost, persists to disk. Sufficient for a single-candidate knowledge base (~500–1000 chunks).

6. **`max_tokens=200` for voice responses** — enforces concise answers for phone calls. Chat uses full token budget.

---

## Evals

See `tests/` for the golden Q&A set and hallucination eval scripts.

Run evals:
```bash
python -m tests.eval_chat      # hallucination rate + retrieval quality
python -m tests.eval_voice     # (manual test calls + latency logging)
```
