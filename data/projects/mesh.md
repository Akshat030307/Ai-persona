---
title: mesh
slug: mesh
order: 7
tags: [AI Agents, Marketplace, Full-Stack, Workflow Builder]
stack: [Next.js, FastAPI, PostgreSQL, Docker, Caddy, E2B, OpenAI API, Razorpay]
github_url: https://github.com/Akshat030307/mesh
drive_video_url: "https://drive.google.com/file/d/1BCzhOyU-mlHkvn1yvjXhFhWmdYAFDFHp/view?usp=sharing"
summary: An AI agent marketplace with a visual, n8n-style builder — developers wire and publish agents, businesses browse, run, buy, or post what they need built.
---
mesh is an AI agent marketplace with a visual builder. Developers wire agents together on an
n8n-style node canvas (Studio) and publish them; businesses browse, run, and buy published agents,
or post what they need built and receive proposals from developers (Requests). Three surfaces, one
codebase: Marketplace (`/`), Studio (`/studio` → `/build/:id`), and Requests (`/requirements`).

**Architecture:** Caddy terminates TLS and reverse-proxies everything to Next.js, which proxies
`/api/*` to FastAPI through a route handler rather than a `next.config` rewrite — rewrite
destinations are resolved at build time and get baked into `routes-manifest.json`, so putting the
internal API URL in Docker Compose's runtime environment would silently do nothing. `POST
/agents/:id/run` only inserts a queued `Run` row and returns immediately; a separate `worker`
process polls for queued runs and executes them, using `SELECT ... FOR UPDATE SKIP LOCKED` so
multiple worker replicas can run without two of them claiming the same run.

**Execution engine** (`backend/app/engine/`): a template resolver for `{{ nodes.x.field }}`-style
references that preserves native types when a string is *exactly* one reference (so a referenced
list stays a list, not stringified JSON); a topological-sort executor with cycle detection where a
node is skipped once every incoming edge is dead, letting branch pruning propagate in a single
pass; and a registry of seven node types (trigger, LLM prompt, HTTP call, sandboxed Python, template
transform, branch, output) that's a backend-only change to extend.

**Cost control** was a first-class design constraint, since the whole thing is meant to run free on
one hobby-tier VPS: one E2B sandbox per *run* (not per node), opened lazily and killed in a
`finally` block; configurable ceilings for runs/day, tokens/run, wall-clock seconds, and max nodes;
and a cumulative free-USD budget per user computed from real token usage against a per-model
pricing table, after which that user's graphs fall back to their own OpenAI key instead of the
shared one.

**Payments** run through Razorpay with HMAC-verified webhooks as the authoritative grant, plus a
manual-UPI fallback (seller shares a UPI ID, buyer pays outside the app, seller confirms receipt)
for the gap before Razorpay KYC is done — explicitly documented as a stopgap, not a permanent
payment rail.

Three offline test suites cover the engine, payments, and worker queue (topological ordering,
cycle rejection, branch pruning, the free-budget gate, FIFO queue draining) without needing
network, a database, or API keys. The README is also explicit about what's deliberately not built
yet: seller payout/split-settlement, bookmarks, and real computed ratings.
