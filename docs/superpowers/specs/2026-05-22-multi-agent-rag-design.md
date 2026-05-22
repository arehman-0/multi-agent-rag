# Multi-Agent RAG Pipeline with Voice AI — Design

**Date:** 2026-05-22
**Status:** Approved for planning

## Goal

Build a local-first, two-agent Retrieval-Augmented Generation system with voice input/output, MCP-based tool integration, and OpenTelemetry tracing. Demoable end-to-end via a Streamlit UI.

## Non-goals

- Cloud LLM providers (OpenAI/Anthropic). Everything runs locally.
- Multi-tenant or auth. Single-user demo.
- Production scaling, sharding, or HA.
- Mobile/web framework beyond Streamlit.
- More than two agents.

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM | Ollama (llama3 or llama3.2 for synthesis, llama3.2:1b for tests) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | Chroma (on-disk) |
| Tool protocol | MCP (Model Context Protocol), stdio transport |
| Voice STT | `faster-whisper` (local) |
| Voice TTS | `pyttsx3` (offline, no model download) |
| UI | Streamlit |
| Observability | OpenTelemetry SDK → Jaeger (local container) |
| Language | Python 3.11+ |

## Architecture

```
┌──────────────┐     ┌───────────────────────────────────┐
│  Streamlit   │────▶│  LangGraph Orchestrator           │
│  (UI + voice)│     │                                   │
└──────────────┘     │   ┌─────────┐   ┌──────────────┐  │
       ▲             │   │Retriever│──▶│ Synthesizer  │  │
       │             │   │ Agent   │   │   Agent      │  │
       │             │   └────┬────┘   └──────┬───────┘  │
       │             └────────┼───────────────┼──────────┘
       │                      ▼               ▼
       │              ┌──────────────┐  ┌──────────┐
       └──TTS audio◀──│  MCP server  │  │  Ollama  │
                      │ (search tool)│  │ (llama3) │
                      └──────┬───────┘  └──────────┘
                             ▼
                      ┌─────────────┐
                      │   Chroma    │
                      └─────────────┘

         All edges traced via OpenTelemetry → Jaeger
```

### Agents

- **Retriever Agent (A).** Receives the user's question (and optional sub-queries from the Planner node). Calls the MCP `search_docs` tool, optionally in parallel for multi-query fan-out. Returns merged, de-duplicated chunks.
- **Synthesizer Agent (B).** Receives `{question, chunks}` from shared state. Prompts the local Ollama model with a grounded-answer template that requires citations. Streams tokens back to the graph.

The agents communicate **only** through the shared `AgentState`. They do not import each other.

### "Simultaneously" semantics

True parallelism happens in two places:

1. **Multi-query retrieval fan-out.** If the Planner node splits a compound question into N sub-queries, the Retriever issues N parallel `search_docs` calls via `asyncio.gather`.
2. **Speculative prefetch.** While the Synthesizer is still generating and the TTS is playing for the previous answer, the Retriever can begin fetching chunks for the next queued question.

This is documented honestly — pure sequential 2-agent RAG would not be parallel, and the design avoids overclaiming.

## Module layout

```
multi-agent-rag/
├── app/
│   ├── streamlit_app.py        # UI: upload, chat, voice record/playback
│   └── voice.py                # Whisper STT + pyttsx3 TTS wrappers
├── agents/
│   ├── graph.py                # LangGraph state machine
│   ├── state.py                # AgentState TypedDict
│   ├── planner_node.py         # Lightweight sub-query splitter
│   ├── retriever_agent.py      # Agent A
│   └── synthesizer_agent.py    # Agent B
├── mcp_server/
│   ├── server.py               # MCP server (stdio)
│   └── tools/
│       ├── search.py           # search_docs tool
│       └── ingest.py           # ingest_doc tool
├── observability/
│   └── tracing.py              # OTel init + Jaeger exporter
├── tests/
│   ├── test_tools.py
│   ├── test_agents.py
│   ├── test_graph_e2e.py
│   └── fixtures/
│       └── sample.pdf
├── config.py
├── docker-compose.yml          # Jaeger (and optionally Ollama)
├── requirements.txt
└── README.md
```

### Module responsibilities

- **`app/`** — only handles user I/O. Knows about Streamlit widgets, file uploads, mic recording, audio playback. Hands a question string + uploaded file to the graph. Does not touch Chroma, Ollama, or MCP directly.
- **`agents/`** — pure graph definition and node functions. Stateless over the typed `AgentState`.
- **`mcp_server/`** — runs as a subprocess via stdio. Exposes `search_docs(query: str, k: int = 4)` and `ingest_doc(path: str)` as MCP tools. Owns the Chroma client. No knowledge of LangGraph.
- **`observability/`** — single `init_tracing(service_name)` entry point. Provides `trace_span(name)` decorator for nodes.
- **`config.py`** — single source of truth for tunables.

## Data flow (per request)

1. **Ingest (one-time per doc).** Upload → `mcp_client.call_tool("ingest_doc", path)` → chunker (1000 char, 200 overlap) → embed via Ollama → upsert to Chroma. Returns doc_id.
2. **Question input.** Voice path: mic → `voice.transcribe()` (Whisper) → text. Text path: typed into Streamlit. Either way → `question: str`.
3. **Graph invocation.** `graph.invoke({"question": text, "trace_id": new_id()})`.
4. **Planner node.** If the question is compound (heuristic: contains "and"/"also"/multiple "?"), split into ≤3 sub-queries. Else pass through unchanged.
5. **Retriever Agent (parallel fan-out).** `asyncio.gather(*[mcp.call("search_docs", q, k=4) for q in sub_queries])`. Merge + de-dupe by chunk hash. Write `chunks` into state.
6. **Synthesizer Agent.** Prompt template includes `{question}` and numbered `{chunks}`. Streams Ollama tokens. Writes `answer` and `citations` to state.
7. **UI render.** Streamlit displays tokens as they stream. On completion: `voice.speak(answer)` → audio widget.
8. **Tracing.** Every step emits OTel spans under one parent trace_id propagated through state.

### AgentState shape

```python
class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    chunks: list[Chunk]   # {text, source, score, hash}
    answer: str
    citations: list[str]
    trace_id: str
    error: str | None
```

## Error handling

Boundaries only. No defensive code inside trusted internals.

- **Upload boundary.** Reject non-PDF/TXT; cap file size at 20 MB. User-facing error surfaced in UI.
- **MCP boundary.** If the MCP server is unreachable or a tool errors, Retriever returns `chunks=[], error="retrieval_failed"`. Synthesizer detects this and replies *"I couldn't find relevant info in the uploaded docs"* — never hallucinates around it.
- **Ollama boundary.** Connection refused → user-facing message: *"Ollama not running. Start it with `ollama serve` then reload."* No retry loops.
- **Voice boundary.** If Whisper fails or no mic detected, fall back silently to text input. TTS failure is non-fatal — text answer still shown.
- **Tracing.** OTel exporter init is wrapped in `try/except`. Logged once at startup if Jaeger is unreachable. Tracing must never crash a request.

## Testing strategy

- **Unit (MCP tools).** Each tool tested directly with a tmp Chroma directory. Real embedding model — no mocked vectors.
- **Unit (agents).** Retriever tested with a stub MCP client; Synthesizer tested with a stub LLM callable. Assert state-in → state-out transitions.
- **Integration (graph E2E).** Ingest a small fixture PDF, run a question with a known answer, assert the answer contains an expected substring. Uses `llama3.2:1b` to keep test runtime reasonable.
- **Trace verification.** One test asserts the expected span tree shape (Retriever → N search spans → Synthesizer → LLM span).
- **UI + voice.** Not unit-tested. Verified manually in the browser per the "test UI features in browser" rule.

## Configuration (config.py defaults)

```python
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2"
LLM_MODEL_TEST = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./.chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MAX_SUB_QUERIES = 3
MAX_UPLOAD_MB = 20
JAEGER_ENDPOINT = "http://localhost:4318/v1/traces"
SERVICE_NAME = "multi-agent-rag"
```

## Deliverables

- Working `streamlit run app/streamlit_app.py` demo.
- `docker compose up` brings Jaeger online for trace viewing.
- README with setup steps: install Ollama, pull models, install Python deps, start Jaeger, run app.
- Passing test suite (`pytest`).

## Out of scope (explicit)

- Authentication / multi-user.
- Cloud LLMs.
- Production deployment (k8s, scaling).
- Document update/delete UI (re-upload to replace).
- Conversation history beyond the current Streamlit session.
- Agent personalities, role-play, or more than 2 agents.
