# Multi-Agent RAG Pipeline with Voice AI

Local-first 2-agent RAG (Retriever + Synthesizer) on LangGraph, with MCP tool integration,
voice I/O (Whisper + pyttsx3), and OpenTelemetry tracing to Jaeger.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- Docker (for Jaeger)

## Setup

```bash
# 1. Install Python deps
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 2. Start Ollama and pull models
ollama serve                    # in a separate terminal
ollama pull llama3.2
ollama pull llama3.2:1b         # used by tests
ollama pull nomic-embed-text

# 3. Start Jaeger
docker compose up -d

# 4. Run the app
streamlit run app/streamlit_app.py
```

The Streamlit UI opens at http://localhost:8501.
Jaeger UI: http://localhost:16686 — service `multi-agent-rag` to see per-request traces.

## Architecture

Two agents on LangGraph:
- **Retriever Agent (A)** — calls the MCP `search_docs` tool, fans out in parallel over sub-queries.
- **Synthesizer Agent (B)** — prompts Ollama with retrieved chunks and produces a grounded answer with citations.

MCP server runs as a stdio subprocess and exposes `ingest_doc` and `search_docs` tools backed by Chroma + Ollama embeddings.

Full design: [docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md](docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md).

## Tests

```bash
pytest                          # unit tests (Ollama-dependent tests skip cleanly)
$env:RUN_E2E="1"; pytest        # PowerShell — full integration (needs Ollama running)
RUN_E2E=1 pytest                # bash equivalent
```

## Repo layout

```
app/                    Streamlit UI + voice wrappers
agents/                 LangGraph nodes + graph wiring + runtime factory
mcp_server/             FastMCP server exposing search/ingest tools
observability/          OpenTelemetry setup
tests/                  Unit and integration tests
docs/superpowers/       Design spec and implementation plan
```
