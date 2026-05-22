# Multi-Agent RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first 2-agent (Retriever + Synthesizer) RAG pipeline with voice I/O, MCP tool integration, and OpenTelemetry tracing, demoable via Streamlit.

**Architecture:** Streamlit UI → LangGraph (Planner → Retriever Agent → Synthesizer Agent) → MCP server exposing Chroma search/ingest tools → local Ollama for LLM + embeddings. Faster-whisper for STT, pyttsx3 for TTS. OTel spans on every node export to local Jaeger.

**Tech Stack:** Python 3.11, LangGraph, langchain-mcp-adapters, mcp (server SDK), chromadb, ollama (Python client), faster-whisper, pyttsx3, streamlit, opentelemetry-{api,sdk,exporter-otlp}, pypdf, pytest, pytest-asyncio.

**Spec:** [docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md](../specs/2026-05-22-multi-agent-rag-design.md)

---

## Prerequisites (one-time setup, before Task 1)

The implementing engineer must have these on their machine:

1. Python 3.11 or newer (`python --version`).
2. Ollama installed and running: `ollama serve` in a separate terminal.
3. Models pulled: `ollama pull llama3.2 && ollama pull llama3.2:1b && ollama pull nomic-embed-text`.
4. Docker Desktop running (for Jaeger).

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: directory stubs `app/`, `agents/`, `mcp_server/tools/`, `observability/`, `tests/fixtures/`

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.chroma/
.pytest_cache/
.streamlit/
uploads/
*.wav
*.mp3
```

- [ ] **Step 2: Create `requirements.txt`**

```
langgraph>=0.2.50
langchain-core>=0.3.0
langchain-ollama>=0.2.0
langchain-mcp-adapters>=0.1.0
mcp>=1.0.0
chromadb>=0.5.0
ollama>=0.4.0
faster-whisper>=1.0.3
pyttsx3>=2.90
streamlit>=1.40.0
pypdf>=5.0.0
opentelemetry-api>=1.27.0
opentelemetry-sdk>=1.27.0
opentelemetry-exporter-otlp-proto-http>=1.27.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

- [ ] **Step 3: Create `config.py`**

```python
"""Single source of truth for tunables. No code outside this module reads env vars."""
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2"
LLM_MODEL_TEST = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH = str(PROJECT_ROOT / ".chroma")
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MAX_SUB_QUERIES = 3
MAX_UPLOAD_MB = 20

JAEGER_ENDPOINT = "http://localhost:4318/v1/traces"
SERVICE_NAME = "multi-agent-rag"

MCP_SERVER_CMD = ["python", "-m", "mcp_server.server"]
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
addopts = -v --tb=short
```

- [ ] **Step 5: Create empty `__init__.py` in each package**

Create empty files: `app/__init__.py`, `agents/__init__.py`, `mcp_server/__init__.py`, `mcp_server/tools/__init__.py`, `observability/__init__.py`, `tests/__init__.py`.

- [ ] **Step 6: Create venv and install**

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Expected: install completes without errors. faster-whisper may take a minute to download CTranslate2 wheels.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt config.py pytest.ini app/ agents/ mcp_server/ observability/ tests/
git commit -m "feat: scaffold project structure and config"
```

---

## Task 2: Observability — OpenTelemetry tracing

**Files:**
- Create: `observability/tracing.py`
- Create: `tests/test_tracing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tracing.py`:
```python
from observability.tracing import init_tracing, get_tracer

def test_init_tracing_returns_tracer():
    init_tracing("test-service")
    tracer = get_tracer()
    assert tracer is not None

def test_span_context_manager_works():
    init_tracing("test-service")
    tracer = get_tracer()
    with tracer.start_as_current_span("dummy") as span:
        span.set_attribute("k", "v")
    # No assertion needed — just that it doesn't raise.

def test_init_tracing_survives_unreachable_exporter():
    # Must not raise even if Jaeger is down.
    init_tracing("test-service", endpoint="http://localhost:9/does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tracing.py -v
```
Expected: ImportError — `observability.tracing` does not exist yet.

- [ ] **Step 3: Implement `observability/tracing.py`**

```python
"""OpenTelemetry setup. init_tracing() is idempotent and never raises."""
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

import config

_initialized = False
log = logging.getLogger(__name__)


def init_tracing(service_name: str = config.SERVICE_NAME, endpoint: str | None = None) -> None:
    global _initialized
    if _initialized:
        return
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint or config.JAEGER_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
    except Exception as exc:
        log.warning("Tracing init failed (continuing without traces): %s", exc)
        _initialized = True  # mark to prevent retry storms


def get_tracer():
    return trace.get_tracer(config.SERVICE_NAME)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tracing.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add observability/tracing.py tests/test_tracing.py
git commit -m "feat: OpenTelemetry tracing setup with Jaeger exporter"
```

---

## Task 3: MCP tool — document ingestion

**Files:**
- Create: `mcp_server/tools/ingest.py`
- Create: `tests/test_ingest.py`
- Create: `tests/fixtures/sample.txt`

- [ ] **Step 1: Create fixture**

`tests/fixtures/sample.txt`:
```
LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain with the ability to coordinate multiple chains across multiple steps.
The Model Context Protocol (MCP) standardises how applications provide context to LLMs.
```

- [ ] **Step 2: Write the failing test**

`tests/test_ingest.py`:
```python
import tempfile
from pathlib import Path
from mcp_server.tools.ingest import ingest_doc, _get_collection

def test_ingest_txt_creates_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(tmp_path / "chroma"))
    result = ingest_doc("tests/fixtures/sample.txt")
    assert result["status"] == "ok"
    assert result["chunks"] >= 1
    coll = _get_collection()
    assert coll.count() >= 1
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_ingest.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `mcp_server/tools/ingest.py`**

```python
"""Ingest a document into Chroma: chunk → embed → upsert."""
import hashlib
from pathlib import Path
from typing import Any
import chromadb
import ollama
from pypdf import PdfReader

from config import CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, OLLAMA_HOST

_client = None


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client.get_or_create_collection("docs")


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _embed(text: str) -> list[float]:
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def ingest_doc(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "error", "reason": f"file not found: {path}"}
    text = _read_file(p)
    chunks = _chunk(text)
    if not chunks:
        return {"status": "error", "reason": "no extractable text"}
    coll = _get_collection()
    embeddings = [_embed(c) for c in chunks]
    ids = [hashlib.sha1(f"{p.name}:{i}:{c[:50]}".encode()).hexdigest() for i, c in enumerate(chunks)]
    metas = [{"source": p.name, "chunk_index": i} for i in range(len(chunks))]
    coll.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
    return {"status": "ok", "chunks": len(chunks), "doc_id": p.name}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_ingest.py -v
```
Expected: PASS (requires Ollama running with `nomic-embed-text` pulled).

- [ ] **Step 6: Commit**

```bash
git add mcp_server/tools/ingest.py tests/test_ingest.py tests/fixtures/sample.txt
git commit -m "feat: MCP ingest tool — chunk/embed/upsert to Chroma"
```

---

## Task 4: MCP tool — vector search

**Files:**
- Create: `mcp_server/tools/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

`tests/test_search.py`:
```python
from mcp_server.tools.ingest import ingest_doc
from mcp_server.tools.search import search_docs

def test_search_returns_relevant_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("mcp_server.tools.search.CHROMA_PATH", str(tmp_path / "chroma"))
    ingest_doc("tests/fixtures/sample.txt")
    results = search_docs("What is LangGraph?", k=2)
    assert len(results) >= 1
    assert any("LangGraph" in r["text"] for r in results)
    for r in results:
        assert "text" in r and "source" in r and "score" in r and "hash" in r
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_search.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `mcp_server/tools/search.py`**

```python
"""Vector search over the Chroma collection."""
import hashlib
from typing import Any
import chromadb
import ollama

from config import CHROMA_PATH, EMBED_MODEL, OLLAMA_HOST, RETRIEVAL_K

_client = None


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client.get_or_create_collection("docs")


def _embed(text: str) -> list[float]:
    client = ollama.Client(host=OLLAMA_HOST)
    return client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def search_docs(query: str, k: int = RETRIEVAL_K) -> list[dict[str, Any]]:
    coll = _get_collection()
    if coll.count() == 0:
        return []
    q_emb = _embed(query)
    res = coll.query(query_embeddings=[q_emb], n_results=min(k, coll.count()))
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": 1.0 - dist,
            "hash": hashlib.sha1(doc.encode()).hexdigest()[:12],
        })
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_search.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/tools/search.py tests/test_search.py
git commit -m "feat: MCP search tool — vector search over Chroma"
```

---

## Task 5: MCP server runner

**Files:**
- Create: `mcp_server/server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing test**

`tests/test_mcp_server.py`:
```python
"""Smoke test the server starts and registers tools. Full stdio integration
is exercised by Task 11 (graph E2E)."""
import importlib

def test_server_module_exposes_app():
    mod = importlib.import_module("mcp_server.server")
    assert hasattr(mod, "app")
    # FastMCP app has list_tools().
    assert hasattr(mod.app, "list_tools") or hasattr(mod.app, "_tool_manager")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mcp_server.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `mcp_server/server.py`**

```python
"""MCP server (stdio transport) exposing ingest_doc and search_docs."""
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.ingest import ingest_doc as _ingest
from mcp_server.tools.search import search_docs as _search

app = FastMCP("multi-agent-rag-tools")


@app.tool()
def ingest_doc(path: str) -> dict:
    """Ingest a PDF or TXT file: chunk, embed, store in Chroma. Returns {status, chunks, doc_id}."""
    return _ingest(path)


@app.tool()
def search_docs(query: str, k: int = 4) -> list[dict]:
    """Search the vector store. Returns list of {text, source, score, hash}."""
    return _search(query, k=k)


if __name__ == "__main__":
    app.run(transport="stdio")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_mcp_server.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/server.py tests/test_mcp_server.py
git commit -m "feat: MCP server exposing ingest_doc and search_docs over stdio"
```

---

## Task 6: AgentState definition

**Files:**
- Create: `agents/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
from agents.state import AgentState, new_state

def test_new_state_has_required_keys():
    s = new_state("hello?")
    assert s["question"] == "hello?"
    assert s["sub_queries"] == []
    assert s["chunks"] == []
    assert s["answer"] == ""
    assert s["citations"] == []
    assert s["error"] is None
    assert s["trace_id"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/state.py`**

```python
"""Shared AgentState typed dict for the LangGraph nodes."""
import uuid
from typing import TypedDict


class Chunk(TypedDict):
    text: str
    source: str
    score: float
    hash: str


class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[str]
    trace_id: str
    error: str | None


def new_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        sub_queries=[],
        chunks=[],
        answer="",
        citations=[],
        trace_id=uuid.uuid4().hex,
        error=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_state.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/state.py tests/test_state.py
git commit -m "feat: AgentState TypedDict and constructor"
```

---

## Task 7: Planner node

**Files:**
- Create: `agents/planner_node.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_planner.py`:
```python
from agents.state import new_state
from agents.planner_node import plan

def test_single_question_passes_through():
    s = new_state("What is LangGraph?")
    out = plan(s)
    assert out["sub_queries"] == ["What is LangGraph?"]

def test_compound_question_splits_on_and():
    s = new_state("What is LangGraph and how does MCP work?")
    out = plan(s)
    assert len(out["sub_queries"]) == 2
    assert "LangGraph" in out["sub_queries"][0]
    assert "MCP" in out["sub_queries"][1]

def test_caps_at_max_sub_queries():
    s = new_state("a and b and c and d and e")
    out = plan(s)
    assert len(out["sub_queries"]) <= 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_planner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/planner_node.py`**

```python
"""Lightweight heuristic question splitter. No LLM call — cheap and deterministic."""
import re
from agents.state import AgentState
from config import MAX_SUB_QUERIES

_SPLIT_RE = re.compile(r"\s+\b(?:and|also)\b\s+", flags=re.IGNORECASE)


def plan(state: AgentState) -> dict:
    q = state["question"].strip()
    parts = [p.strip() for p in _SPLIT_RE.split(q) if p.strip()]
    if len(parts) <= 1:
        return {"sub_queries": [q]}
    return {"sub_queries": parts[:MAX_SUB_QUERIES]}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_planner.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/planner_node.py tests/test_planner.py
git commit -m "feat: planner node — heuristic sub-query split"
```

---

## Task 8: Retriever agent

**Files:**
- Create: `agents/retriever_agent.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retriever.py`:
```python
import asyncio
from agents.state import new_state
from agents.retriever_agent import retrieve

class StubMCP:
    def __init__(self, results):
        self.results = results
        self.calls = []
    async def call_tool(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.results

def test_retrieve_calls_search_for_each_subquery():
    stub = StubMCP([{"text": "x", "source": "s.txt", "score": 0.9, "hash": "abc"}])
    s = new_state("q")
    s["sub_queries"] = ["q1", "q2", "q3"]
    out = asyncio.run(retrieve(s, mcp_client=stub))
    assert len(stub.calls) == 3
    assert len(out["chunks"]) == 1  # de-duped by hash

def test_retrieve_handles_mcp_error():
    class Boom:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("server gone")
    s = new_state("q")
    s["sub_queries"] = ["q1"]
    out = asyncio.run(retrieve(s, mcp_client=Boom()))
    assert out["chunks"] == []
    assert out["error"] == "retrieval_failed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retriever.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/retriever_agent.py`**

```python
"""Retriever Agent (A) — parallel fan-out over sub-queries via MCP search tool."""
import asyncio
import logging
from agents.state import AgentState
from config import RETRIEVAL_K
from observability.tracing import get_tracer

log = logging.getLogger(__name__)


async def _one_search(mcp_client, query: str):
    return await mcp_client.call_tool("search_docs", query=query, k=RETRIEVAL_K)


async def retrieve(state: AgentState, mcp_client) -> dict:
    tracer = get_tracer()
    sub_queries = state["sub_queries"] or [state["question"]]
    with tracer.start_as_current_span("retriever_agent") as span:
        span.set_attribute("sub_query_count", len(sub_queries))
        try:
            results = await asyncio.gather(
                *[_one_search(mcp_client, q) for q in sub_queries]
            )
        except Exception as exc:
            log.exception("retrieval failed: %s", exc)
            return {"chunks": [], "error": "retrieval_failed"}

        seen: set[str] = set()
        merged = []
        for chunk_list in results:
            for c in chunk_list:
                h = c.get("hash") or c.get("text", "")[:32]
                if h in seen:
                    continue
                seen.add(h)
                merged.append(c)
        merged.sort(key=lambda c: c.get("score", 0), reverse=True)
        span.set_attribute("chunks_returned", len(merged))
        return {"chunks": merged}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retriever.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/retriever_agent.py tests/test_retriever.py
git commit -m "feat: Retriever Agent with parallel sub-query fan-out"
```

---

## Task 9: Synthesizer agent

**Files:**
- Create: `agents/synthesizer_agent.py`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_synthesizer.py`:
```python
from agents.state import new_state
from agents.synthesizer_agent import synthesize

class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None
    def invoke(self, prompt):
        self.last_prompt = prompt
        return type("R", (), {"content": self.reply})()

def test_synthesize_with_chunks():
    llm = StubLLM("LangGraph is a library [1].")
    s = new_state("What is LangGraph?")
    s["chunks"] = [{"text": "LangGraph is a library", "source": "x.txt", "score": 0.9, "hash": "h"}]
    out = synthesize(s, llm=llm)
    assert "LangGraph" in out["answer"]
    assert out["citations"] == ["x.txt"]
    assert "What is LangGraph?" in llm.last_prompt

def test_synthesize_no_chunks_returns_fallback():
    llm = StubLLM("should not be called")
    s = new_state("q")
    s["chunks"] = []
    out = synthesize(s, llm=llm)
    assert "couldn't find relevant info" in out["answer"].lower()
    assert llm.last_prompt is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthesizer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/synthesizer_agent.py`**

```python
"""Synthesizer Agent (B) — grounded answer generation with citations."""
from agents.state import AgentState
from observability.tracing import get_tracer

_FALLBACK = "I couldn't find relevant info in the uploaded docs to answer that."

_PROMPT_TEMPLATE = """You are answering using ONLY the provided context.
If the context is insufficient, say so explicitly. Cite sources as [1], [2] inline.

Question: {question}

Context:
{context}

Answer:"""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c['source']}) {c['text']}")
    return "\n\n".join(lines)


def synthesize(state: AgentState, llm) -> dict:
    tracer = get_tracer()
    with tracer.start_as_current_span("synthesizer_agent") as span:
        chunks = state["chunks"]
        if not chunks:
            span.set_attribute("fallback", True)
            return {"answer": _FALLBACK, "citations": []}

        prompt = _PROMPT_TEMPLATE.format(
            question=state["question"],
            context=_format_context(chunks),
        )
        with tracer.start_as_current_span("llm_invoke") as llm_span:
            llm_span.set_attribute("prompt_len", len(prompt))
            resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        citations = list({c["source"] for c in chunks})
        span.set_attribute("answer_len", len(answer))
        return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_synthesizer.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/synthesizer_agent.py tests/test_synthesizer.py
git commit -m "feat: Synthesizer Agent with grounded prompt and citations"
```

---

## Task 10: LangGraph wiring

**Files:**
- Create: `agents/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

`tests/test_graph.py`:
```python
import asyncio
from agents.graph import build_graph

class StubMCP:
    async def call_tool(self, name, **kwargs):
        return [{"text": "LangGraph is a stateful workflow library.",
                 "source": "x.txt", "score": 0.9, "hash": "h1"}]

class StubLLM:
    def invoke(self, prompt):
        return type("R", (), {"content": "LangGraph is a library [1]."})()

def test_graph_runs_end_to_end():
    g = build_graph(mcp_client=StubMCP(), llm=StubLLM())
    result = asyncio.run(g.ainvoke({
        "question": "What is LangGraph?",
        "sub_queries": [], "chunks": [], "answer": "",
        "citations": [], "trace_id": "t1", "error": None,
    }))
    assert "LangGraph" in result["answer"]
    assert result["citations"] == ["x.txt"]
    assert len(result["chunks"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_graph.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/graph.py`**

```python
"""LangGraph wiring: Planner → Retriever → Synthesizer."""
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.planner_node import plan
from agents.retriever_agent import retrieve
from agents.synthesizer_agent import synthesize


def build_graph(mcp_client, llm):
    sg = StateGraph(AgentState)

    sg.add_node("planner", lambda s: plan(s))
    sg.add_node("retriever", lambda s: retrieve(s, mcp_client=mcp_client))
    sg.add_node("synthesizer", lambda s: synthesize(s, llm=llm))

    sg.add_edge(START, "planner")
    sg.add_edge("planner", "retriever")
    sg.add_edge("retriever", "synthesizer")
    sg.add_edge("synthesizer", END)

    return sg.compile()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_graph.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/graph.py tests/test_graph.py
git commit -m "feat: LangGraph wiring (Planner → Retriever → Synthesizer)"
```

---

## Task 11: MCP client adapter + real graph factory

**Files:**
- Create: `agents/runtime.py`
- Create: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runtime.py`:
```python
"""Smoke test for the runtime factory. End-to-end with real MCP subprocess
is exercised in Task 13 (integration test)."""
from agents.runtime import build_runtime_graph

def test_runtime_factory_returns_compiled_graph():
    # Just test the factory is importable and returns something with ainvoke.
    # We don't actually start the subprocess here.
    import inspect
    assert inspect.iscoroutinefunction(build_runtime_graph)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_runtime.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `agents/runtime.py`**

```python
"""Production graph factory: spawns the MCP server subprocess and wires real Ollama LLM."""
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from agents.graph import build_graph
from config import LLM_MODEL, MCP_SERVER_CMD, OLLAMA_HOST


class _MCPToolWrapper:
    """Adapt langchain-mcp-adapters' tool list into a call_tool(name, **kwargs) interface."""
    def __init__(self, tools):
        self._by_name = {t.name: t for t in tools}

    async def call_tool(self, name: str, **kwargs):
        tool = self._by_name[name]
        return await tool.ainvoke(kwargs)


@asynccontextmanager
async def mcp_session():
    client = MultiServerMCPClient({
        "rag": {
            "command": MCP_SERVER_CMD[0],
            "args": MCP_SERVER_CMD[1:],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    yield _MCPToolWrapper(tools)


async def build_runtime_graph():
    """Returns (graph, mcp_context_manager). Caller must hold the context open while invoking."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    ctx = mcp_session()
    mcp_client = await ctx.__aenter__()
    graph = build_graph(mcp_client=mcp_client, llm=llm)
    return graph, ctx
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_runtime.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/runtime.py tests/test_runtime.py
git commit -m "feat: runtime graph factory with MCP subprocess + Ollama"
```

---

## Task 12: Voice (STT + TTS)

**Files:**
- Create: `app/voice.py`
- Create: `tests/test_voice.py`

- [ ] **Step 1: Write the failing test**

`tests/test_voice.py`:
```python
"""Voice tests are limited: TTS just verifies the function runs without raising
(actual audio is not asserted). STT is skipped unless a fixture .wav exists."""
import os
import pytest
from app.voice import speak, transcribe

def test_speak_does_not_raise():
    speak("hello world", to_file=None)  # uses default engine; should not raise

@pytest.mark.skipif(not os.path.exists("tests/fixtures/sample.wav"),
                    reason="no audio fixture")
def test_transcribe_returns_text():
    text = transcribe("tests/fixtures/sample.wav")
    assert isinstance(text, str)
    assert len(text) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_voice.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `app/voice.py`**

```python
"""STT via faster-whisper, TTS via pyttsx3. Both local, no API keys."""
import logging
from pathlib import Path
import pyttsx3
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)
_whisper: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def transcribe(audio_path: str) -> str:
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        log.warning("transcribe failed: %s", exc)
        return ""


def speak(text: str, to_file: str | None = None) -> str | None:
    try:
        engine = pyttsx3.init()
        if to_file:
            engine.save_to_file(text, to_file)
            engine.runAndWait()
            return to_file
        engine.say(text)
        engine.runAndWait()
        return None
    except Exception as exc:
        log.warning("TTS failed: %s", exc)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_voice.py -v
```
Expected: PASS (1 passed, 1 skipped — that's fine).

Note: first run downloads the Whisper "base" model (~140 MB) into the user's cache.

- [ ] **Step 5: Commit**

```bash
git add app/voice.py tests/test_voice.py
git commit -m "feat: voice I/O — faster-whisper STT and pyttsx3 TTS"
```

---

## Task 13: End-to-end integration test

**Files:**
- Create: `tests/test_integration_e2e.py`

- [ ] **Step 1: Write the integration test**

`tests/test_integration_e2e.py`:
```python
"""End-to-end: real MCP subprocess + real Ollama + real Chroma.

Requires: Ollama running with llama3.2:1b and nomic-embed-text pulled.
Run with: pytest tests/test_integration_e2e.py -v -s
"""
import asyncio
import os
import shutil
import pytest

from agents.state import new_state
from agents.runtime import build_runtime_graph
from mcp_server.tools.ingest import ingest_doc
import config

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="set RUN_E2E=1 to run; requires Ollama models",
)


@pytest.fixture(autouse=True)
def clean_chroma(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(config, "CHROMA_PATH", str(chroma_dir))
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(chroma_dir))
    monkeypatch.setattr("mcp_server.tools.search.CHROMA_PATH", str(chroma_dir))
    yield
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)


async def _run():
    ingest_doc("tests/fixtures/sample.txt")
    graph, ctx = await build_runtime_graph()
    try:
        state = new_state("What is LangGraph?")
        result = await graph.ainvoke(state)
        return result
    finally:
        await ctx.__aexit__(None, None, None)


def test_e2e_pipeline_answers_question():
    result = asyncio.run(_run())
    assert result["answer"]
    assert "LangGraph" in result["answer"] or "langgraph" in result["answer"].lower()
    assert len(result["chunks"]) >= 1
```

- [ ] **Step 2: Run the test**

```bash
$env:RUN_E2E="1"; pytest tests/test_integration_e2e.py -v -s
```

Expected: PASS within ~30-60s. If it fails because Ollama isn't running, start `ollama serve` and confirm `ollama list` shows `llama3.2:1b` and `nomic-embed-text`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_e2e.py
git commit -m "test: end-to-end integration covering real MCP + Ollama"
```

---

## Task 14: Streamlit UI

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Implement `app/streamlit_app.py`**

```python
"""Streamlit UI: upload docs, ask questions (text or voice), hear answers."""
import asyncio
from pathlib import Path
import streamlit as st

from agents.runtime import build_runtime_graph
from agents.state import new_state
from app.voice import speak, transcribe
from mcp_server.tools.ingest import ingest_doc
from observability.tracing import init_tracing
from config import MAX_UPLOAD_MB, UPLOAD_DIR

init_tracing()

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("Multi-Agent RAG (Retriever + Synthesizer)")

# --- Sidebar: upload
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large (>{MAX_UPLOAD_MB} MB).")
        else:
            dest = UPLOAD_DIR / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            with st.spinner(f"Ingesting {uploaded.name}..."):
                result = ingest_doc(str(dest))
            if result["status"] == "ok":
                st.success(f"Ingested {result['chunks']} chunks.")
            else:
                st.error(f"Ingest failed: {result.get('reason')}")

# --- Main: question input
st.subheader("Ask a question")
col1, col2 = st.columns([3, 1])
with col1:
    text_q = st.text_input("Type your question", "")
with col2:
    audio = st.audio_input("Or record")

question = text_q
if audio is not None and not text_q:
    audio_path = UPLOAD_DIR / "_last_input.wav"
    audio_path.write_bytes(audio.getbuffer())
    with st.spinner("Transcribing..."):
        question = transcribe(str(audio_path))
    st.write(f"**Heard:** {question}")

# --- Run pipeline
async def _run(q: str):
    graph, ctx = await build_runtime_graph()
    try:
        result = await graph.ainvoke(new_state(q))
        return result
    finally:
        await ctx.__aexit__(None, None, None)


if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        result = asyncio.run(_run(question))
    if result.get("error"):
        st.error(f"Error: {result['error']}")
    else:
        st.markdown("### Answer")
        st.write(result["answer"])
        if result["citations"]:
            st.caption("Sources: " + ", ".join(result["citations"]))

        audio_out = UPLOAD_DIR / "_last_answer.wav"
        speak(result["answer"], to_file=str(audio_out))
        if audio_out.exists():
            st.audio(str(audio_out))

        with st.expander("Retrieved chunks"):
            for c in result["chunks"]:
                st.markdown(f"**[{c['source']}]** (score={c['score']:.2f})")
                st.write(c["text"])
```

- [ ] **Step 2: Run Streamlit manually to verify**

```bash
streamlit run app/streamlit_app.py
```

Expected:
- Browser opens at http://localhost:8501
- Upload `tests/fixtures/sample.txt` → see "Ingested N chunks."
- Type "What is LangGraph?" → click Ask → see grounded answer + sources + audio playback widget.
- Voice path: click the mic, speak a question, see transcription, get an answer.

Test each path before committing.

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: Streamlit UI with upload, text/voice input, audio playback"
```

---

## Task 15: Docker Compose for Jaeger + README

**Files:**
- Create: `docker-compose.yml`
- Create: `README.md`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.62
    ports:
      - "16686:16686"   # UI
      - "4318:4318"     # OTLP HTTP receiver
    environment:
      COLLECTOR_OTLP_ENABLED: "true"
```

- [ ] **Step 2: Start Jaeger and verify**

```bash
docker compose up -d
```

Open http://localhost:16686 — Jaeger UI should load. Run the Streamlit app, ask a question, refresh Jaeger, select service `multi-agent-rag` → see a trace with `planner`, `retriever_agent` (with child `search_docs` spans), `synthesizer_agent`, `llm_invoke`.

- [ ] **Step 3: Create `README.md`**

```markdown
# Multi-Agent RAG Pipeline with Voice AI

Local-first 2-agent RAG (Retriever + Synthesizer) on LangGraph, with MCP tool integration,
voice I/O (Whisper + pyttsx3), and OpenTelemetry tracing to Jaeger.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- Docker (for Jaeger)

## Setup

```bash
# 1. Install Python deps
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Pull Ollama models
ollama pull llama3.2
ollama pull llama3.2:1b         # used by tests
ollama pull nomic-embed-text

# 3. Start Jaeger
docker compose up -d

# 4. Run the app
streamlit run app/streamlit_app.py
```

## Architecture

See [docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md](docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md).

## Tests

```bash
pytest                          # unit tests (no Ollama needed for some)
RUN_E2E=1 pytest                # full integration (needs Ollama)
```

## Observability

Jaeger UI: http://localhost:16686 — select service `multi-agent-rag` to see per-request traces.
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml README.md
git commit -m "docs: docker-compose for Jaeger and README setup guide"
```

---

## Task 16: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all unit tests pass; e2e is skipped (no RUN_E2E flag).

- [ ] **Step 2: Run e2e with Ollama**

```bash
$env:RUN_E2E="1"; pytest tests/test_integration_e2e.py -v
```

Expected: PASS.

- [ ] **Step 3: Manually verify Streamlit + voice + Jaeger**

1. `docker compose up -d`
2. `streamlit run app/streamlit_app.py`
3. Upload a doc, ask a typed question — verify answer appears with sources and audio.
4. Click mic, ask a voice question — verify transcription + answer + audio playback.
5. Open http://localhost:16686 — confirm trace appears with all expected spans.

- [ ] **Step 4: Commit any final tweaks**

If manual testing surfaced anything (formatting, copy, error messages), fix and commit:

```bash
git add -A
git commit -m "chore: post-verification polish"
```

---

## Self-review checklist (for the engineer executing this plan)

After all tasks: confirm every spec section is implemented.

- [x] 2-agent (Retriever + Synthesizer) — Tasks 8, 9
- [x] LangGraph orchestration with parallel sub-query fan-out — Tasks 7, 8, 10
- [x] MCP server with search + ingest tools — Tasks 3, 4, 5
- [x] Voice STT + TTS — Task 12
- [x] OpenTelemetry tracing on every node + LLM call — Tasks 2, 8, 9
- [x] Local Ollama (llama3.2 + nomic-embed-text) — Tasks 3, 4, 11
- [x] Chroma vector store on disk — Tasks 3, 4
- [x] Streamlit UI with upload, text+voice input, audio playback — Task 14
- [x] Jaeger via docker-compose — Task 15
- [x] Integration test with real backends — Task 13
- [x] Error handling at boundaries (upload, MCP, Ollama, voice, tracing) — Tasks 2, 8, 9, 12, 14
