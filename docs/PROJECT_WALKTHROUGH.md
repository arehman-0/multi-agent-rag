# Multi-Agent RAG Pipeline — Project Walkthrough & Interview Guide

This document explains what the project does, how it was built, what went wrong while building it, and how to talk about it confidently in an interview.

---

## Part 1 — What the project does (in one paragraph)

A user uploads a PDF or text document through a Streamlit web UI. They can either type a question or record one with their microphone. The question goes through a LangGraph state machine where two specialized agents collaborate: **Agent A (Retriever)** searches the document chunks stored in a Chroma vector database, fanning out multiple parallel searches when the question is compound; **Agent B (Synthesizer)** takes the retrieved chunks and prompts a local Ollama LLM to write a grounded, cited answer. The answer is shown on screen, spoken aloud through text-to-speech, and every step is traced with OpenTelemetry so you can see the full execution waterfall in Jaeger. Everything runs locally — no cloud APIs.

---

## Part 2 — Architecture (what talks to what)

```
┌──────────────┐      ┌───────────────────────────────────┐
│  Streamlit   │─────▶│  LangGraph Orchestrator           │
│  (UI + voice)│      │                                   │
└──────────────┘      │  Planner ─▶ Retriever ─▶ Synthr.  │
       ▲              └────────┬───────────────┬──────────┘
       │                       ▼               ▼
       │             ┌──────────────────┐  ┌──────────┐
       └──TTS audio◀─│  MCP server      │  │  Ollama  │
                     │  (stdio process) │  │ (LLM +   │
                     │                  │  │  embeds) │
                     │  search_docs     │  └──────────┘
                     │  ingest_doc      │
                     └──────┬───────────┘
                            ▼
                     ┌─────────────┐
                     │   Chroma    │
                     │ (vector DB) │
                     └─────────────┘

     All steps emit OpenTelemetry spans → Jaeger UI on :16686
```

**Key boundaries:**

- `app/` only handles UI and voice. It does **not** import Chroma or Ollama directly. It talks to the MCP client for tools and to the LangGraph for orchestration.
- `agents/` only knows about the typed state and the two-agent + planner workflow. It receives an `mcp_client` and an `llm` as dependencies — it does not construct them.
- `mcp_server/` exposes tools over stdio. The Chroma client lives here. It does not import LangGraph.
- `observability/` is one tiny module. Every other module imports `get_tracer()` from it but doesn't know about Jaeger.

This separation means each layer can be tested independently with stubs.

---

## Part 3 — How the build was done (the workflow)

The project followed a strict three-phase workflow, with each phase committed before the next started:

### Phase 1 — Brainstorming (design)

Open-ended questions narrowed the scope: which two agents (Retriever + Synthesizer), which LLM backend (local Ollama), what UI (Streamlit), what doc source (user upload), what to include in v1 (everything — RAG + MCP + voice + tracing). The result is [`docs/superpowers/specs/2026-05-22-multi-agent-rag-design.md`](superpowers/specs/2026-05-22-multi-agent-rag-design.md).

### Phase 2 — Planning

The spec was turned into a 16-task implementation plan — each task with exact file paths, a failing test, the implementation code, the command to run, the expected output, and the commit message. See [`docs/superpowers/plans/2026-05-22-multi-agent-rag.md`](superpowers/plans/2026-05-22-multi-agent-rag.md).

### Phase 3 — Subagent-driven execution

Each of the 16 tasks was implemented by a fresh subagent, followed by spec-compliance and code-quality review. TDD throughout: failing test first, then implementation, then commit. A final cross-cutting code review at the end caught three critical issues that were fixed in follow-up commits.

**Why this matters:** the final repo has 21 atomic commits where each one is testable in isolation. If anything breaks later, you can `git bisect` to the exact change. No giant "wrote the whole project" commit.

---

## Part 4 — Module-by-module walkthrough

### `config.py`
Single source of truth for tunables (model names, paths, chunk size, K, endpoints). The only module that reads environment variables. `CHROMA_PATH` is read from `os.environ` with a default — this is critical for the MCP subprocess (see "Hard Problems" below).

### `agents/state.py`
A `TypedDict` called `AgentState` holds the running state of one request: `question`, `sub_queries`, `chunks`, `answer`, `citations`, `trace_id`, `error`. `new_state(question)` constructs an empty one with a fresh UUID trace_id.

### `agents/planner_node.py`
Deterministic, no LLM. Splits a compound question on the regex `\s+\b(?:and|also)\b\s+`, capped at 3 sub-queries. *"What is LangGraph and how does MCP work?"* → `["What is LangGraph", "how does MCP work?"]`.

### `agents/retriever_agent.py` (Agent A)
For each sub-query, fires a parallel call to the MCP `search_docs` tool via `asyncio.gather`. Merges and de-duplicates results by chunk hash, sorts by score. Wraps everything in an OTel span. On MCP failure, returns `chunks=[], error="retrieval_failed"` — never crashes the graph.

### `agents/synthesizer_agent.py` (Agent B)
Formats retrieved chunks into a numbered context block, prompts the LLM with a grounded-answer template (`"answer using ONLY the provided context"`), extracts the source list as citations. Catches Ollama connection errors and returns a user-facing message instead of crashing.

### `agents/graph.py`
Wires the three nodes into a LangGraph `StateGraph`: `START → planner → retriever → synthesizer → END`. The retriever node is an `async def` (LangGraph auto-awaits async nodes — a sync lambda that *returns* a coroutine does **not** work, that was a real bug we hit).

### `agents/runtime.py`
The production wiring: spins up the MCP server subprocess via `MultiServerMCPClient` with stdio transport, creates a `ChatOllama` instance, returns `(graph, mcp_client, ctx)`. Forwards `CHROMA_PATH` to the subprocess via the `env` field.

### `mcp_server/server.py`
A FastMCP app exposing two tools: `ingest_doc(path)` and `search_docs(query, k)`. Runs as `python -m mcp_server.server` with `transport="stdio"`.

### `mcp_server/tools/ingest.py`
Reads PDF (via `pypdf`) or TXT, chunks (1000 chars, 200 overlap), embeds each chunk via Ollama's `nomic-embed-text`, upserts into Chroma. Returns `{status, chunks, doc_id}` or a structured error if Ollama is unreachable.

### `mcp_server/tools/search.py`
Embeds the query, runs a similarity search against Chroma, returns the top-k chunks as `{text, source, score, hash}` dicts. Returns `[]` if Ollama is down — the synthesizer's empty-chunks fallback handles it.

### `observability/tracing.py`
`init_tracing()` sets up an OTel `TracerProvider` with a `BatchSpanProcessor` and `OTLPSpanExporter` pointed at Jaeger's HTTP endpoint on :4318. Wrapped in `try/except` and idempotent — if Jaeger isn't running, it logs a warning and the app continues without tracing instead of crashing.

### `app/voice.py`
`transcribe(audio_path)` uses `faster-whisper` with the "base" model (CPU, int8) — lazily loaded singleton so the model file is only read once. `speak(text, to_file)` uses `pyttsx3` (Windows SAPI5 / macOS NSSpeechSynthesizer / Linux espeak — no API key, no network). Both functions return safe defaults on failure rather than raising.

### `app/streamlit_app.py`
The UI. On first interaction it builds the runtime graph and caches it in `st.session_state`. Upload routes through `mcp_client.call_tool("ingest_doc", path=...)`. Question routes through `graph.ainvoke(new_state(q))`. Renders answer + citations + audio playback + a collapsible "retrieved chunks" expander.

---

## Part 5 — Hard problems we hit and how we solved them

These are the moments where the build stopped and we had to think. They're the most useful material for interview questions.

### Problem 1: LangGraph silently failed on async nodes wrapped in sync lambdas

**Symptom:** `InvalidUpdateError: Expected dict, got <coroutine object>`.

**What happened:** We registered the retriever node as `sg.add_node("retriever", lambda s: retrieve(s, mcp_client=mcp_client))`. The lambda is synchronous but `retrieve` is `async def`, so calling it returns a coroutine object — LangGraph then tried to use that coroutine as the state update.

**Fix:** Define an explicit `async def retriever_node(s)` inside `build_graph` that `await`s the call. LangGraph correctly detects and awaits proper async callables.

**Lesson:** "Returns a coroutine" and "is awaitable by the framework" are not the same thing. Async-aware frameworks usually inspect the callable itself, not what it returns.

### Problem 2: The MCP subprocess didn't see test fixtures

**Symptom:** The E2E test wrote chunks to a temp Chroma dir but the MCP server's `search_docs` returned empty results.

**What happened:** The test used `monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", tmp_dir)`. That mutates an attribute *in the current Python process*. But `build_runtime_graph` spawns the MCP server as a **separate OS process** via `python -m mcp_server.server`. The subprocess imports `config.py` fresh and reads the default path. So we had a writer (in-process ingest) pointed at `/tmp/...` and a reader (subprocess search) pointed at `.chroma`.

**Fix:** Made `config.py` read `CHROMA_PATH` from `os.environ` with a fallback default. The MCP client's `StdioConnection.env` field forwards the env var to the subprocess. The test now uses `monkeypatch.setenv("CHROMA_PATH", tmp_dir)`, which is honored by both processes.

**Lesson:** Mocking and monkeypatching only reach into your own process. Anything you spawn is a clean slate. When the boundary is a subprocess, the only contract is what you pass through stdin, env vars, or files on disk.

### Problem 3: Ollama-down crashed the whole app

**Symptom:** Stop Ollama, ask a question, see a Python traceback in the Streamlit UI.

**What happened:** `synthesizer_agent.py` called `llm.invoke(prompt)` with no error handling. When Ollama wasn't running, `ChatOllama` raised a `ConnectionError`. That propagated up through the graph and out of the Streamlit handler. Same problem in the ingest path — `_embed()` raised and broke uploads.

**Fix:** Wrap every Ollama call at the boundary. Synthesizer catches the exception and returns a structured response: `{"answer": "Ollama not running. Start it with ollama serve...", "citations": []}`. Ingest returns `{"status": "error", "reason": "ollama_unreachable: ..."}`. The UI checks for the error and renders a friendly message.

**Lesson:** The boundary between your code and an external service is the **only** place you defend. Inside the trusted layers (the agents, the graph, the UI logic), no try/except. At the network boundary, always.

### Problem 4: Streamlit was bypassing the MCP boundary

**Symptom:** Code review flagged that `streamlit_app.py` imported `ingest_doc` directly from `mcp_server.tools.ingest`. This created a second `chromadb.PersistentClient` in the Streamlit process while the MCP subprocess had its own. Two clients writing to the same on-disk dir is asking for file-lock conflicts.

**Fix:** Refactored to cache `(graph, mcp_client, ctx)` in `st.session_state` on first use. Upload now goes through `mcp_client.call_tool("ingest_doc", path=...)` — the same channel as search. One Chroma client per process, one process owns the database.

**Lesson:** Module boundaries on paper are not boundaries in practice unless your code respects them. A direct import looks innocent but breaks process isolation.

### Problem 5: The OpenTelemetry exporter timeout could kill the request

**Symptom:** When Jaeger wasn't running, span export attempts kept retrying and printing scary HTTP errors.

**What happened:** `OTLPSpanExporter` defaults to retrying connection failures. If Jaeger was down, this could delay the process exit and pollute logs.

**Fix:** Wrap `init_tracing` in `try/except`. Set `_initialized = True` even on failure so we don't keep retrying. Tracing failure is logged once and the app keeps working.

**Lesson:** Observability infrastructure must never affect application availability. If your tracing is on the critical path of a request, you've built it wrong.

### Problem 6: Dependencies took a long time and one component needed a model download

**Symptom:** First test run failed because `faster-whisper` needed to download its model file. Pip install of the full stack (LangGraph, langchain-ollama, chromadb, faster-whisper, opentelemetry, streamlit, pypdf...) took several minutes.

**Fix:** Lazy-load the Whisper model the first time `transcribe()` is called (singleton pattern). Acknowledge in the README that first run downloads ~140 MB.

**Lesson:** First-run latency is a real UX problem. Lazy initialization and a README that warns about it goes a long way.

---

## Part 6 — Interview story (use these as templates)

Below are framed answers to common interview questions about this project. They follow the **STAR** pattern (Situation, Task, Action, Result). Practice these out loud — natural delivery matters more than memorization.

### Q: "Tell me about a project you're proud of."

> I built a local-first multi-agent Retrieval-Augmented Generation pipeline with voice I/O. The user uploads a document, asks a question by typing or speaking, and two specialized agents work together — one retrieves relevant passages from a vector database, the other writes a grounded answer with citations. Everything runs locally on Ollama, with OpenTelemetry tracing so I can see the full execution path in Jaeger.
>
> What I'm most proud of is how disciplined the build was. I started with a written design spec, turned it into a 16-task plan where each task had a failing test before implementation, and ended up with 21 atomic commits. Every component is independently testable. When the final code review flagged three critical issues — like the MCP subprocess not seeing my test config, or Ollama-down crashing the app — I could fix each one in a focused commit without touching anything else. That's the kind of codebase I want to ship.

### Q: "What was the hardest problem you ran into?"

> The hardest one was a subtle process-boundary bug in the integration test. I had a tool server running in a separate Python subprocess, talking to the main app over stdio. My test was setting up a temporary Chroma database using pytest's `monkeypatch.setattr`, and everything passed locally — except the answer always came back empty.
>
> It took me a while to realize: `monkeypatch.setattr` mutates an attribute in *my* Python process. The subprocess was a completely separate OS process. It imported `config.py` fresh and read the default path. So I was writing chunks to `/tmp/...` but searching `.chroma`. Two databases, looked the same in code, totally disconnected at runtime.
>
> The fix was to switch `CHROMA_PATH` to read from `os.environ`, and forward the env var to the subprocess through the MCP client's `env` field. Both processes now agree on the same path.
>
> The lesson I took away: mocking and monkeypatching only reach into your own process. When you cross a subprocess boundary, the only contract is what you pass through stdin, env vars, or files. It's the same principle as crossing a network boundary.

### Q: "Describe a time you got something wrong and had to fix it."

> Halfway through, I had the Streamlit UI calling the ingest function directly — `from mcp_server.tools.ingest import ingest_doc`. It worked, my tests passed, the demo ran. A code review flagged it.
>
> The issue was that I was creating two Chroma clients pointing at the same on-disk directory — one in the Streamlit process, one in the MCP subprocess. Chroma's file locking would probably prevent corruption, but it was a latent bug waiting to happen, and it violated the architecture boundary I'd written into the spec.
>
> I refactored the UI to route both upload and query through a single MCP client cached in `st.session_state`. One Chroma client per process, one process owns the database. It was about three files of changes.
>
> What I learned was that "I wrote it down in the design" and "the code actually respects it" are two different things. A direct import looks innocent but it broke process isolation in a way I couldn't see from reading any single file. Now I treat module boundaries as something to enforce in the code review, not just in the spec.

### Q: "Why did you choose those specific technologies?"

> Three big choices.
>
> **LangGraph over plain LangChain chains** because I needed real orchestration: parallel sub-query fan-out, explicit shared state, and the option to add cycles or human-in-the-loop later. LangGraph's `StateGraph` gives you a typed state dict that nodes mutate explicitly — much easier to reason about than chain composition.
>
> **MCP for tool integration**, even though I could have just imported the functions, because MCP is the emerging standard for how LLM apps talk to tools, and decoupling tool implementation from the agents means the same tools could be exposed to Claude, ChatGPT, or any MCP client. It's portable.
>
> **Local Ollama** instead of OpenAI or Anthropic because the project is meant to be demoable without API keys, runs offline, and the resume bullet specifically calls out local capability. Ollama with `llama3.2` and `nomic-embed-text` is good enough for grounded Q&A and gives the demo a privacy story.

### Q: "How did you think about testing?"

> I used real dependencies for the data layer and stubs for the agents.
>
> The Chroma store uses a real on-disk database in a `tmp_path` for each test. I don't mock the vector store — mocks of databases hide the actual bugs you care about. The unit tests are gated by an Ollama-reachable check, so they skip cleanly on machines without it instead of failing falsely.
>
> The agents themselves get tested with stub LLM and stub MCP clients. That's the right place to mock — the agent's contract is "given this state, transition to that state," and a stub lets me assert the transition without paying for an LLM call.
>
> The end-to-end test wires up everything for real — real MCP subprocess, real Ollama, real Chroma — and is gated by a `RUN_E2E=1` environment variable so it doesn't run in fast feedback loops but does run when I want full validation.

### Q: "What would you do differently if you started over?"

> Three things.
>
> First, I'd wire **token streaming** from the start. Right now the Streamlit UI shows a spinner until the synthesizer finishes — fine for short answers, frustrating for long ones. LangGraph's `astream_events` plus Streamlit's `st.write_stream` would let tokens render as they arrive. I deferred it because the spec said v1 doesn't need it, but if I were starting fresh I'd build it in.
>
> Second, I'd **link the trace_id to the OpenTelemetry parent context**. Each request has a `trace_id` in its AgentState, but I'm not propagating it as the OTel context parent across nodes. That means in Jaeger, individual spans show up but cross-request correlation doesn't work. It's a one-day fix and I should have caught it earlier.
>
> Third, I'd write the **span-tree assertion test** that I mentioned in the spec — a test that uses OTel's `InMemorySpanExporter` to assert "this request emitted exactly these spans in this parent-child shape." That would have caught the trace_id issue automatically.
>
> Those three things are all in my notes as follow-ups, not blockers.

### Q: "What did you learn from this project?"

> A few things, in rough order of importance.
>
> **Boundaries are everything.** Once I had clean module boundaries — UI doesn't touch Chroma, agents don't touch the network, the MCP server doesn't know about LangGraph — every bug had an obvious home and every fix was contained. The moments things broke were exactly the moments boundaries leaked.
>
> **TDD with subagent dispatch is fast.** Writing the failing test first sounds slow, but when each task is bite-sized — test, implement, commit — the loop becomes very tight. The whole build was 21 commits with each one independently green.
>
> **Process boundaries are a special kind of boundary.** They look like module boundaries in your code but behave like network boundaries at runtime. Subprocess + env var + stdio is a different mental model than function call + import.
>
> **Defensive code belongs at the edges, not the middle.** Try/except inside trusted agent logic is noise that hides bugs. Try/except at the Ollama boundary or the upload boundary is essential. Knowing which is which is most of code quality.

---

## Part 7 — Quick numbers (for "what did you build?" answers)

- **Lines of code:** ~700 lines across 12 modules.
- **Tests:** 16 passing, 4 skipped (Ollama-dependent + E2E + audio fixture).
- **Commits:** 21 atomic commits, TDD throughout.
- **Tasks in plan:** 16, each with failing test → implementation → commit.
- **Critical issues caught and fixed in final review:** 3 (Ollama-crash, subprocess env propagation, Streamlit boundary).
- **External dependencies:** LangGraph, langchain-ollama, langchain-mcp-adapters, mcp, chromadb, ollama, faster-whisper, pyttsx3, streamlit, pypdf, opentelemetry-{api,sdk,exporter-otlp}, pytest, pytest-asyncio.

---

## Part 8 — How to demo it live (if asked)

1. `ollama serve` — start Ollama in one terminal.
2. `ollama pull llama3.2 && ollama pull nomic-embed-text` — first time only.
3. `docker compose up -d` — bring up Jaeger.
4. `.venv/Scripts/Activate.ps1 && streamlit run app/streamlit_app.py` — launch the UI.
5. Drop a PDF into the sidebar. Type or speak a question. Watch the answer appear with citations. Listen to the spoken answer.
6. Open http://localhost:16686 — pick service `multi-agent-rag` — show the trace waterfall with planner / retriever / search_docs / synthesizer / llm_invoke spans.

If anything fails on stage, the most likely culprits are:
- Ollama not running → the UI will show *"Ollama not running. Start it with `ollama serve`."*
- Wrong model not pulled → ingest will return a structured error.
- Jaeger down → the app still works, traces just won't show. Tracing is non-fatal by design.
