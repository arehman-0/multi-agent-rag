"""End-to-end: real MCP subprocess + real Ollama + real Chroma.

Requires: Ollama running with llama3.2:1b and nomic-embed-text pulled.
Run with: RUN_E2E=1 pytest tests/test_integration_e2e.py -v -s
"""
import asyncio
import importlib
import os
import shutil
import pytest

from agents.state import new_state
from agents.runtime import build_runtime_graph
from mcp_server.tools.ingest import ingest_doc
from config import LLM_MODEL_TEST

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="set RUN_E2E=1 to run; requires Ollama models",
)


@pytest.fixture(autouse=True)
def clean_chroma(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma"
    # Set env var so config.py re-reads it fresh (for subprocesses and reloads).
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))
    # Also patch module-level attributes so in-process code picks up the tmp dir.
    import config as _cfg
    importlib.reload(_cfg)
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(chroma_dir))
    monkeypatch.setattr("mcp_server.tools.ingest._client", None)
    monkeypatch.setattr("mcp_server.tools.search.CHROMA_PATH", str(chroma_dir))
    monkeypatch.setattr("mcp_server.tools.search._client", None)
    yield
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)


async def _run():
    ingest_doc("tests/fixtures/sample.txt")
    graph, mcp_client, ctx = await build_runtime_graph(model=LLM_MODEL_TEST)
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
