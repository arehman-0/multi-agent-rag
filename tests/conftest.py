"""Shared test helpers — Ollama reachability check used by tests that need real embeddings."""
import urllib.request
import pytest


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5).read()
        return True
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not _ollama_up(),
    reason="Ollama not running at localhost:11434 — start with `ollama serve`",
)
