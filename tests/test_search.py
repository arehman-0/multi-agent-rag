from tests.conftest import skip_if_no_ollama
from mcp_server.tools.ingest import ingest_doc
from mcp_server.tools.search import search_docs

@skip_if_no_ollama
def test_search_returns_relevant_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("mcp_server.tools.ingest._client", None)
    monkeypatch.setattr("mcp_server.tools.search.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("mcp_server.tools.search._client", None)
    ingest_doc("tests/fixtures/sample.txt")
    results = search_docs("What is LangGraph?", k=2)
    assert len(results) >= 1
    assert any("LangGraph" in r["text"] for r in results)
    for r in results:
        assert "text" in r and "source" in r and "score" in r and "hash" in r
