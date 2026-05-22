from tests.conftest import skip_if_no_ollama
from mcp_server.tools.ingest import ingest_doc, _get_collection

@skip_if_no_ollama
def test_ingest_txt_creates_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.ingest.CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("mcp_server.tools.ingest._client", None)
    result = ingest_doc("tests/fixtures/sample.txt")
    assert result["status"] == "ok"
    assert result["chunks"] >= 1
    coll = _get_collection()
    assert coll.count() >= 1
