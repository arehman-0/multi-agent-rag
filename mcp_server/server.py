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
