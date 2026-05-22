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
    try:
        embeddings = [_embed(c) for c in chunks]
    except Exception:
        return {"status": "error", "reason": "ollama_unreachable: start with `ollama serve`"}

    ids = [hashlib.sha1(f"{p.name}:{i}:{c[:50]}".encode()).hexdigest() for i, c in enumerate(chunks)]
    metas = [{"source": p.name, "chunk_index": i} for i in range(len(chunks))]
    coll.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
    return {"status": "ok", "chunks": len(chunks), "doc_id": p.name}
