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
