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
