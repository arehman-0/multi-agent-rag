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
