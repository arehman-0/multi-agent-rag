import asyncio
from agents.graph import build_graph

class StubMCP:
    async def call_tool(self, name, **kwargs):
        return [{"text": "LangGraph is a stateful workflow library.",
                 "source": "x.txt", "score": 0.9, "hash": "h1"}]

class StubLLM:
    def invoke(self, prompt):
        return type("R", (), {"content": "LangGraph is a library [1]."})()

def test_graph_runs_end_to_end():
    g = build_graph(mcp_client=StubMCP(), llm=StubLLM())
    result = asyncio.run(g.ainvoke({
        "question": "What is LangGraph?",
        "sub_queries": [], "chunks": [], "answer": "",
        "citations": [], "trace_id": "t1", "error": None,
    }))
    assert "LangGraph" in result["answer"]
    assert result["citations"] == ["x.txt"]
    assert len(result["chunks"]) == 1
