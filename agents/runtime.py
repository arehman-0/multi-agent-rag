"""Production graph factory: spawns the MCP server subprocess and wires real Ollama LLM."""
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from agents.graph import build_graph
from config import LLM_MODEL, MCP_SERVER_CMD, OLLAMA_HOST


class _MCPToolWrapper:
    """Adapt langchain-mcp-adapters' tool list into a call_tool(name, **kwargs) interface."""
    def __init__(self, tools):
        self._by_name = {t.name: t for t in tools}

    async def call_tool(self, name: str, **kwargs):
        tool = self._by_name[name]
        return await tool.ainvoke(kwargs)


@asynccontextmanager
async def mcp_session():
    client = MultiServerMCPClient({
        "rag": {
            "command": MCP_SERVER_CMD[0],
            "args": MCP_SERVER_CMD[1:],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    yield _MCPToolWrapper(tools)


async def build_runtime_graph():
    """Returns (graph, mcp_context_manager). Caller must hold the context open while invoking."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    ctx = mcp_session()
    mcp_client = await ctx.__aenter__()
    graph = build_graph(mcp_client=mcp_client, llm=llm)
    return graph, ctx
