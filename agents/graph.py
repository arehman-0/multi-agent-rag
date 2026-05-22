"""LangGraph wiring: Planner → Retriever → Synthesizer."""
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.planner_node import plan
from agents.retriever_agent import retrieve
from agents.synthesizer_agent import synthesize


def build_graph(mcp_client, llm):
    sg = StateGraph(AgentState)

    async def retriever_node(s):
        return await retrieve(s, mcp_client=mcp_client)

    sg.add_node("planner", lambda s: plan(s))
    sg.add_node("retriever", retriever_node)
    sg.add_node("synthesizer", lambda s: synthesize(s, llm=llm))

    sg.add_edge(START, "planner")
    sg.add_edge("planner", "retriever")
    sg.add_edge("retriever", "synthesizer")
    sg.add_edge("synthesizer", END)

    return sg.compile()
