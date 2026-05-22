"""Shared AgentState typed dict for the LangGraph nodes."""
import uuid
from typing import TypedDict


class Chunk(TypedDict):
    text: str
    source: str
    score: float
    hash: str


class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[str]
    trace_id: str
    error: str | None


def new_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        sub_queries=[],
        chunks=[],
        answer="",
        citations=[],
        trace_id=uuid.uuid4().hex,
        error=None,
    )
