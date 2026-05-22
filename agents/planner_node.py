"""Lightweight heuristic question splitter. No LLM call — cheap and deterministic."""
import re
from agents.state import AgentState
from config import MAX_SUB_QUERIES

_SPLIT_RE = re.compile(r"\s+\b(?:and|also)\b\s+", flags=re.IGNORECASE)


def plan(state: AgentState) -> dict:
    q = state["question"].strip()
    parts = [p.strip() for p in _SPLIT_RE.split(q) if p.strip()]
    if len(parts) <= 1:
        return {"sub_queries": [q]}
    return {"sub_queries": parts[:MAX_SUB_QUERIES]}
