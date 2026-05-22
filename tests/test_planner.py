from agents.state import new_state
from agents.planner_node import plan

def test_single_question_passes_through():
    s = new_state("What is LangGraph?")
    out = plan(s)
    assert out["sub_queries"] == ["What is LangGraph?"]

def test_compound_question_splits_on_and():
    s = new_state("What is LangGraph and how does MCP work?")
    out = plan(s)
    assert len(out["sub_queries"]) == 2
    assert "LangGraph" in out["sub_queries"][0]
    assert "MCP" in out["sub_queries"][1]

def test_caps_at_max_sub_queries():
    s = new_state("a and b and c and d and e")
    out = plan(s)
    assert len(out["sub_queries"]) <= 3
