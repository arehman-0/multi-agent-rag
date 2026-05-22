from agents.state import AgentState, new_state

def test_new_state_has_required_keys():
    s = new_state("hello?")
    assert s["question"] == "hello?"
    assert s["sub_queries"] == []
    assert s["chunks"] == []
    assert s["answer"] == ""
    assert s["citations"] == []
    assert s["error"] is None
    assert s["trace_id"]
