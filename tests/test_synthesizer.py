from agents.state import new_state
from agents.synthesizer_agent import synthesize

class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None
    def invoke(self, prompt):
        self.last_prompt = prompt
        return type("R", (), {"content": self.reply})()

def test_synthesize_with_chunks():
    llm = StubLLM("LangGraph is a library [1].")
    s = new_state("What is LangGraph?")
    s["chunks"] = [{"text": "LangGraph is a library", "source": "x.txt", "score": 0.9, "hash": "h"}]
    out = synthesize(s, llm=llm)
    assert "LangGraph" in out["answer"]
    assert out["citations"] == ["x.txt"]
    assert "What is LangGraph?" in llm.last_prompt

def test_synthesize_no_chunks_returns_fallback():
    llm = StubLLM("should not be called")
    s = new_state("q")
    s["chunks"] = []
    out = synthesize(s, llm=llm)
    assert "couldn't find relevant info" in out["answer"].lower()
    assert llm.last_prompt is None

def test_synthesize_handles_llm_connection_error():
    class BrokenLLM:
        def invoke(self, prompt):
            raise ConnectionError("ollama down")
    s = new_state("q")
    s["chunks"] = [{"text": "x", "source": "s", "score": 0.9, "hash": "h"}]
    out = synthesize(s, llm=BrokenLLM())
    assert "ollama" in out["answer"].lower()
    assert out["citations"] == []
