"""Smoke test for the runtime factory. End-to-end with real MCP subprocess
is exercised in Task 13 (integration test)."""
from agents.runtime import build_runtime_graph

def test_runtime_factory_is_coroutine():
    import inspect
    assert inspect.iscoroutinefunction(build_runtime_graph)
