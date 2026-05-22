"""Smoke test the server module imports and exposes tools.
Full stdio integration is exercised by the E2E test in Task 13."""
import importlib

def test_server_module_exposes_app():
    mod = importlib.import_module("mcp_server.server")
    assert hasattr(mod, "app")
    # FastMCP exposes registered tools via _tool_manager or list_tools().
    assert hasattr(mod.app, "list_tools") or hasattr(mod.app, "_tool_manager")
