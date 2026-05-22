"""Single source of truth for tunables. No code outside this module reads env vars."""
import os
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2"
LLM_MODEL_TEST = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH = os.environ.get("CHROMA_PATH", str(PROJECT_ROOT / ".chroma"))
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MAX_SUB_QUERIES = 3
MAX_UPLOAD_MB = 20

JAEGER_ENDPOINT = "http://localhost:4318/v1/traces"
SERVICE_NAME = "multi-agent-rag"

MCP_SERVER_CMD = ["python", "-m", "mcp_server.server"]
