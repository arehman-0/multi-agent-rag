"""Streamlit UI: upload docs, ask questions (text or voice), hear answers."""
import asyncio
from pathlib import Path
import streamlit as st

from agents.runtime import build_runtime_graph
from agents.state import new_state
from app.voice import speak, transcribe
from observability.tracing import init_tracing
from config import MAX_UPLOAD_MB, UPLOAD_DIR

init_tracing()

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("Multi-Agent RAG (Retriever + Synthesizer)")


@st.cache_resource
def _get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _runtime():
    """Build or fetch the cached graph + MCP client + context for this session."""
    if "runtime" not in st.session_state:
        loop = _get_event_loop()
        graph, mcp_client, ctx = loop.run_until_complete(build_runtime_graph())
        st.session_state["runtime"] = (graph, mcp_client, ctx)
        st.session_state["loop"] = loop
    return st.session_state["runtime"]


# --- Sidebar: upload
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large (>{MAX_UPLOAD_MB} MB).")
        else:
            dest = UPLOAD_DIR / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            try:
                _, mcp_client, _ = _runtime()
                loop = st.session_state["loop"]
                with st.spinner(f"Ingesting {uploaded.name}..."):
                    result = loop.run_until_complete(
                        mcp_client.call_tool("ingest_doc", path=str(dest))
                    )
                if isinstance(result, dict) and result.get("status") == "ok":
                    st.success(f"Ingested {result['chunks']} chunks.")
                elif isinstance(result, dict):
                    st.error(f"Ingest failed: {result.get('reason', 'unknown')}")
                else:
                    # Tool returned a non-dict (e.g. a ToolMessage string); treat as ok
                    st.success(f"Ingested {uploaded.name}.")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

# --- Main: question input
st.subheader("Ask a question")
col1, col2 = st.columns([3, 1])
with col1:
    text_q = st.text_input("Type your question", "")
with col2:
    audio = st.audio_input("Or record")

question = text_q
if audio is not None and not text_q:
    audio_path = UPLOAD_DIR / "_last_input.wav"
    audio_path.write_bytes(audio.getbuffer())
    with st.spinner("Transcribing..."):
        question = transcribe(str(audio_path))
    st.write(f"**Heard:** {question}")

# --- Run pipeline
async def _ask(q: str):
    graph, _, _ = _runtime()
    return await graph.ainvoke(new_state(q))


if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        loop = _get_event_loop()
        result = loop.run_until_complete(_ask(question))
    if result.get("error"):
        st.error(f"Error: {result['error']}")
    else:
        st.markdown("### Answer")
        st.write(result["answer"])
        if result["citations"]:
            st.caption("Sources: " + ", ".join(result["citations"]))

        audio_out = UPLOAD_DIR / "_last_answer.wav"
        speak(result["answer"], to_file=str(audio_out))
        if audio_out.exists():
            st.audio(str(audio_out))

        with st.expander("Retrieved chunks"):
            for c in result["chunks"]:
                st.markdown(f"**[{c['source']}]** (score={c['score']:.2f})")
                st.write(c["text"])
