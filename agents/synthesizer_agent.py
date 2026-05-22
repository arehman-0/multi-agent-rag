"""Synthesizer Agent (B) — grounded answer generation with citations."""
from agents.state import AgentState
from observability.tracing import get_tracer

_FALLBACK = "I couldn't find relevant info in the uploaded docs to answer that."

_PROMPT_TEMPLATE = """You are answering using ONLY the provided context.
If the context is insufficient, say so explicitly. Cite sources as [1], [2] inline.

Question: {question}

Context:
{context}

Answer:"""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c['source']}) {c['text']}")
    return "\n\n".join(lines)


def synthesize(state: AgentState, llm) -> dict:
    tracer = get_tracer()
    with tracer.start_as_current_span("synthesizer_agent") as span:
        chunks = state["chunks"]
        if not chunks:
            span.set_attribute("fallback", True)
            return {"answer": _FALLBACK, "citations": []}

        prompt = _PROMPT_TEMPLATE.format(
            question=state["question"],
            context=_format_context(chunks),
        )
        with tracer.start_as_current_span("llm_invoke") as llm_span:
            llm_span.set_attribute("prompt_len", len(prompt))
            resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        citations = list({c["source"] for c in chunks})
        span.set_attribute("answer_len", len(answer))
        return {"answer": answer, "citations": citations}
