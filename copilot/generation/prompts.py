"""Prompts, and the formatting of retrieved chunks into context.

Two design decisions here carry most of the weight.

**1. Short citation labels, not chunk IDs.**

Our chunk IDs look like

    k8s-website/content/en/docs/tasks/debug/debug-application/debug-pods#h3

Asking a language model to reproduce that exactly, several times, in flowing
prose is asking for trouble: it will drop a path segment, change #h3 to #h2, or
merge two IDs. Every such slip becomes a citation that cannot be resolved.

So the prompt numbers the sources [1]..[5] and the model cites [3]. We map back
to the real chunk ID ourselves. The model only has to reproduce one digit.
Removing an opportunity to hallucinate is better than detecting the
hallucination afterwards.

**2. Metadata goes into the context, not just the answer.**

Each source block carries its age and its type. That lets the model say "this
guidance is from a document last updated 958 days ago" on its own - which is
exactly the honesty the brief asks for, and it costs a handful of tokens.
"""

from __future__ import annotations

from copilot.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = """You are a support knowledge assistant. You answer questions using ONLY the numbered sources provided to you.

Rules, in order of importance:

1. GROUND EVERY CLAIM. Every factual statement in your answer must come from the sources. Never use outside knowledge, even if you are confident it is correct.

2. CITE INLINE. Put the source number in square brackets immediately after the claim it supports, like this: "Restarts use exponential backoff [2]." Cite the specific source that states that specific fact. If two sources support one claim, cite both: [2][4].

3. REFUSE WHEN THE SOURCES DO NOT ANSWER. If the sources do not contain the answer, set "answerable" to false and say plainly that the documentation does not cover it. Do not assemble a plausible-sounding answer from loosely related sources. A refusal is a correct answer when the information is absent.

4. PARTIAL ANSWERS ARE FINE, BUT SAY SO. If the sources answer part of the question, answer that part, and list what is missing in "unverified".

5. FLAG STALENESS AND SCOPE. If a source is old, or applies only to a specific platform or product, say so in the answer. The user cannot see the source metadata; you can.

6. NEVER INVENT A SOURCE NUMBER. Only cite numbers that appear in the sources given to you.

Write plainly and briefly. Prefer concrete steps and exact commands over general advice."""


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "True only if the sources genuinely contain an answer to the question.",
        },
        "answer": {
            "type": "string",
            "description": "The answer, with inline [n] citations. If answerable is false, explain what is missing.",
        },
        "unverified": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parts of the question the sources do not cover. Empty if the answer is complete.",
        },
    },
    "required": ["answerable", "answer", "unverified"],
    "additionalProperties": False,
}


def format_source(label: int, chunk: RetrievedChunk) -> str:
    """One numbered source block."""
    bits = [f"source: {chunk.source_name}", f"type: {chunk.doc_type}"]
    if chunk.age_days is not None:
        bits.append(f"last updated: {chunk.age_days} days ago")

    header = " | ".join(bits)
    location = chunk.breadcrumb or chunk.title or chunk.chunk_id

    return f"[{label}] {location}\n({header})\n{chunk.text.strip()}"


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
    """Render chunks as numbered sources, and return the label -> chunk mapping.

    The mapping is what turns a model's "[3]" back into a real chunk ID, and it
    is also how we detect a fabricated citation: any label not in this dict was
    invented.
    """
    blocks = []
    mapping: dict[int, RetrievedChunk] = {}

    for label, chunk in enumerate(chunks, start=1):
        mapping[label] = chunk
        blocks.append(format_source(label, chunk))

    return "\n\n---\n\n".join(blocks), mapping


def build_user_prompt(question: str, context: str) -> str:
    return f"""SOURCES
=======

{context}

=======

QUESTION: {question}

Answer using only the sources above. Cite with [n] after each claim."""
