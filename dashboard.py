#!/usr/bin/env python
"""Streamlit dashboard for the support copilot.

    python serve.py                          # terminal 1: the API
    streamlit run dashboard.py               # terminal 2: this

WHY THIS TALKS TO THE API INSTEAD OF IMPORTING THE PIPELINE
-----------------------------------------------------------
It would be fewer moving parts to `from copilot.generation.pipeline import Copilot`
and call it directly. Two reasons not to.

First, Streamlit re-runs this entire script top to bottom on every single
interaction - every keystroke in a text box, every checkbox. Importing the
pipeline here means the embedder, the 43 MB BM25 index and the cross-encoder all
load inside the UI process, and Streamlit's caching becomes the only thing
standing between you and a 30-second reload on every click.

Second, it is the honest architecture. The API is the product; the dashboard is
one client of it. Wiring the UI straight into the internals would let the two
drift apart, and would hide the fact that the contract in Step 5 is what a real
consumer actually gets.

STREAMLIT'S EXECUTION MODEL, BRIEFLY
------------------------------------
There is no event handler and no component tree. Streamlit runs the script again
from line 1 on every interaction and diffs the output. Consequences that shape
the code below:

  - anything expensive must be cached, or it repeats on every keystroke
  - anything that must survive a rerun goes in st.session_state
  - "the last answer" is not a variable, it is state you have to store
"""

from __future__ import annotations

import time

import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"
MODES = ["rerank", "hybrid", "dense", "sparse"]

VERDICT_STYLE = {
    "supported": ("✅", "verified"),
    "partial": ("⚠️", "partially supported"),
    "unsupported": ("❌", "does NOT support the claim"),
    "unverifiable": ("➖", "no claim attached"),
}

st.set_page_config(page_title="Support Knowledge Copilot", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
# ttl=10 on the health check: fresh enough to notice the API dying, cached enough
# that it is not re-requested on every keystroke.

@st.cache_data(ttl=10, show_spinner=False)
def get_health(api: str) -> dict | None:
    try:
        response = requests.get(f"{api}/health", timeout=5)
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_stats(api: str) -> dict | None:
    try:
        return requests.get(f"{api}/stats", timeout=5).json()
    except requests.RequestException:
        return None


# Cached on the full argument set, so toggling a filter re-requests but asking the
# same question twice does not. This matters because /ask costs real money.
@st.cache_data(ttl=3600, show_spinner=False)
def call_ask(api: str, payload: dict) -> tuple[dict | None, str | None, float]:
    started = time.perf_counter()
    try:
        response = requests.post(f"{api}/ask", json=payload, timeout=180)
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            return None, f"{response.status_code}: {response.text[:300]}", elapsed
        return response.json(), None, elapsed
    except requests.RequestException as exc:
        return None, str(exc), time.perf_counter() - started


@st.cache_data(ttl=3600, show_spinner=False)
def call_search(api: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(f"{api}/search", json=payload, timeout=120)
        if response.status_code != 200:
            return None, f"{response.status_code}: {response.text[:300]}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_confidence(response: dict) -> None:
    confidence = response.get("confidence", 0.0)
    breakdown = response.get("confidence_breakdown") or {}

    label = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.45 else "low"
    st.metric("Confidence", f"{confidence:.2f}", label)
    st.progress(min(1.0, max(0.0, confidence)))

    if not breakdown:
        return

    st.caption("**Breakdown** — the number is only useful with its components")
    parts = [
        ("Citation support", breakdown.get("citation_support_rate", 0.0), 0.40),
        ("Retrieval strength", breakdown.get("retrieval_strength", 0.0), 0.25),
        ("Grounding", breakdown.get("grounding_rate", 0.0), 0.20),
        ("Completeness", breakdown.get("completeness", 0.0), 0.15),
    ]
    for name, value, weight in parts:
        st.caption(f"{name} · weight {weight}")
        st.progress(min(1.0, max(0.0, float(value))), text=f"{value:.2f}")

    penalty = breakdown.get("staleness_penalty", 0.0)
    if penalty:
        st.caption(f"⚠️ staleness penalty −{penalty:.2f} "
                   f"({breakdown.get('stale_citations', 0)} cited docs over 2 years old)")

    for note in breakdown.get("notes", []):
        st.caption(f"· {note}")


def render_citations(response: dict) -> None:
    citations = response.get("citations") or []
    if not citations:
        st.info("The answer cited no sources.")
        return

    failed = [c for c in citations if c.get("verdict") == "unsupported"]
    if failed:
        st.error(
            f"**{len(failed)} of {len(citations)} citations did not survive verification.** "
            "These are claims that pointed at a real, relevant-looking document which does not "
            "actually say what was claimed."
        )

    for cite in citations:
        icon, meaning = VERDICT_STYLE.get(cite.get("verdict"), ("❔", "unchecked"))
        stale = " · ⏳ over 2 years old" if cite.get("is_stale") else ""
        header = f"{icon} [{cite['label']}] {cite.get('breadcrumb') or cite['chunk_id']}{stale}"

        with st.expander(header, expanded=bool(failed) and cite.get("verdict") == "unsupported"):
            st.caption(f"verdict: **{meaning}**")
            if cite.get("claim"):
                st.markdown(f"**Claim it was attached to:** {cite['claim']}")
            if cite.get("verdict_reason"):
                st.markdown(f"**Judge's reasoning:** {cite['verdict_reason']}")
            if cite.get("url"):
                st.markdown(f"[Open the source]({cite['url']})")
            st.caption(f"{cite.get('source_name', '')} · {cite.get('doc_type', '')} · "
                       f"updated {cite.get('age_days', '?')} days ago")
            st.text_area(
                "Cited text", cite.get("text", ""), height=170,
                key=f"cite-text-{cite['label']}-{cite['chunk_id']}", disabled=True,
            )


def render_answer(response: dict, elapsed: float) -> None:
    if response.get("answered"):
        st.success("Answered")
    else:
        st.warning(f"Refused — {response.get('refusal_reason', 'low confidence')}")

    st.markdown(response.get("answer", ""))

    nearest = response.get("nearest_sections") or []
    if nearest:
        st.markdown("**Closest matching sections** — offered instead of a guess")
        for near in nearest:
            if near.get("url"):
                st.markdown(f"- [{near['breadcrumb']}]({near['url']})")
            else:
                st.markdown(f"- {near['breadcrumb']}")

    unverified = response.get("unverified") or []
    if unverified:
        st.markdown("#### What I could not verify")
        for item in unverified:
            st.markdown(f"- {item}")

    timings = response.get("timings_ms") or {}
    usage = response.get("usage") or {}
    bits = [f"{k} {v:.0f}ms" for k, v in timings.items()]
    caption = " · ".join(bits) + f" · round trip {elapsed:.1f}s"
    if usage:
        caption += f" · ${usage.get('estimated_cost_usd', 0):.4f} cumulative"
    st.caption(caption)

    if response.get("citations") and response.get("judge_shares_model_with_generator"):
        st.caption(
            "⚠️ The citation judge is the same model that wrote the answer, so it is predisposed "
            "to agree with itself. Set `JUDGE_MODEL` in .env for an independent check."
        )


def render_comparison(api: str, question: str, top_k: int, filters: dict) -> None:
    """Dense-only against the full pipeline, on one question.

    Uses /search rather than /ask so switching modes costs nothing - the point is
    to see how the retrieved set changes, and generation would only add latency
    and expense to that.
    """
    st.caption(
        "Retrieval only, no LLM call. Measured across the whole evaluation set, "
        "dense scored 87.8% and the full pipeline 93.9% — this shows one question's worth of that."
    )

    left, right = st.columns(2)
    payloads = [("dense", left, "Dense only"), ("rerank", right, "Hybrid + reranker")]
    doc_sets: dict[str, set[str]] = {}

    for mode, column, title in payloads:
        result, error = call_search(api, {"query": question, "mode": mode, "top_k": top_k, **filters})
        with column:
            st.markdown(f"##### {title}")
            if error:
                st.error(error)
                continue
            total = sum((result.get("timings_ms") or {}).values())
            st.caption(f"{total:.0f}ms")
            doc_sets[mode] = set()
            for hit in result.get("hits", []):
                doc_sets[mode].add(hit["chunk_id"])
                found = "+".join(hit.get("found_by") or []) or "-"
                st.markdown(f"**{hit['rank']}. {hit['breadcrumb'][:70]}**")
                st.caption(f"score {hit['score']:.3f} · found by {found} · "
                           f"{hit['source_name']} · {hit['doc_type']}")

    if len(doc_sets) == 2:
        only_rerank = doc_sets["rerank"] - doc_sets["dense"]
        st.markdown("---")
        st.markdown(
            f"**{len(only_rerank)} of {top_k} chunks in the hybrid result were not in the "
            f"dense-only result.** Those are what fusion and reranking contributed."
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🔎 Support Knowledge Copilot")
st.caption("Hybrid retrieval · grounded answers · **every citation verified against its claim**")

with st.sidebar:
    st.header("Settings")
    api = st.text_input("API URL", DEFAULT_API)

    health = get_health(api)
    if health is None:
        st.error("API unreachable.\n\nStart it with:\n```\npython serve.py\n```")
    elif health["status"] == "ok":
        st.success(f"API healthy · {health['indexed_chunks']:,} chunks indexed")
    else:
        st.warning("API degraded")
        for line in health.get("detail", []):
            st.caption(f"· {line}")

    st.divider()
    mode = st.selectbox("Retrieval mode", MODES, index=0,
                        help="rerank is the full pipeline; the others are ablations")
    top_k = st.slider("Chunks sent to the model", 1, 10, 5)
    verify = st.checkbox("Verify citations", value=True,
                         help="Costs one extra LLM call per citation. Turning it off makes "
                              "confidence far less meaningful.")

    st.divider()
    st.caption("**Filters** (metadata stored at ingestion)")
    source = st.selectbox("Source", ["any", "k8s-website", "posthog", "zulip", "fastapi"])
    doc_type = st.selectbox("Doc type", ["any", "troubleshooting", "faq", "guide", "concept",
                                        "api_reference", "policy", "onboarding", "tutorial",
                                        "release_notes"])
    access = st.selectbox("Access level", ["any", "public", "internal", "confidential"])

    stats = get_stats(api)
    if stats:
        st.divider()
        st.caption("**Active configuration**")
        retrieval = stats.get("retrieval", {})
        st.caption(f"rrf_k {retrieval.get('rrf_k')} · "
                   f"dense_top_k {retrieval.get('dense_top_k')} · "
                   f"rerank_top_n {retrieval.get('rerank_top_n')}")
        st.caption(f"min_confidence {stats.get('min_confidence')}")
        st.caption(f"embeddings: `{stats.get('models', {}).get('embedding', '')}`")
        st.caption(f"llm: `{stats.get('models', {}).get('llm', '')}`")

filters = {}
if source != "any":
    filters["source_name"] = source
if doc_type != "any":
    filters["doc_type"] = doc_type
if access != "any":
    filters["access_level"] = access

EXAMPLES = [
    "why is my pod stuck in pending state",
    "how do I archive a channel",
    "what is the parental leave policy",
    "what is the pricing for Kubernetes support plans",
    "how do I receive uploaded files in FastAPI",
]

st.markdown("##### Ask a question")
chosen = st.selectbox("Examples", ["(type your own)"] + EXAMPLES, index=0,
                      label_visibility="collapsed")
default_question = "" if chosen == "(type your own)" else chosen
question = st.text_input("Question", value=default_question, label_visibility="collapsed",
                         placeholder="e.g. why is my pod stuck in pending state")

col_ask, col_hint = st.columns([1, 5])
with col_ask:
    submitted = st.button("Ask", type="primary", use_container_width=True)
with col_hint:
    st.caption("Try the pricing example to see a refusal, and the FastAPI upload example "
               "to see a stale-source warning.")

if submitted and question.strip():
    # Stored in session_state so the answer survives Streamlit's rerun when you
    # expand a citation or move a slider.
    payload = {"question": question.strip(), "mode": mode, "top_k": top_k,
               "verify": verify, **filters}
    with st.spinner("retrieving, answering, verifying citations..."):
        response, error, elapsed = call_ask(api, payload)
    st.session_state["response"] = response
    st.session_state["error"] = error
    st.session_state["elapsed"] = elapsed
    st.session_state["question"] = question.strip()

response = st.session_state.get("response")
error = st.session_state.get("error")

if error:
    st.error(f"Request failed: {error}")

if response:
    answer_tab, citations_tab, sources_tab, compare_tab = st.tabs(
        ["Answer", "Citation verdicts", "Retrieved chunks", "Dense vs hybrid"]
    )

    with answer_tab:
        left, right = st.columns([3, 1])
        with left:
            render_answer(response, st.session_state.get("elapsed", 0.0))
        with right:
            render_confidence(response)

    with citations_tab:
        render_citations(response)

    with sources_tab:
        result, search_error = call_search(
            api, {"query": st.session_state["question"], "mode": mode, "top_k": top_k, **filters}
        )
        if search_error:
            st.error(search_error)
        elif result:
            coverage = result.get("coverage") or {}
            if coverage:
                st.caption(f"found by both: {coverage.get('both', 0)} · "
                           f"dense only: {coverage.get('dense_only', 0)} · "
                           f"sparse only: {coverage.get('sparse_only', 0)}")
            for hit in result.get("hits", []):
                found = "+".join(hit.get("found_by") or []) or "-"
                with st.expander(f"{hit['rank']}. [{hit['score']:.3f}] {hit['breadcrumb'][:78]}"):
                    st.caption(f"found by {found} · dense #{hit.get('dense_rank')} · "
                               f"sparse #{hit.get('sparse_rank')} · "
                               f"{hit['source_name']} · {hit['doc_type']} · "
                               f"updated {hit.get('age_days', '?')} days ago")
                    if hit.get("url"):
                        st.markdown(f"[Open the source]({hit['url']})")
                    st.text(hit["text"][:1800])

    with compare_tab:
        render_comparison(api, st.session_state["question"], top_k, filters)
else:
    st.info("Ask a question to begin. The dashboard shows the answer, every citation's "
            "verdict, the retrieved chunks, and a dense-vs-hybrid comparison.")
