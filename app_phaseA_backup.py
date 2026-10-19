
"""Streamlit UI for the local multi-source Agentic RAG application."""

import streamlit as st

from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index

NODE_LABELS = {
    "choose_workflow": "Deciding whether planning is needed...",
    "create_plan": "Creating a multi-step plan...",
    "execute_step": "Executing the next planned task...",
    "finish_plan": "Combining evidence into the final answer...",
    "route": "Routing the question...",
    "retrieve": "Searching the knowledge base...",
    "grade_documents": "Checking document relevance...",
    "rewrite_query": "Rewriting the query...",
    "web_search": "Searching the web...",
    "query_sql": "Querying the local database...",
    "generate": "Writing and checking the answer...",
}

st.set_page_config(page_title="Agentic RAG", page_icon="🧠", layout="centered")
st.title("Multi-source Agentic RAG")
st.caption(
    "Local LangGraph agent with SQL, document retrieval, web search, "
    "self-correction, and bounded plan-and-execute workflows."
)


@st.cache_resource(show_spinner="Loading models and index...")
def get_graph():
    ensure_index()
    return build_graph()


graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = {}

        with st.status("Thinking...", expanded=True) as status:
            try:
                for update in graph.stream(
                    {"question": question}, stream_mode="updates"
                ):
                    for node, values in update.items():
                        st.write(NODE_LABELS.get(node, node))
                        if values:
                            state.update(values)

                status.update(label="Completed", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Execution failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()

        plan = state.get("plan", [])
        trace = state.get("execution_trace", [])

        if plan:
            with st.expander("Execution plan", expanded=True):
                for index, step in enumerate(plan, start=1):
                    st.markdown(
                        f"**Step {index}: {step['tool']}** — "
                        f"{step['task']}"
                    )
                    st.caption(f"Status: {step.get('status', 'pending')}")

        if trace:
            with st.expander("Agent execution trace"):
                for item in trace:
                    st.markdown(
                        f"**Step {item['step']} — {item['tool']}**: "
                        f"{item['status']}"
                    )
                    st.write(item["task"])
                    st.caption(f"Results: {item['result_count']}")
                    if item.get("error"):
                        st.warning(item["error"])

        answer = state.get(
            "generation", "No answer was produced. Check the execution logs."
        )
        st.markdown(answer)

        source_banners = {
            "vectorstore": "Answered using indexed documents.",
            "web_search": "Answered using public web search.",
            "sql": "Answered using the local SQL database.",
            "plan_execute": "Answered using a multi-step execution plan.",
        }
        if banner := source_banners.get(state.get("datasource", "")):
            st.caption(banner)

        documents = state.get("documents", [])
        if documents:
            with st.expander(f"Evidence ({len(documents)} items)"):
                for doc in documents:
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Source: {source}**")
                    st.text(doc.page_content[:1000])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
