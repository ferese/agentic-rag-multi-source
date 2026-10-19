
"""Professional Streamlit UI for the local Multi-source Agentic RAG application."""

import streamlit as st

from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI | Research Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            120deg,
            rgba(79, 70, 229, 0.12),
            rgba(14, 165, 233, 0.06)
        );
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 750;
        letter-spacing: -0.8px;
    }

    .hero p {
        margin: 0.65rem 0 0 0;
        opacity: 0.8;
        font-size: 1rem;
    }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .source-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1rem;
        min-height: 100px;
        margin-bottom: 0.5rem;
    }

    .source-card-title {
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .source-card-description {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------
NODE_LABELS = {
    "choose_workflow": "Selecting the appropriate workflow",
    "create_plan": "Creating an execution plan",
    "execute_step": "Executing the next planned task",
    "finish_plan": "Combining evidence into the final answer",
    "route": "Routing the question",
    "retrieve": "Searching indexed documents",
    "grade_documents": "Checking document relevance",
    "rewrite_query": "Improving the search query",
    "web_search": "Searching public web sources",
    "query_sql": "Querying the local SQL database",
    "generate": "Generating the answer",
}

TOOL_LABELS = {
    "sql": "SQL Database",
    "vectorstore": "Document Retrieval",
    "web_search": "Web Search",
}


@st.cache_resource(show_spinner="Loading the local AI system and knowledge base...")
def get_graph():
    ensure_index()
    return build_graph()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Agentic AI")
    st.caption("Multi-source research workspace")
    st.divider()

    st.markdown("### Connected capabilities")

    st.markdown("🗃️ **SQL Database**")
    st.caption("Structured data and database queries")

    st.markdown("📚 **Document Retrieval**")
    st.caption("Search through indexed local documents")

    st.markdown("🌐 **Web Search**")
    st.caption("Retrieve public information when needed")

    st.markdown("🔗 **LangGraph**")
    st.caption("Plan and execute multi-step workflows")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by your local Agentic RAG pipeline.")
    st.caption("Model and indexing configuration remain unchanged.")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Multi-source Agentic RAG</h1>
        <p>
            Ask questions, explore your knowledge base, and let the agent
            choose and execute the tools needed to find an answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Source capability cards
# ---------------------------------------------------------
st.markdown(
    '<div class="section-heading">Your research tools</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-card-title">🗃️ SQL Database</div>
            <div class="source-card-description">
                Query structured records and retrieve data from tables.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-card-title">📚 Knowledge Base</div>
            <div class="source-card-description">
                Retrieve relevant passages from your indexed documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-card-title">🌐 Web Search</div>
            <div class="source-card-description">
                Search public web information when the agent selects this tool.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------
try:
    graph = get_graph()
    st.success("AI system loaded and ready.", icon="✅")
except Exception as exc:
    st.error(f"Could not initialise the AI system: {exc}")
    st.info(
        "Check that Ollama is running, the required model is available, "
        "and the project dependencies are installed."
    )
    st.stop()


# ---------------------------------------------------------
# Conversation history
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("plan"):
            with st.expander("View execution plan"):
                for index, step in enumerate(message["plan"], start=1):
                    st.markdown(
                        f"**Step {index}: {step.get('tool', 'Unknown tool')}**"
                    )
                    st.write(step.get("task", ""))
                    st.caption(f"Status: {step.get('status', 'unknown')}")

        if message["role"] == "assistant" and message.get("trace"):
            with st.expander("View execution details"):
                for item in message["trace"]:
                    st.markdown(
                        f"**Step {item.get('step', '?')} — "
                        f"{item.get('tool', 'Unknown tool')}**"
                    )
                    st.write(item.get("task", ""))
                    st.caption(
                        f"Status: {item.get('status', 'unknown')} · "
                        f"Results: {item.get('result_count', 0)}"
                    )

        if message["role"] == "assistant" and message.get("evidence"):
            with st.expander(
                f"View evidence ({len(message['evidence'])} items)"
            ):
                for item in message["evidence"]:
                    st.markdown(f"**Source: {item['source']}**")
                    st.text(item["content"])
                    st.divider()


# ---------------------------------------------------------
# User input and agent execution
# ---------------------------------------------------------
question = st.chat_input(
    "Ask a question about your data, documents, or public information..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = {}

        try:
            with st.status(
                "Agent is analysing your question...",
                expanded=True,
            ) as status:
                for update in graph.stream(
                    {"question": question},
                    stream_mode="updates",
                ):
                    for node, values in update.items():
                        label = NODE_LABELS.get(
                            node,
                            node.replace("_", " ").capitalize(),
                        )
                        st.write(f"⚙️ {label}")

                        if values:
                            state.update(values)

                status.update(
                    label="Execution completed",
                    state="complete",
                    expanded=False,
                )

        except Exception as exc:
            st.error(f"Agent execution failed: {type(exc).__name__}: {exc}")
            st.caption(
                "Your saved backup is unchanged. Check the terminal for "
                "additional error details."
            )
            st.stop()

        # ---------------------------------------------
        # Execution overview
        # ---------------------------------------------
        plan = state.get("plan", []) or []
        trace = state.get("execution_trace", []) or []
        datasource = state.get("datasource", "")

        used_tools = []
        for item in trace:
            tool = item.get("tool")
            if tool and tool not in used_tools:
                used_tools.append(tool)

        if not used_tools and datasource:
            used_tools.append(datasource)

        if plan or trace:
            st.markdown("### Agent execution")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("Planned steps", len(plan))
            metric2.metric("Executed steps", len(trace))
            metric3.metric("Tools used", len(used_tools))

            if plan:
                with st.expander("📋 Execution plan", expanded=True):
                    for index, step in enumerate(plan, start=1):
                        tool = step.get("tool", "Unknown tool")
                        task = step.get("task", "")
                        result_status = step.get("status", "pending")

                        st.markdown(
                            f"**{index}. {TOOL_LABELS.get(tool, tool)}**"
                        )
                        st.write(task)

                        if result_status.lower() in (
                            "completed",
                            "complete",
                            "success",
                        ):
                            st.success(f"Status: {result_status}")
                        elif result_status.lower() in (
                            "failed",
                            "error",
                            "no_results",
                        ):
                            st.warning(f"Status: {result_status}")
                        else:
                            st.caption(f"Status: {result_status}")

            if trace:
                with st.expander("🔍 Detailed execution trace"):
                    for item in trace:
                        st.markdown(
                            f"**Step {item.get('step', '?')} — "
                            f"{TOOL_LABELS.get(item.get('tool'), item.get('tool', 'Unknown'))}**"
                        )
                        st.write(item.get("task", ""))
                        st.caption(
                            f"Status: {item.get('status', 'unknown')} · "
                            f"Results: {item.get('result_count', 0)}"
                        )

                        if item.get("error"):
                            st.warning(item["error"])

        # ---------------------------------------------
        # Final answer
        # ---------------------------------------------
        st.divider()
        st.markdown("### 💡 Answer")

        answer = state.get(
            "generation",
            "No answer was produced. Review the execution details.",
        )

        st.markdown(answer)

        # ---------------------------------------------
        # Source and evidence panels
        # ---------------------------------------------
        source_names = {
            "vectorstore": "Indexed documents",
            "web_search": "Public web search",
            "sql": "Local SQL database",
            "plan_execute": "Multi-step agent workflow",
        }

        if datasource:
            st.caption(
                f"Workflow selected: "
                f"{source_names.get(datasource, datasource)}"
            )

        if used_tools:
            st.markdown("**Tools involved in this response**")
            tool_columns = st.columns(min(len(used_tools), 3))

            for index, tool in enumerate(used_tools):
                with tool_columns[index % len(tool_columns)]:
                    st.info(TOOL_LABELS.get(tool, tool))

        documents = state.get("documents", []) or []
        evidence_items = []

        for doc in documents:
            evidence_items.append(
                {
                    "source": doc.metadata.get("source", "Unknown source"),
                    "content": doc.page_content[:1500],
                }
            )

        gathered = state.get("gathered_evidence", []) or []

        for item in gathered:
            if hasattr(item, "metadata") and hasattr(item, "page_content"):
                evidence_items.append(
                    {
                        "source": item.metadata.get("source", "Unknown source"),
                        "content": item.page_content[:1500],
                    }
                )
            elif isinstance(item, dict):
                evidence_items.append(
                    {
                        "source": item.get("source", "Agent evidence"),
                        "content": str(
                            item.get("content", item.get("text", item))
                        )[:1500],
                    }
                )
            else:
                evidence_items.append(
                    {"source": "Agent evidence", "content": str(item)[:1500]}
                )

        # Avoid displaying the same source excerpt more than once.
        unique_evidence = []
        seen_evidence = set()

        for item in evidence_items:
            key = (item["source"], item["content"])
            if key not in seen_evidence:
                seen_evidence.add(key)
                unique_evidence.append(item)

        if unique_evidence:
            with st.expander(
                f"📎 Supporting evidence ({len(unique_evidence)} items)"
            ):
                for item in unique_evidence:
                    st.markdown(f"**Source: {item['source']}**")
                    st.text(item["content"])
                    st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "plan": plan,
                "trace": trace,
                "evidence": unique_evidence,
            }
        )
