
"""Shared state that flows through the LangGraph nodes."""

from typing import Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # Existing single-source workflow
    question: str
    original_question: str
    datasource: str
    documents: list[Document]
    generation: str
    retries: int

    # Plan-and-execute workflow
    use_planner: bool
    plan: list[dict[str, Any]]
    current_step: int
    execution_trace: list[dict[str, Any]]
    gathered_evidence: list[Document]
    planner_error: str
