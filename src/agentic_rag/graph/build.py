
"""Assemble the original and plan-and-execute LangGraph workflows."""

from langgraph.graph import END, START, StateGraph

from . import nodes, planning
from .state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    # Entry and workflow selection
    workflow.add_node("choose_workflow", planning.choose_workflow)
    workflow.add_node("create_plan", planning.create_plan)
    workflow.add_node("execute_step", planning.execute_step)
    workflow.add_node("finish_plan", planning.finish_plan)

    # Original single-source workflow
    workflow.add_node("route", nodes.route)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("query_sql", nodes.query_sql)
    workflow.add_node("generate", nodes.generate)

    workflow.add_edge(START, "choose_workflow")

    workflow.add_conditional_edges(
        "choose_workflow",
        lambda state: "plan" if state.get("use_planner") else "single",
        {"plan": "create_plan", "single": "route"},
    )

    # Plan-and-execute branch
    workflow.add_edge("create_plan", "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        planning.next_plan_action,
        {"execute_step": "execute_step", "finish_plan": "finish_plan"},
    )
    workflow.add_edge("finish_plan", END)

    # Original workflow, unchanged
    workflow.add_conditional_edges(
        "route",
        nodes.select_datasource,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "sql": "query_sql",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
        },
    )
    workflow.add_conditional_edges(
        "rewrite_query",
        nodes.route_after_rewrite,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("query_sql", "generate")
    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation,
        {"useful": END, "rewrite": "rewrite_query"},
    )

    return workflow.compile()
