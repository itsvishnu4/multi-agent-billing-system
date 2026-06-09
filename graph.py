from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from agents import (eligibility_agent, eligibility_router,
                    claim_creator_agent, claim_auditor_agent, audit_router)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    patient_data: dict
    service_data: dict
    eligibility_verdict: str | None
    eligibility_reason: str | None
    coverage_gaps: list[str] | None
    claim_json: dict | None
    missing_fields: list[str] | None
    audit_verdict: str | None
    audit_issues: list[str] | None
    retry_count: int


def build_graph():
    g = StateGraph(State)

    g.add_node("eligibility_agent", eligibility_agent)
    g.add_node("claim_creator", claim_creator_agent)
    g.add_node("claim_auditor", claim_auditor_agent)

    g.add_edge(START, "eligibility_agent")
    g.add_conditional_edges(
        "eligibility_agent", eligibility_router,
        {"create_claim": "claim_creator", "denied": END}
    )
    g.add_edge("claim_creator", "claim_auditor")
    g.add_conditional_edges(
        "claim_auditor", audit_router,
        {"approved": END, "create_claim": "claim_creator"}
    )

    return g.compile()


graph = build_graph()
