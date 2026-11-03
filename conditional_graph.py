
from typing import TypedDict
import random

from langgraph.graph import StateGraph, START, END


class WebPilotState(TypedDict):
    mission: str
    attempts: int
    success: bool
    result: str


def research_node(state: WebPilotState):
    attempt = state["attempts"] + 1

    print(f"Research attempt {attempt}")

    # Simulate an unreliable research operation
    success = random.random() < 0.4

    return {
        "attempts": attempt,
        "success": success,
        "result": "Research completed" if success else "Research failed"
    }


def decide_next_step(state: WebPilotState):
    if state["success"]:
        return "report"

    if state["attempts"] < 3:
        return "retry"

    return "failed"


def report_node(state: WebPilotState):
    print("Generating report...")
    return {"result": "Report generated successfully"}


def failure_node(state: WebPilotState):
    print("Research failed after maximum retries")
    return {"result": "Unable to complete research"}


builder = StateGraph(WebPilotState)

builder.add_node("research", research_node)
builder.add_node("report", report_node)
builder.add_node("failure", failure_node)

builder.add_edge(START, "research")

# Conditional routing
builder.add_conditional_edges(
    "research",
    decide_next_step,
    {
        "retry": "research",
        "report": "report",
        "failed": "failure"
    }
)

builder.add_edge("report", END)
builder.add_edge("failure", END)

graph = builder.compile()

initial_state = {
    "mission": "Research AI startups",
    "attempts": 0,
    "success": False,
    "result": ""
}

final_state = graph.invoke(initial_state)

print("\nFinal state:", final_state)
