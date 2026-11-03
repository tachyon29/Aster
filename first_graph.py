
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define shared state
class WebPilotState(TypedDict):
    mission: str
    plan: str
    result: str


# 2. Define nodes

def plan_node(state: WebPilotState):
    print("Planning mission...")

    return {
        "plan": f"Research the following: {state['mission']}"
    }


def research_node(state: WebPilotState):
    print("Researching...")

    return {
        "result": f"Research completed for: {state['mission']}"
    }


def report_node(state: WebPilotState):
    print("Generating report...")

    return {
        "result": state["result"] + " | Report generated!"
    }


# 3. Build graph
builder = StateGraph(WebPilotState)

builder.add_node("plan", plan_node)
builder.add_node("research", research_node)
builder.add_node("report", report_node)

# 4. Define execution flow
builder.add_edge(START, "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "report")
builder.add_edge("report", END)

# 5. Compile graph
graph = builder.compile()

# 6. Execute graph
initial_state = {
    "mission": "Research AI startups in Bangalore",
    "plan": "",
    "result": ""
}

final_state = graph.invoke(initial_state)

print("\nFinal state:")
print(final_state)
