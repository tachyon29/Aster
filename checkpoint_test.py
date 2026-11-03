
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class State(TypedDict):
    mission: str
    progress: int


def research_node(state: State):
    new_progress = state["progress"] + 1

    print(f"Research progress: {new_progress}")

    return {"progress": new_progress}


async def main():

    async with AsyncSqliteSaver.from_conn_string(
        "webpilot_checkpoints.db"
    ) as checkpointer:

        builder = StateGraph(State)

        builder.add_node("research", research_node)
        builder.add_edge(START, "research")
        builder.add_edge("research", END)

        graph = builder.compile(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": "research_mission_1"
            }
        }

        # First execution: create the mission
        print("\n--- First execution ---")

        result = await graph.ainvoke(
            {
                "mission": "Research AI startups",
                "progress": 0
            },
            config=config
        )

        print("Saved state:", result)

        # Second execution: resume from existing state
        print("\n--- Resuming mission ---")

        result = await graph.ainvoke(
            None,
            config=config
        )

        print("Updated state:", result)


if __name__ == "__main__":
    asyncio.run(main())
