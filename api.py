
import asyncio
import json
import queue
from contextlib import AsyncExitStack

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from webpilot_graph_simple import (
    connect_mcp_servers,
    build_graph,
)


app = FastAPI(title="WebPilot API")


# Allow the React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    mission: str


def describe_exception(exc: BaseException) -> str:
    nested = getattr(exc, "exceptions", None)
    if nested:
        return "; ".join(describe_exception(item) for item in nested)
    return f"{type(exc).__name__}: {exc}"


@app.get("/")
def home():
    return {"message": "WebPilot API is running"}


@app.post("/api/research")
async def run_research(request: ResearchRequest):

    mission = request.mission.strip()

    if not mission:
        raise HTTPException(
            status_code=400,
            detail="Research mission cannot be empty",
        )

    async def events():
        def send(payload):
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            yield send({"type": "status", "stage": "planner", "message": "Preparing research"})
            yield send({"type": "graph", "action": "reset"})
            progress_queue = queue.Queue()
            current_node = "start"
            yield send({
                "type": "graph", "action": "visit", "node_id": "start",
                "label": "Research request", "purpose": "The mission entered by the user.",
                "parent": None, "stage": "planner",
            })
            async with AsyncExitStack() as exit_stack:
                tools, tool_to_client = await connect_mcp_servers(exit_stack)
                graph = build_graph()
                initial_state = {
                    "mission": mission,
                    "plan": {},
                    "tools": tools,
                    "tool_to_client": tool_to_client,
                    "tool_results": [],
                    "pending_tool_calls": [],
                    "steps": 0,
                    "report_saved": False,
                    "report_filename": "",
                    "final_output": "",
                    "progress_callback": progress_queue.put,
                }

                graph_task = asyncio.create_task(
                    graph.ainvoke(initial_state, config={"recursion_limit": 40})
                )

                while not graph_task.done():
                    try:
                        progress = progress_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.04)
                        continue

                    if progress.get("kind") == "node_enter":
                        parent = current_node
                        current_node = progress["node_id"]
                        yield send({
                            "type": "graph", "action": "visit",
                            "node_id": current_node,
                            "label": progress.get("label", current_node),
                            "purpose": progress.get("purpose", "Research workflow node."),
                            "detail": progress.get("detail", ""),
                            "parent": parent,
                            "stage": progress.get("stage", "research"),
                        })
                        stage = progress.get("stage", "research")
                        stage_message = {
                            "planner": "Planning the research",
                            "research": "Checking evidence and choosing the next action",
                            "browser": "Running a browser research tool",
                            "answer": "Writing a concise answer",
                        }.get(stage, "Working")
                        yield send({"type": "status", "stage": stage, "message": stage_message, "node": progress.get("label", current_node)})

                while True:
                    try:
                        progress = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    if progress.get("kind") == "node_enter":
                        parent = current_node
                        current_node = progress["node_id"]
                        yield send({
                            "type": "graph", "action": "visit", "node_id": current_node,
                            "label": progress.get("label", current_node),
                            "purpose": progress.get("purpose", "Research workflow node."),
                            "detail": progress.get("detail", ""), "parent": parent,
                            "stage": progress.get("stage", "research"),
                        })
                        stage = progress.get("stage", "research")
                        yield send({"type": "status", "stage": stage, "message": "Writing a concise answer" if stage == "answer" else "Research in progress", "node": progress.get("label", current_node)})

                result = await graph_task

                yield send({
                    "type": "graph", "action": "visit", "node_id": "end",
                    "label": "Complete", "purpose": "The LangGraph run finished and the answer is ready.",
                    "parent": current_node, "stage": "complete",
                })

                yield send({
                    "type": "complete",
                    "final_output": result.get("final_output", "Research completed."),
                })
        except Exception as exc:
            yield send({"type": "error", "message": describe_exception(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
