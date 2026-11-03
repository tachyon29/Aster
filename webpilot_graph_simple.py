import asyncio
import json
import os
import re
import sys
import uuid
from contextlib import AsyncExitStack
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

import requests
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client



# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_RESEARCH_STEPS = 10
MAX_AGENT_RETRIES = 3
MAX_TOOL_OUTPUT_CHARS = 7_000
MAX_TOTAL_CONTEXT_CHARS = 14_000
MAX_REPORT_CONTEXT_CHARS = 4_500
REPORT_NUM_PREDICT = 350
REPORT_TIMEOUT_SECONDS = 45
RESEARCH_TIMEOUT_SECONDS = 60

SERVERS = {
    "browser": "mcp_server.py",
}


# ============================================================
# LANGGRAPH STATE
# ============================================================

class WebPilotState(TypedDict):
    mission: str
    plan: dict
    tools: list
    tool_to_client: Any
    tool_results: list
    pending_tool_calls: list
    steps: int
    report_saved: bool
    report_filename: str
    final_output: str
    progress_callback: Any


def publish_progress(state: WebPilotState, event: dict) -> None:
    callback = state.get("progress_callback")
    if callback:
        callback(event)


# ============================================================
# OLLAMA HELPERS
# ============================================================

def ollama_chat(
    messages: list,
    tools: list | None = None,
    num_predict: int = 3000,
    timeout: int = 300,
    output_format: str | dict | None = None,
) -> dict:
    """Call Ollama's native chat API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": num_predict,
        },
    }
    if tools:
        payload["tools"] = tools
    if output_format:
        payload["format"] = output_format

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {})
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc


def strip_think_tags(value: str) -> str:
    has_open = re.search(r"<think\b[^>]*>", value, flags=re.IGNORECASE)
    has_close = re.search(r"</think\s*>", value, flags=re.IGNORECASE)
    if has_close and not has_open:
        value = re.split(r"</think\s*>", value, flags=re.IGNORECASE)[-1]
    elif has_open and not has_close:
        value = value[:has_open.start()]
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"</?think\b[^>]*>", "", value, flags=re.IGNORECASE)
    return value.strip()


# ============================================================
# MCP RESULT HELPERS
# ============================================================

def extract_tool_text(result) -> tuple[str, bool]:
    data = result.model_dump(by_alias=True)
    is_error = data.get("isError", data.get("is_error", False))

    parts = []
    for item in getattr(result, "content", []) or []:
        text_content = getattr(item, "text", None)
        if text_content is not None:
            parts.append(text_content)
        else:
            parts.append(str(item))

    output = "\n".join(parts).strip()
    if not output:
        structured = data.get("structuredContent") or data.get("structured_content")
        output = json.dumps(structured, ensure_ascii=False) if structured is not None else "Tool returned no content."

    return output, bool(is_error)


def parse_json_text(value: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def collect_strings(value: Any) -> list[str]:
    """Flatten nested JSON strings from the browser MCP response."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result = []
        for nested in value.values():
            result.extend(collect_strings(nested))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(collect_strings(nested))
        return result
    return []


def result_page_text(output: str) -> str:
    parsed = parse_json_text(output)
    if parsed is None:
        return output
    return "\n".join(collect_strings(parsed))


def extract_page_metadata(output: str) -> tuple[str, str, str]:
    """Return title, URL and page text when the browser returns JSON."""
    parsed = parse_json_text(output)
    if isinstance(parsed, dict):
        title = str(parsed.get("title") or "").strip()
        url = str(parsed.get("url") or parsed.get("source_url") or "").strip()
        content = str(parsed.get("content") or parsed.get("text") or "").strip()
        if content:
            return title, url, content
    return "", "", output


def extract_source_url(tool_name: str, arguments: dict, output: str) -> str:
    _title, returned_url, _content = extract_page_metadata(output)
    if returned_url.startswith(("https://", "http://")):
        return returned_url
    if tool_name == "open_website":
        requested_url = str(arguments.get("url") or "")
        if requested_url.startswith(("https://", "http://")):
            return requested_url
    return ""


# ============================================================
# PLANNER
# ============================================================

def planner_node(state: WebPilotState) -> dict:
    print("\n========== PLANNER NODE ==========\n")
    print(f"Mission: {state['mission']}")

    publish_progress(state, {
        "kind": "node_enter", "node_id": "planner", "label": "Planner",
        "purpose": "Turns the question into an ordered plan for discovery, source gathering, verification, and answering.",
        "stage": "planner",
    })

    # Planning these stable steps locally avoids an extra slow model round trip.
    plan = {
        "goal": state["mission"],
        "tasks": [
            {"id": 1, "description": "Find authoritative sources for the requested facts", "status": "pending"},
            {"id": 2, "description": "Retrieve the source pages that answer the question", "status": "pending"},
            {"id": 3, "description": "Check the requested details against the retrieved text", "status": "pending"},
            {"id": 4, "description": "Prepare a concise answer with source links", "status": "pending"},
        ],
    }

    print("\nGenerated Plan:\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return {"plan": plan}


# ============================================================
# RESEARCH CONTEXT AND AGENT
# ============================================================

def build_research_context(state: WebPilotState) -> str:
    """Give Qwen the question and extracted page text, without rigid categories."""
    sections = []
    total_chars = 0

    for index, item in enumerate(state["tool_results"], start=1):
        if item.get("is_error"):
            sections.append(f"RETRIEVAL {index} FAILED: {item.get('output', '')[:500]}")
            continue

        if item.get("tool") == "search_web":
            search_data = parse_json_text(item.get("output", "")) or {}
            results = search_data.get("results", [])
            if results:
                snippets = [
                    f"- {result.get('title', 'Search result')}\n  URL: {result.get('url', '')}\n  {result.get('snippet', '')}"
                    for result in results
                ]
                sections.append("WEB SEARCH RESULTS:\n" + "\n".join(snippets))
            continue

        page_text = result_page_text(item.get("output", ""))
        if not page_text.strip():
            continue

        remaining = MAX_REPORT_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        # Limit report context to reduce prompt processing latency.
        page_text = page_text[:min(6_000, remaining)]
        total_chars += len(page_text)

        title, returned_url, _ = extract_page_metadata(item.get("output", ""))
        url = item.get("source_url") or returned_url or "URL not provided"
        sections.append(
            f"PAGE {index}\nTitle: {title or 'Untitled page'}\nURL: {url}\n\n{page_text}"
        )

    collected_text = "\n\n---\n\n".join(sections) or "No webpage text has been collected yet."
    already_opened = [
        item.get("source_url")
        for item in state["tool_results"]
        if item.get("source_url")
    ]

    return f"""USER'S RESEARCH QUESTION:
{state['mission']}

WEBPAGE TEXT COLLECTED SO FAR:
{collected_text}

ALREADY OPENED URLS:
{json.dumps(already_opened, ensure_ascii=False)}

Your job is to research the user's question, not to follow a fixed report template.
Read the text collected so far and decide what information is still missing.
Choose the next research action from the evidence and available tools.
"""


async def agent_node(state: WebPilotState) -> dict:
    print("\n========== RESEARCH AGENT NODE ==========\n")

    publish_progress(state, {
        "kind": "node_enter", "node_id": "research_agent", "label": "Research agent",
        "purpose": "Compares collected evidence with the mission, chooses a tool when information is missing, and resumes after each tool returns.",
        "stage": "research",
    })

    if state["steps"] >= MAX_RESEARCH_STEPS:
        print("Maximum research steps reached; generating report from collected pages.")
        return {"pending_tool_calls": [], "steps": state["steps"]}

    research_tools = [
        tool for tool in state["tools"]
        if tool.get("function", {}).get("name") not in {"save_report", "list_reports"}
    ]

    available_tools = [tool["function"]["name"] for tool in research_tools]
    decision_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": available_tools + ["complete"]},
            "arguments": {"type": "object"},
        },
        "required": ["action", "arguments"],
        "additionalProperties": False,
    }
    system_prompt = """You are WebPilot's Qwen research planner. Choose the next research action based on the user's exact question and evidence collected so far. Never assume a fixed topic, site, or URL.
Return one JSON object only, with no markdown or explanation. `action` must exactly match an available tool name, or be `complete`. Use one of these forms:
{"action":"search_web","arguments":{"query":"..."}}
{"action":"open_website","arguments":{"url":"https://..."}}
{"action":"search_knowledge_base","arguments":{"query":"..."}}
{"action":"complete","arguments":{}}

Use search_web when the relevant primary source or requested facts are not yet clear. Prefer official sources in the results, then open the best result and verify its page text. Search results or an irrelevant homepage alone are not enough to complete research. Choose complete only after the collected source text supports the requested facts. The subject and search query must come from the user's question. Do not expose reasoning.
If the collected context already contains web search results and no source page has been opened yet, your next action MUST be open_website with the URL of the most relevant official result. Do not run another search until you have opened at least one result page."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"AVAILABLE TOOLS: {', '.join(available_tools)}\n\n{build_research_context(state)}"},
    ]

    for attempt in range(MAX_AGENT_RETRIES):
        print(f"Ollama research attempt {attempt + 1}/{MAX_AGENT_RETRIES}")
        try:
            assistant_message = await asyncio.to_thread(
                ollama_chat, messages, num_predict=160,
                timeout=RESEARCH_TIMEOUT_SECONDS, output_format=decision_schema,
            )
        except Exception as exc:
            print(f"Research agent model call failed; answering from collected evidence: {exc}")
            return {"pending_tool_calls": [], "steps": state["steps"] + 1}

        content = strip_think_tags(assistant_message.get("content", ""))
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            decision = {}

        action = (decision.get("action") or decision.get("tool") or decision.get("name")) if isinstance(decision, dict) else None
        arguments = (decision.get("arguments") or decision.get("parameters")) if isinstance(decision, dict) else None
        if not isinstance(arguments, dict) and isinstance(decision, dict):
            arguments = {key: decision[key] for key in ("query", "url") if key in decision}
        if action in {"search", "web_search"} and "search_web" in available_tools:
            action = "search_web"
        if action in {"open", "open_url", "browse"} and "open_website" in available_tools:
            action = "open_website"
        if isinstance(action, str):
            action = action.casefold()
        if not isinstance(arguments, dict) and action != "complete":
            arguments = {}
        if action == "complete":
            print("Qwen marked the research complete from retrieved evidence.")
            return {"pending_tool_calls": [], "steps": state["steps"] + 1}
        if action in available_tools and isinstance(arguments, dict):
            print(f"Qwen selected research tool: {action}")
            return {
                "pending_tool_calls": [{"function": {"name": action, "arguments": arguments}}],
                "steps": state["steps"] + 1,
            }

        print(f"Qwen returned an invalid research action ({content[:240]!r}); requesting a JSON tool decision again.")
        messages.append({"role": "assistant", "content": content or "{}"})
        messages.append({
            "role": "user",
            "content": (
                f"Return valid JSON only. Use the exact tool names from this list: {', '.join(available_tools)}. For an existing result, set action to open_website and put its URL in arguments.url. If source text fully answers the request, set action to complete. Do not return prose."
            ),
        })

    print("No additional tool call was produced after retries; using collected text for the report.")
    return {
        "pending_tool_calls": [],
        "steps": state["steps"] + 1,
    }


# ============================================================
# MCP TOOLS
# ============================================================

async def call_mcp_tool(state: WebPilotState, tool_name: str, arguments: dict) -> dict:
    client = state["tool_to_client"].get(tool_name)
    if client is None:
        raise ValueError(f"No MCP client found for tool: {tool_name}")

    result = await client.call_tool(tool_name, arguments)
    output, is_error = extract_tool_text(result)
    return {
        "tool": tool_name,
        "arguments": arguments,
        "output": output,
        "is_error": is_error,
        "source_url": extract_source_url(tool_name, arguments, output),
    }


async def tools_node(state: WebPilotState) -> dict:
    print("\n========== MCP TOOLS NODE ==========\n")

    results = list(state["tool_results"])
    pending = state.get("pending_tool_calls", []) or []

    for tool_call in pending:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                results.append({
                    "tool": tool_name,
                    "arguments": {},
                    "output": f"Invalid tool arguments: {exc}",
                    "is_error": True,
                    "source_url": "",
                })
                continue

        if not isinstance(arguments, dict):
            results.append({
                "tool": tool_name,
                "arguments": {},
                "output": "Tool arguments must be a JSON object.",
                "is_error": True,
                "source_url": "",
            })
            continue

        if tool_name in {"save_report", "list_reports"}:
            print(f"Skipping graph-managed tool call: {tool_name}")
            continue

        publish_progress(state, {
            "kind": "node_enter", "node_id": "tool_dispatch", "label": "MCP tool dispatcher",
            "purpose": "Validates the agent's requested call and routes it to the MCP server that owns the function.",
            "stage": "browser",
        })
        print(f"Calling MCP tool: {tool_name} with {arguments}")
        tool_description = next((
            tool.get("function", {}).get("description", "")
            for tool in state["tools"]
            if tool.get("function", {}).get("name") == tool_name
        ), "Executes a research action requested by the agent.")
        if tool_name == "open_website":
            call_detail = str(arguments.get("url") or "URL not provided")
        elif tool_name == "search_web":
            call_detail = "Query: " + str(arguments.get("query") or "query not provided")
        elif tool_name == "search_knowledge_base":
            call_detail = "Query: " + str(arguments.get("query") or "query not provided")
        else:
            call_detail = json.dumps(arguments, ensure_ascii=False)
        publish_progress(state, {
            "kind": "node_enter",
            "node_id": f"tool:{tool_name}:{uuid.uuid4().hex[:8]}",
            "label": tool_name,
            "purpose": tool_description,
            "stage": "browser",
            "detail": call_detail,
        })
        try:
            record = await call_mcp_tool(state, tool_name, arguments)
        except Exception as exc:
            record = {
                "tool": tool_name,
                "arguments": arguments,
                "output": str(exc),
                "is_error": True,
                "source_url": "",
            }

        print("Tool result:")
        print(record["output"][:2500])
        results.append(record)

    return {
        "tool_results": results,
        "pending_tool_calls": [],
    }


# ============================================================
# REPORT GENERATION: TEXT + USER QUESTION -> ANSWER
# ============================================================

def build_answer_context(mission: str, tool_results: list[dict]) -> tuple[str, list[str]]:
    blocks = []
    source_urls = []
    total_chars = 0

    for index, item in enumerate(tool_results, start=1):
        if item.get("is_error"):
            continue

        if item.get("tool") == "search_web":
            search_data = parse_json_text(item.get("output", "")) or {}
            for result in search_data.get("results", []):
                title = str(result.get("title") or "Search result")
                url = str(result.get("url") or "")
                snippet = str(result.get("snippet") or "")
                if not (url or snippet):
                    continue
                block = f"SEARCH RESULT: {title}\nURL: {url}\n{snippet}"
                remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
                if remaining <= 0:
                    break
                block = block[:remaining]
                total_chars += len(block)
                blocks.append(block)
                if url and url not in source_urls:
                    source_urls.append(url)
            continue

        title, returned_url, page_text = extract_page_metadata(item.get("output", ""))
        if not page_text:
            page_text = result_page_text(item.get("output", ""))
        url = item.get("source_url") or returned_url
        if not page_text.strip():
            continue

        remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        page_text = page_text[:min(MAX_TOOL_OUTPUT_CHARS, remaining)]
        total_chars += len(page_text)

        blocks.append(
            f"PAGE {index}: {title or 'Webpage'}\nURL: {url or 'not provided'}\n\n{page_text}"
        )
        if url and url not in source_urls:
            source_urls.append(url)

    return "\n\n---\n\n".join(blocks), source_urls


def extractive_answer(mission: str, tool_results: list[dict]) -> str:
    """Build a short source-grounded fallback if Ollama cannot summarize."""
    stop_words = {
        "about", "after", "also", "and", "are", "based", "find", "findings",
        "from", "give", "has", "have", "important", "into", "list", "made",
        "nasa", "official", "report", "save", "source", "sources", "that",
        "the", "their", "three", "this", "what", "when", "which", "with",
    }
    terms = {
        word for word in re.findall(r"[a-z0-9]+", mission.casefold())
        if len(word) > 2 and word not in stop_words | {"spacecraft", "official", "website", "research", "report", "important", "three", "made", "launch", "launched", "discoveries", "discovery"}
    }
    candidates = []
    source_urls = []
    for item in tool_results:
        if item.get("is_error"):
            continue
        if item.get("tool") == "search_web":
            search_data = parse_json_text(item.get("output", "")) or {}
            for result in search_data.get("results", []):
                url = str(result.get("url") or "")
                if url and url not in source_urls:
                    source_urls.append(url)
        title, returned_url, content = extract_page_metadata(item.get("output", ""))
        content = content or result_page_text(item.get("output", ""))
        url = item.get("source_url") or returned_url
        if url and url not in source_urls:
            source_urls.append(url)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -•\t")
            if len(sentence) < 35:
                continue
            lowered = sentence.casefold()
            tokens = set(re.findall(r"[a-z0-9]+", lowered))
            if terms and not (terms & tokens):
                continue
            score = len(terms & tokens)
            if "webb" in lowered:
                score += 2
            if any(word in lowered for word in ("discover", "detected", "revealed", "found")):
                score += 2
            if any(word in lowered for word in ("launch", "launched")):
                score += 3
            if score:
                candidates.append((score, sentence, title))

    chosen = []
    seen = set()
    for _score, sentence, _title in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence[:450])
        if len(chosen) >= 5 or sum(map(len, chosen)) >= 1_500:
            break

    answer = "\n".join(f"- {sentence}" for sentence in chosen)
    if not chosen:
        return "I couldn’t verify the requested facts in the retrieved pages. Try again so I can search for more relevant sources."
    if source_urls:
        answer += "\n\nSources\n" + "\n".join(f"- {url}" for url in source_urls[:4])
    return answer.strip()


def answer_matches_subject(mission: str, answer: str) -> bool:
    ignored = {
        "about", "after", "also", "and", "are", "based", "find", "finding", "findings",
        "from", "give", "has", "have", "important", "into", "list", "made", "nasa",
        "official", "report", "save", "source", "sources", "that", "the", "their",
        "three", "this", "what", "when", "which", "with", "spacecraft", "website",
        "research", "discover", "discovery", "discoveries", "launch", "launched",
    }
    subject_terms = {
        token for token in re.findall(r"[a-z0-9]+", mission.casefold())
        if len(token) > 2 and token not in ignored
    }
    if not subject_terms:
        return True
    answer_text = re.split(r"\n\s*(?:sources|references):?\s*\n", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    answer_terms = set(re.findall(r"[a-z0-9]+", answer_text.casefold()))
    return bool(subject_terms & answer_terms)


async def report_node(state: WebPilotState) -> dict:
    print("\n========== REPORT NODE ==========\n")

    publish_progress(state, {
        "kind": "node_enter", "node_id": "answer_writer", "label": "Answer writer",
        "purpose": "Drafts a short answer supported by the retrieved source text.",
        "stage": "answer",
    })

    page_text, _source_urls = build_answer_context(state["mission"], state["tool_results"])
    generation_error = None

    if page_text.strip():
        prompt = f"""Answer the user's research question directly and concisely.

USER'S RESEARCH QUESTION:
{state['mission']}

EXTRACTED WEBPAGE TEXT (use this as your context):
{page_text}

Instructions:
- Include only information needed to answer this specific question.
- Use only facts supported by the extracted text. Do not add background, generic introductions, conclusions, or unrelated details.
- Never reveal reasoning, thinking, analysis, or tool narration. Output only the answer.
- Use the shortest useful format. Aim for 1-3 short paragraphs or a small list; keep it under 250 words unless the question clearly requires more.
- If requested information is missing, say so briefly and answer the supported parts.
- If the user requests source links or a report, include a short Sources section with the retrieved URLs. Otherwise cite URLs inline only when useful.

Write the answer now."""

        try:
            answer_message = await asyncio.to_thread(
                ollama_chat,
                [
                    {
                        "role": "system",
                        "content": "You are WebPilot's answer writer. Return only the concise answer supported by the supplied webpage text. Never include reasoning or tool narration.",
                    },
                    {"role": "user", "content": prompt},
                ],
                num_predict=REPORT_NUM_PREDICT,
                timeout=REPORT_TIMEOUT_SECONDS,
            )
            report_content = strip_think_tags(answer_message.get("content", ""))
            if not report_content:
                generation_error = "Ollama returned an empty answer."
        except Exception as exc:
            generation_error = str(exc)
            print(f"Report generation failed: {generation_error}")
            report_content = ""
    else:
        generation_error = "No usable webpage text was retrieved."
        report_content = ""

    if report_content and not answer_matches_subject(state["mission"], report_content):
        print("Answer did not mention the requested subject; replacing it with relevant source extracts.")
        report_content = ""

    if not report_content:
        report_content = extractive_answer(state["mission"], state["tool_results"])
    if not report_content:
        report_content = "I couldn't find enough source text to answer this question. Please try again with a more specific source or question."

    mission_text = state["mission"].casefold()
    if any(term in mission_text for term in ("source link", "sources", "citations", "references")):
        _context, source_urls = build_answer_context(state["mission"], state["tool_results"])
        missing_urls = [url for url in source_urls if url not in report_content]
        if missing_urls:
            report_content = report_content.rstrip() + "\n\nSources\n" + "\n".join(f"- {url}" for url in missing_urls[:4])

    return {
        "report_saved": False,
        "report_filename": "",
        "final_output": report_content,
    }


# ============================================================
# GRAPH ROUTING
# ============================================================

def route_after_agent(state: WebPilotState):
    if state["steps"] >= MAX_RESEARCH_STEPS:
        return "report"
    if state.get("pending_tool_calls"):
        return "tools"
    return "report"


# ============================================================
# MCP SERVER CONNECTIONS
# ============================================================

async def connect_mcp_servers(exit_stack: AsyncExitStack):
    print("\n========== CONNECTING MCP SERVERS ==========\n")

    all_tools = []
    tool_to_client = {}

    for server_name, server_file in SERVERS.items():
        print(f"Connecting to MCP server: {server_name}")
        server_path = os.path.abspath(server_file)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"MCP server file not found: {server_path}")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
        )
        transport = await exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = transport
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            print(f"Discovered tool: {tool.name}")
            input_schema = getattr(tool, "input_schema", None) or {
                "type": "object",
                "properties": {},
            }
            all_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": input_schema,
                },
            })
            tool_to_client[tool.name] = session

    print("\nAll MCP servers connected successfully.")
    print(f"Total tools discovered: {len(all_tools)}")
    return all_tools, tool_to_client


# ============================================================
# BUILD AND RUN GRAPH
# ============================================================

def build_graph():
    builder = StateGraph(WebPilotState)
    builder.add_node("planner", planner_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "report": "report"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("report", END)
    return builder.compile()


async def main():
    print("\n========================================")
    print("       WEBPILOT RESEARCH AGENT")
    print("========================================\n")

    mission = input("Enter your research mission: ").strip()
    if not mission:
        print("Mission cannot be empty.")
        return

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
        }

        print("\n========== STARTING MISSION ==========\n")
        result = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 40},
        )

        print("\n========================================")
        print("          WEBPILOT FINAL OUTPUT")
        print("========================================\n")
        print(result.get("final_output") or "No final output was generated.")
        print("\n========================================")
        print("       MISSION EXECUTION COMPLETE")
        print("========================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWebPilot interrupted by user.")
