import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from typing import Any, TypedDict

import requests
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from planner import create_plan


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_RESEARCH_STEPS = 10
MAX_EMPTY_RETRIES = 2
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_TOTAL_EVIDENCE_CHARS = 28_000

SERVERS = {
    "browser": "mcp_server.py",
    "reports": "report_server.py",
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


# ============================================================
# OLLAMA HELPERS
# ============================================================

def ollama_chat(
    messages: list,
    tools: list | None = None,
    num_predict: int = 2200,
    json_mode: bool = False,
) -> dict:
    """Call Ollama's native /api/chat endpoint."""

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
        },
    }

    if tools:
        payload["tools"] = tools

    # Ollama supports constrained JSON output on /api/chat.
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {})
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc


def strip_think_tags(value: str) -> str:
    """Remove Qwen-style think blocks if the model emits them in content."""
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()


def extract_json_object(value: str) -> dict | None:
    """Parse a JSON object, tolerating harmless text surrounding it."""
    cleaned = strip_think_tags(value)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ============================================================
# MCP RESULT HELPERS
# ============================================================

def extract_tool_text(result) -> tuple[str, bool]:
    """Extract text/structured content and the MCP error flag."""
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
        structured = (
            data.get("structuredContent")
            or data.get("structured_content")
        )
        if structured is not None:
            output = json.dumps(structured, ensure_ascii=False)
        else:
            output = "Tool returned no text or structured content."

    return output, bool(is_error)


def parse_json_text(value: str) -> dict | list | None:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def collect_strings(value: Any) -> list[str]:
    """Collect nested string values from a tool's JSON response."""
    found = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            found.extend(collect_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(collect_strings(nested))
    return found


def result_page_text(output: str) -> str:
    parsed = parse_json_text(output)
    if parsed is None:
        return output
    return "\n".join(collect_strings(parsed))


def normalize_for_match(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_source_url(tool_name: str, arguments: dict, output: str) -> str:
    """Prefer the URL returned by the browser, then the URL we requested."""
    parsed = parse_json_text(output)
    if isinstance(parsed, dict):
        for key in ("url", "source_url"):
            candidate = parsed.get(key)
            if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
                return candidate
    if tool_name == "open_website":
        candidate = arguments.get("url", "")
        if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
            return candidate
    return ""


def knowledge_result_is_relevant(output: str, query: str, mission: str) -> tuple[bool, str]:
    """Reject obvious vector-store mismatches instead of treating top-k as truth."""
    parsed = parse_json_text(output)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        # For non-JSON responses, require a meaningful query anchor to appear.
        anchors = meaningful_query_anchors(query)
        text = output.casefold()
        if anchors and not any(anchor in text for anchor in anchors):
            return False, "No meaningful query terms appeared in the retrieved text."
        return True, ""

    results = parsed.get("results", [])
    if not results:
        return False, "The knowledge base returned no results."

    anchors = meaningful_query_anchors(query)
    kept = []
    rejected = []

    official_python_request = (
        "python" in mission.casefold()
        and "official" in mission.casefold()
        and any(word in mission.casefold() for word in ("feature", "release", "version", "change"))
    )

    allowed_python_hosts = (
        "python.org",
        "docs.python.org",
        "peps.python.org",
        "github.com/python/",
    )

    for item in results:
        item_text = " ".join(collect_strings(item)).casefold()
        anchor_match = not anchors or any(anchor in item_text for anchor in anchors)
        source_url = ""
        if isinstance(item, dict):
            metadata = item.get("metadata", {})
            if isinstance(metadata, dict):
                source_url = str(metadata.get("source_url", ""))

        official_match = (
            not official_python_request
            or any(host in source_url.casefold() for host in allowed_python_hosts)
        )

        if anchor_match and official_match:
            kept.append(item)
        else:
            rejected.append(item)

    if not kept:
        reason = "All retrieved knowledge-base chunks were irrelevant or failed the official-source requirement."
        return False, reason

    # Keep only the relevant chunks. This also prevents irrelevant snippets from
    # leaking into later model context or the final report.
    filtered = {
        "results": kept,
        "rejected_results": len(rejected),
    }
    return True, json.dumps(filtered, ensure_ascii=False)


def meaningful_query_anchors(query: str) -> list[str]:
    stop_words = {
        "what", "when", "where", "which", "who", "why", "how", "the", "a", "an",
        "and", "or", "for", "to", "of", "in", "on", "about", "find", "search",
        "latest", "important", "features", "feature", "information", "details",
        "explain", "show", "give", "with", "from", "official", "website",
    }
    tokens = re.findall(r"[a-zA-Z0-9.]+", query.casefold())
    return [token for token in tokens if len(token) > 2 and token not in stop_words]


def extract_python_version(output: str) -> tuple[str, str] | None:
    """Return (full_version, minor_version) from Python.org page text."""
    text = result_page_text(output)
    patterns = (
        r"Latest:\s*Python\s*(\d+\.\d+\.\d+)",
        r"Python\s+(\d+\.\d+\.\d+)\s+is\s+the\s+latest",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            version = match.group(1)
            major_minor = ".".join(version.split(".")[:2])
            return version, major_minor
    return None


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: WebPilotState) -> dict:
    print("\n========== PLANNER NODE ==========\n")
    print(f"Mission: {state['mission']}")

    plan = create_plan(state["mission"])
    if not plan:
        raise ValueError("Planner failed to generate a plan.")

    print("\nGenerated Plan:\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return {"plan": plan}


# ============================================================
# RESEARCH CONTEXT
# ============================================================

def build_research_context(state: WebPilotState) -> str:
    """Build context exclusively from MCP results, never model narration."""
    evidence_sections = []
    rejected_sections = []
    total_chars = 0

    for index, item in enumerate(state["tool_results"], start=1):
        if item.get("is_error"):
            rejected_sections.append(
                f"{index}. {item.get('tool')} failed: {item.get('output', '')[:500]}"
            )
            continue

        if not item.get("accepted", True):
            rejected_sections.append(
                f"{index}. {item.get('tool')} result rejected: {item.get('rejection_reason', 'not relevant')}"
            )
            continue

        output = item.get("output", "")
        remaining = MAX_TOTAL_EVIDENCE_CHARS - total_chars
        if remaining <= 0:
            break
        output = output[: min(MAX_TOOL_OUTPUT_CHARS, remaining)]
        total_chars += len(output)

        section = (
            f"SOURCE {index}\n"
            f"Tool: {item.get('tool')}\n"
            f"Requested URL/query: {json.dumps(item.get('arguments', {}), ensure_ascii=False)}\n"
            f"Resolved URL: {item.get('source_url') or 'not supplied'}\n"
            f"Tool output:\n{output}"
        )
        evidence_sections.append(section)

    evidence = "\n\n---\n\n".join(evidence_sections) or "No accepted source evidence has been collected yet."
    rejected = "\n".join(rejected_sections) or "None."

    return f"""MISSION:
{state['mission']}

PLAN:
{json.dumps(state['plan'].get('tasks', []), ensure_ascii=False, indent=2)}

ACCEPTED SOURCE EVIDENCE (the only material you may use as facts):
{evidence}

REJECTED/FAILED RETRIEVALS (diagnostics only; do NOT use them as evidence):
{rejected}

Continue the research. Identify which requested deliverables are still unsupported by accepted evidence. If information is missing, make an actual MCP tool call to retrieve it. For software releases, prefer the project's official release notes/changelog over a local knowledge base. For Python, when a version X.Y.Z is found and the mission asks for release features, the official What's New page is:
https://docs.python.org/X.Y/whatsnew/X.Y.html
Use open_website on that URL if it has not already been retrieved. Do not use search_knowledge_base as a substitute for current official release notes.

Do not narrate or simulate tool calls. When you need evidence, return a real tool call. When research is genuinely complete or no further source can be retrieved, return a very short status sentence. Never invent missing findings."""


# ============================================================
# RESEARCH AGENT NODE
# ============================================================

def agent_node(state: WebPilotState) -> dict:
    print("\n========== RESEARCH AGENT NODE ==========\n")

    if state["steps"] >= MAX_RESEARCH_STEPS:
        print("Research step limit reached; moving to report generation.")
        return {"pending_tool_calls": [], "steps": state["steps"]}

    research_tools = [
        tool for tool in state["tools"]
        if tool.get("function", {}).get("name") != "save_report"
    ]

    system_prompt = """You are WebPilot, an autonomous web research agent.
You must use the supplied MCP tools for research. Tool outputs are evidence; your own prior text is not evidence.
Never reveal chain-of-thought or write a step-by-step internal monologue. Do not say that you are going to call a tool: call it.
If the requested facts are not in a retrieved page, browse the relevant official documentation/release notes. A local vector knowledge base is not a web search engine and its results may be unrelated.
Make progress on missing deliverables instead of stopping after the first useful fact. Avoid repeating the same tool call with identical arguments.
When no more research is needed, return only a one-sentence completion status."""

    user_prompt = build_research_context(state)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_EMPTY_RETRIES):
        print(f"Ollama research attempt {attempt + 1}/{MAX_EMPTY_RETRIES}")
        if attempt:
            messages.append({
                "role": "user",
                "content": "Return a real MCP tool call if evidence is still missing; otherwise return a one-sentence completion status. Do not provide analysis or a simulated call.",
            })

        assistant_message = ollama_chat(
            messages,
            tools=research_tools,
            num_predict=1800,
        )

        tool_calls = assistant_message.get("tool_calls", []) or []
        content = strip_think_tags(assistant_message.get("content", ""))

        # IMPORTANT: Deliberately do not store the assistant's content. Qwen3:4b
        # often emits long reasoning text even when think=False. It is not evidence.
        if tool_calls:
            print(f"Model requested {len(tool_calls)} real tool call(s).")
            return {
                "pending_tool_calls": tool_calls,
                "steps": state["steps"] + 1,
            }

        if content:
            print("Model returned a completion/status message; moving to report generation.")
            return {
                "pending_tool_calls": [],
                "steps": state["steps"] + 1,
            }

        print("Ollama returned neither content nor tool calls; retrying.")

    return {
        "pending_tool_calls": [],
        "steps": state["steps"] + 1,
    }


# ============================================================
# MCP TOOLS NODE
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
        "accepted": not is_error,
        "rejection_reason": "",
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
                    "output": f"Invalid JSON tool arguments: {exc}",
                    "is_error": True,
                    "accepted": False,
                    "rejection_reason": "Invalid tool arguments",
                    "source_url": "",
                })
                continue

        if not isinstance(arguments, dict):
            results.append({
                "tool": tool_name,
                "arguments": {},
                "output": "Tool arguments must be a JSON object.",
                "is_error": True,
                "accepted": False,
                "rejection_reason": "Invalid tool arguments",
                "source_url": "",
            })
            continue

        if tool_name == "save_report":
            results.append({
                "tool": tool_name,
                "arguments": arguments,
                "output": "save_report is called only by the graph's report node.",
                "is_error": True,
                "accepted": False,
                "rejection_reason": "Report saving is graph-managed",
                "source_url": "",
            })
            continue

        print(f"Calling MCP tool: {tool_name} with {arguments}")
        try:
            record = await call_mcp_tool(state, tool_name, arguments)
        except Exception as exc:
            record = {
                "tool": tool_name,
                "arguments": arguments,
                "output": str(exc),
                "is_error": True,
                "accepted": False,
                "rejection_reason": "MCP execution error",
                "source_url": "",
            }

        # Guard against unrelated vector-store hits. Do not pass them onward as
        # research evidence and do not let the report writer cite them.
        if tool_name == "search_knowledge_base" and not record["is_error"]:
            query = str(arguments.get("query", ""))
            relevant, filtered_or_reason = knowledge_result_is_relevant(
                record["output"], query, state["mission"]
            )
            if relevant:
                record["output"] = filtered_or_reason
            else:
                record["accepted"] = False
                record["rejection_reason"] = filtered_or_reason

        print("Tool result:")
        print(record["output"][:2500])
        if record.get("rejection_reason"):
            print("Rejected retrieval:", record["rejection_reason"])
        results.append(record)

        # Deterministic Python release-note follow-up. The home page gives us
        # the exact X.Y.Z version; feature details belong to the official X.Y
        # What's New page, not to the local vector database.
        mission_lower = state["mission"].casefold()
        wants_python_release_details = (
            "python" in mission_lower
            and any(word in mission_lower for word in ("feature", "release", "change", "what's new", "whatsnew"))
        )
        if tool_name == "open_website" and wants_python_release_details:
            already_fetched = any(
                "docs.python.org/" in (item.get("source_url") or "")
                and "/whatsnew/" in (item.get("source_url") or "")
                for item in results
            )
            version_info = extract_python_version(record["output"])
            if version_info and not already_fetched:
                full_version, minor_version = version_info
                release_notes_url = (
                    f"https://docs.python.org/{minor_version}/whatsnew/{minor_version}.html"
                )
                print(f"\nAutomatically opening official Python release notes: {release_notes_url}")
                try:
                    release_record = await call_mcp_tool(
                        state,
                        "open_website",
                        {"url": release_notes_url},
                    )
                    release_record["auto_followup"] = True
                    release_record["detected_version"] = full_version
                    results.append(release_record)
                    print("Release notes tool result:")
                    print(release_record["output"][:2500])
                except Exception as exc:
                    results.append({
                        "tool": "open_website",
                        "arguments": {"url": release_notes_url},
                        "output": str(exc),
                        "is_error": True,
                        "accepted": False,
                        "rejection_reason": "Automatic official release-notes fetch failed",
                        "source_url": release_notes_url,
                        "auto_followup": True,
                    })

    return {
        "tool_results": results,
        "pending_tool_calls": [],
    }


# ============================================================
# REPORT GENERATION AND SAVING
# ============================================================

def accepted_evidence(state: WebPilotState) -> list[dict]:
    return [
        item for item in state["tool_results"]
        if item.get("accepted", True) and not item.get("is_error")
    ]


def find_matching_source(quote: str, evidence: list[dict]) -> dict | None:
    normalized_quote = normalize_for_match(quote)
    if not normalized_quote:
        return None
    for item in evidence:
        page_text = normalize_for_match(result_page_text(item.get("output", "")))
        if normalized_quote in page_text:
            return item
    return None


def safe_report_fallback(mission: str, evidence: list[dict]) -> str:
    """Produce a clean, evidence-only report if structured generation fails."""
    lines = [
        f"# Research Report",
        "",
        f"**Mission:** {mission}",
        "",
        "## Confirmed source material",
        "",
    ]

    if not evidence:
        lines.extend([
            "No successful, relevant source material was retrieved.",
            "",
            "## Limitations",
            "",
            "The requested findings could not be verified from the available MCP tool results.",
        ])
    else:
        for item in evidence:
            source = item.get("source_url") or "URL not provided by tool"
            text = result_page_text(item.get("output", ""))
            lines.append(f"### Source: {source}")
            lines.append("")

            # Preserve only short, non-empty lines from the actual page text.
            useful_lines = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line and len(line) > 18 and line not in useful_lines:
                    useful_lines.append(line)
                if len(useful_lines) >= 8:
                    break
            lines.extend(f"- {line[:350]}" for line in useful_lines)
            lines.append("")

        lines.extend([
            "## Limitations",
            "",
            "This report contains excerpts from retrieved sources. Any requested detail not explicitly present in those excerpts remains unverified.",
        ])

    return "\n".join(lines).strip() + "\n"


def get_verified_findings(report_data: dict, evidence: list[dict]) -> list[dict]:
    findings = report_data.get("findings", [])
    if not isinstance(findings, list):
        return []

    verified = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        quote = str(item.get("evidence_quote") or "").strip()
        matching_source = find_matching_source(quote, evidence)
        if not claim or not quote or matching_source is None:
            continue

        source_text = normalize_for_match(
            result_page_text(matching_source.get("output", ""))
        )
        if normalize_for_match(claim) not in source_text:
            continue

        verified.append({
            "claim": claim,
            "quote": quote,
            "source_url": matching_source.get("source_url") or "",
        })
    return verified


def get_latest_python_version(evidence: list[dict]) -> tuple[str, str, str] | None:
    for item in evidence:
        parsed = extract_python_version(item.get("output", ""))
        if not parsed:
            continue
        full_version, _minor = parsed
        quote = f"Latest: Python {full_version}"
        if quote.casefold() in normalize_for_match(result_page_text(item.get("output", ""))):
            return full_version, quote, item.get("source_url") or ""
    return None


def build_markdown_report(mission: str, report_data: dict, evidence: list[dict]) -> str:
    verified_findings = get_verified_findings(report_data, evidence)
    version_info = get_latest_python_version(evidence)
    mission_lower = mission.casefold()

    lines = [
        "# Research Report",
        "",
        f"**Research mission:** {mission}",
        "",
    ]

    if "version" in mission_lower or "latest" in mission_lower:
        lines.extend(["## Latest version", ""])
        if version_info:
            version, quote, source_url = version_info
            lines.append(f"**Python {version}**")
            lines.append("")
            lines.append(f"> {quote}")
            lines.append("")
            if source_url:
                lines.append(f"Source: {source_url}")
                lines.append("")
        else:
            lines.extend(["The latest version could not be verified from the retrieved sources.", ""])

    lines.extend(["## Features", ""])
    if verified_findings:
        for index, finding in enumerate(verified_findings, start=1):
            lines.append(f"### {index}. {finding['claim']}")
            lines.append("")
            lines.append(f"> {finding['quote']}")
            lines.append("")
            if finding["source_url"]:
                lines.append(f"Source: {finding['source_url']}")
                lines.append("")
    else:
        lines.extend([
            "No feature finding passed the source-text verification checks.",
            "",
        ])

    if "three" in mission_lower or re.search(r"\b3\s+(?:important\s+)?features?\b", mission_lower):
        if len(verified_findings) < 3:
            lines.extend([
                "## Limitations",
                "",
                f"The mission requested three features, but only {len(verified_findings)} source-verified feature(s) were extracted. Missing items have not been guessed.",
                "",
            ])

    # Clarify the distinction between the current patch release and series-level
    # new features when Python's What's New page is part of the evidence.
    has_python_whatsnew = any(
        "docs.python.org/" in (item.get("source_url") or "")
        and "/whatsnew/" in (item.get("source_url") or "")
        for item in evidence
    )
    if version_info and has_python_whatsnew:
        version_series = ".".join(version_info[0].split(".")[:2])
        lines.extend([
            "## Versioning note",
            "",
            f"The version above is the latest patch release shown on Python.org. The feature list comes from the Python {version_series} series release notes and describes changes in that series, not necessarily changes introduced specifically in the latest patch release.",
            "",
        ])

    source_urls = []
    for item in evidence:
        url = item.get("source_url")
        if url and url not in source_urls:
            source_urls.append(url)

    if source_urls:
        lines.extend(["## Sources", ""])
        lines.extend(f"- {url}" for url in source_urls)
        lines.append("")

    if not evidence:
        lines.extend([
            "## Limitations",
            "",
            "No successful, relevant source material was retrieved. The requested findings remain unverified.",
            "",
        ])

    return "\n".join(lines).strip() + "\n"


async def report_node(state: WebPilotState) -> dict:
    print("\n========== REPORT NODE ==========\n")

    if state.get("report_saved"):
        return {}

    evidence = accepted_evidence(state)
    evidence_payload = []
    for item in evidence:
        evidence_payload.append({
            "tool": item.get("tool"),
            "source_url": item.get("source_url"),
            "output": item.get("output", "")[:MAX_TOOL_OUTPUT_CHARS],
        })

    report_prompt = f"""Extract source-verified findings for this mission:
{state['mission']}

Use only the source material in the JSON below. Do not use prior knowledge. Do not discuss your reasoning, the user, the tools, or the research process. Do not invent missing details.
For every finding, provide:
- claim: a short phrase copied verbatim from the source text (the program checks this)
- evidence_quote: an exact, continuous quote copied verbatim from the same source
- source_url: the URL of that source
If fewer findings are supported than requested, return only supported findings. Never pad the list.

Return valid JSON only, with this structure:
{{
  "findings": [
    {{"claim": "...", "evidence_quote": "...", "source_url": "..."}}
  ]
}}

SOURCE MATERIAL:
{json.dumps(evidence_payload, ensure_ascii=False)}"""

    report_data = None
    if evidence:
        try:
            report_message = ollama_chat(
                [
                    {"role": "system", "content": "You are a structured research-report writer. Return only JSON matching the requested structure."},
                    {"role": "user", "content": report_prompt},
                ],
                num_predict=2600,
                json_mode=True,
            )
            report_data = extract_json_object(report_message.get("content", ""))
        except Exception as exc:
            print(f"Structured report generation failed; using source-only fallback: {exc}")

    if not report_data:
        report_data = {"findings": []}

    report_content = build_markdown_report(
        state["mission"], report_data, evidence
    )

    title = re.sub(r"[^a-zA-Z0-9]+", "_", state["mission"].lower()).strip("_")[:60]
    filename = f"{title or 'research_report'}.md"

    report_client = state["tool_to_client"].get("save_report")
    if report_client is None:
        raise RuntimeError("MCP client for save_report was not found.")

    save_result = await report_client.call_tool(
        "save_report",
        {"filename": filename, "content": report_content},
    )
    save_output, save_error = extract_tool_text(save_result)
    print("Save report result:", save_output)

    # MCP may expose a returned dict as text content or structured content.
    save_data = parse_json_text(save_output)
    saved = (
        not save_error
        and isinstance(save_data, dict)
        and save_data.get("status") == "saved"
        and save_data.get("filename") == filename
    )

    if not saved:
        final_output = f"Report save was not confirmed. MCP response: {save_output}"
        return {
            "report_saved": False,
            "report_filename": filename,
            "final_output": final_output,
        }

    verified_count = len(get_verified_findings(report_data, evidence)) if report_data else 0

    final_output = (
        f"Research report saved successfully: `{filename}`\n"
        f"Accepted source documents: {len(evidence)}\n"
        f"Source-verified findings: {verified_count}\n\n"
        f"{report_content}"
    )

    return {
        "report_saved": True,
        "report_filename": filename,
        "final_output": final_output,
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_agent(state: WebPilotState):
    print("\n========== ROUTING ==========\n")

    if state["steps"] >= MAX_RESEARCH_STEPS:
        print("Research step limit reached; moving to report generation.")
        return "report"

    if state.get("pending_tool_calls"):
        print("Routing to MCP tools node.")
        return "tools"

    print("No further tool calls; moving to report generation.")
    return "report"


# ============================================================
# CONNECT MCP SERVERS
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
# BUILD LANGGRAPH
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
        {
            "tools": "tools",
            "report": "report",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("report", END)
    return builder.compile()


# ============================================================
# MAIN
# ============================================================

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
