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
    research_spec: dict
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
    """Reject only empty KB responses; semantic relevance is query-aware.

    Do not use keyword overlap as a hard rejection rule: relevant results may
    use synonyms or terminology different from the user's wording. The agent
    and the final extraction model receive the mission and decide relevance.
    """
    parsed = parse_json_text(output)
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        if not parsed["results"]:
            return False, "The knowledge base returned no results."
        return True, output

    if not output.strip():
        return False, "The knowledge base returned an empty response."

    return True, output



# ============================================================
# QUERY-AWARE INTENT PLANNING
# ============================================================

def create_research_spec(mission: str) -> dict:
    """Infer what information the user wants before research begins."""
    prompt = f"""Read the user's research mission and identify exactly what information
WebPilot should collect. This is query understanding, not research and not fact
extraction.

User mission:
{mission}

Return JSON with this structure:
{{
  "objective": "one-sentence description of the user's goal",
  "requested_content_types": ["the kinds of information requested"],
  "required_details": ["specific details or fields the user asks for"],
  "requested_count": null,
  "preferred_structure": "how the answer/report should be organized",
  "source_requirements": ["any source constraints explicitly requested or clearly needed"],
  "explicit_exclusions": ["only things the user explicitly says to omit"]
}}

Rules:
- Interpret the actual mission; do not assume the user wants features, versions,
  summaries, or any other fixed information type.
- If the user asks for versions, numbers and dates may be essential.
- If the user asks for an explanation, general descriptions may be essential.
- Do not invent exclusions. Keep explicit_exclusions empty unless the mission
  actually excludes something.
- requested_count must be an integer only when the user specifies a count;
  otherwise use null.
- Return valid JSON only."""

    fallback = {
        "objective": mission,
        "requested_content_types": ["Information directly relevant to the mission"],
        "required_details": [],
        "requested_count": None,
        "preferred_structure": "Organize relevant findings into clear sections",
        "source_requirements": [],
        "explicit_exclusions": [],
    }

    try:
        message = ollama_chat(
            [
                {"role": "system", "content": "You are a precise research-intent planner. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            num_predict=1000,
            json_mode=True,
        )
        parsed = extract_json_object(message.get("content", ""))
        if not isinstance(parsed, dict):
            print("Could not parse the research specification; using a safe generic specification.")
            return fallback

        # Normalize the fields so later nodes can rely on predictable types.
        spec = {
            "objective": str(parsed.get("objective") or mission).strip(),
            "requested_content_types": parsed.get("requested_content_types", []),
            "required_details": parsed.get("required_details", []),
            "requested_count": parsed.get("requested_count"),
            "preferred_structure": str(parsed.get("preferred_structure") or "Organize findings into clear sections").strip(),
            "source_requirements": parsed.get("source_requirements", []),
            "explicit_exclusions": parsed.get("explicit_exclusions", []),
        }

        for key in ("requested_content_types", "required_details", "source_requirements", "explicit_exclusions"):
            value = spec[key]
            if not isinstance(value, list):
                spec[key] = []
            else:
                spec[key] = [str(item).strip() for item in value if str(item).strip()]

        count = spec["requested_count"]
        if isinstance(count, str) and count.strip().isdigit():
            count = int(count.strip())
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            spec["requested_count"] = None
        else:
            spec["requested_count"] = count

        return spec
    except Exception as exc:
        print(f"Research-intent planning failed; using generic specification: {exc}")
        return fallback


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: WebPilotState) -> dict:
    print("\n========== PLANNER NODE ==========\n")
    print(f"Mission: {state['mission']}")

    plan = create_plan(state["mission"])
    if not plan:
        raise ValueError("Planner failed to generate a plan.")

    research_spec = create_research_spec(state["mission"])

    print("\nGenerated Research Plan:\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print("\nQuery-Aware Research Specification:\n")
    print(json.dumps(research_spec, indent=2, ensure_ascii=False))
    return {"plan": plan, "research_spec": research_spec}


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

QUERY-AWARE RESEARCH SPECIFICATION:
{json.dumps(state.get('research_spec', {}), ensure_ascii=False, indent=2)}

PLAN:
{json.dumps(state['plan'].get('tasks', []), ensure_ascii=False, indent=2)}

ACCEPTED SOURCE EVIDENCE (the only material you may use as facts):
{evidence}

REJECTED/FAILED RETRIEVALS (diagnostics only; do NOT use them as evidence):
{rejected}

Continue researching until the requested information types and required details
in the research specification are adequately supported. Choose sources appropriate
to the mission; prefer primary or official sources when they are available and
relevant. Do not apply universal exclusions: numbers, dates, versions, technical
features, general descriptions, examples, or other information may all be relevant
depending on the user's request. Exclude something only when it is irrelevant to
the mission or explicitly excluded by the user.

If information is missing, make an actual MCP tool call to retrieve it. Do not
narrate or simulate tool calls. When research is genuinely complete or no further
source can be retrieved, return a very short status sentence. Never invent missing
findings."""


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

        # This is only a retrieval sanity check. It does not classify content
        # as a feature/version/description; the mission-aware extractor does that.
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


def validate_report_sections(
    report_data: dict,
    evidence: list[dict],
    requested_count: int | None = None,
) -> list[dict]:
    """Keep any mission-relevant item whose evidence quote exists in a source.

    No content category is globally accepted or rejected here. In particular,
    numeric values, versions, dates, descriptions, and feature claims are all
    treated the same; the extraction LLM decides relevance from the mission.
    Python performs only traceability, schema, and duplicate checks.
    """
    raw_sections = report_data.get("sections", [])
    if not isinstance(raw_sections, list):
        raw_sections = []

    # Compatibility with older model output schemas.
    if not raw_sections and isinstance(report_data.get("findings"), list):
        raw_sections = [{"heading": "Findings", "items": report_data["findings"]}]

    validated_sections = []
    seen_items = set()
    remaining_items = requested_count if isinstance(requested_count, int) and requested_count > 0 else None

    for raw_section in raw_sections:
        if remaining_items is not None and remaining_items <= 0:
            break
        if not isinstance(raw_section, dict):
            continue

        heading = str(
            raw_section.get("heading")
            or raw_section.get("title")
            or raw_section.get("category")
            or "Findings"
        ).strip()
        raw_items = raw_section.get("items", raw_section.get("findings", []))
        if not isinstance(raw_items, list):
            continue

        valid_items = []
        for raw_item in raw_items:
            if remaining_items is not None and remaining_items <= 0:
                break
            if not isinstance(raw_item, dict):
                continue

            title = str(raw_item.get("title") or raw_item.get("claim") or "").strip()
            description = str(
                raw_item.get("description")
                or raw_item.get("summary")
                or raw_item.get("detail")
                or ""
            ).strip()
            quote = str(
                raw_item.get("evidence_quote")
                or raw_item.get("quote")
                or ""
            ).strip()

            if not title or not quote:
                continue

            matching_source = find_matching_source(quote, evidence)
            if matching_source is None:
                continue

            # Deduplicate identical titles within/across sections without
            # imposing any semantic preference for one kind of content.
            dedupe_key = (
                normalize_for_match(title),
                normalize_for_match(quote),
            )
            if dedupe_key in seen_items:
                continue
            seen_items.add(dedupe_key)

            valid_items.append({
                "title": title,
                "description": description,
                "quote": quote,
                "source_url": matching_source.get("source_url") or "",
            })
            if remaining_items is not None:
                remaining_items -= 1

        if valid_items:
            validated_sections.append({
                "heading": heading,
                "items": valid_items,
            })

    return validated_sections


def count_validated_items(sections: list[dict]) -> int:
    return sum(len(section.get("items", [])) for section in sections)


def build_markdown_report(
    mission: str,
    sections: list[dict],
    evidence: list[dict],
    research_spec: dict,
) -> str:
    lines = [
        "# Research Report",
        "",
        f"**Research mission:** {mission}",
        "",
    ]

    if sections:
        for section in sections:
            lines.extend([f"## {section['heading']}", ""])
            for index, item in enumerate(section["items"], start=1):
                lines.append(f"### {index}. {item['title']}")
                lines.append("")
                if item.get("description"):
                    lines.extend([item["description"], ""])
                lines.extend([f"> {item['quote']}", ""])
                if item.get("source_url"):
                    lines.extend([f"Source: {item['source_url']}", ""])
    else:
        lines.extend([
            "## Findings",
            "",
            "No extracted finding passed the source-evidence checks.",
            "",
        ])

    requested_count = research_spec.get("requested_count")
    actual_count = count_validated_items(sections)
    if isinstance(requested_count, int) and requested_count > actual_count:
        lines.extend([
            "## Limitations",
            "",
            f"The mission requested {requested_count} item(s), but only {actual_count} item(s) could be verified against retrieved source text. Missing items have not been invented.",
            "",
        ])

    if not evidence:
        lines.extend([
            "## Limitations",
            "",
            "No successful source material was retrieved. The requested information remains unverified.",
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

    extraction_prompt = f"""You are extracting research findings for a user's specific mission.

USER MISSION:
{state['mission']}

QUERY-AWARE RESEARCH SPECIFICATION:
{json.dumps(state.get('research_spec', {}), ensure_ascii=False, indent=2)}

Use only the source material below. Extract the information the user actually
requested. The requested information types are determined by the mission, not
by a fixed template.

Important:
- Do not assume the answer must contain features.
- Do not automatically discard numbers, version identifiers, dates, general
descriptions, examples, or any other category. Keep them when they answer the
mission; omit them only when irrelevant or explicitly excluded.
- Group the extracted information into useful sections whose headings reflect
the user's request.
- Each item must have a concise title, a useful description, an exact continuous
evidence quote copied from one supplied source, and that source's URL.
- The description may explain the quote in your own words, but it must not add
facts unsupported by the supplied source.
- Do not combine unrelated facts into one item.
- If the user requested a specific number of items, return up to that number of
well-supported items. Do not pad the result with weak or invented items.
- If information is not present in the sources, do not invent it.

Return valid JSON only, with this structure:
{{
  "sections": [
    {{
      "heading": "A section name suited to the mission",
      "items": [
        {{
          "title": "A concise finding title",
          "description": "A source-supported explanation",
          "evidence_quote": "An exact continuous quote from the source",
          "source_url": "The exact URL of that source"
        }}
      ]
    }}
  ]
}}

SOURCE MATERIAL:
{json.dumps(evidence_payload, ensure_ascii=False)}"""

    report_data = {"sections": []}
    if evidence:
        try:
            report_message = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": "You are a careful, query-aware research extractor. Return only JSON matching the requested schema.",
                    },
                    {"role": "user", "content": extraction_prompt},
                ],
                num_predict=3200,
                json_mode=True,
            )
            parsed_report = extract_json_object(report_message.get("content", ""))
            if isinstance(parsed_report, dict):
                report_data = parsed_report
        except Exception as exc:
            print(f"Structured report extraction failed: {exc}")

    # Mechanical validation only: schema, exact source quote, source attribution,
    # and duplicate titles. There is no second LLM judgement/filtering call.
    validated_sections = validate_report_sections(
        report_data,
        evidence,
        state.get("research_spec", {}).get("requested_count"),
    )
    report_content = build_markdown_report(
        state["mission"],
        validated_sections,
        evidence,
        state.get("research_spec", {}),
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

    verified_count = count_validated_items(validated_sections)

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
            "research_spec": {},
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
