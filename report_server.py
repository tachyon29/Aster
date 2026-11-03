
from pathlib import Path
from mcp.server import MCPServer

mcp = MCPServer("WebPilot Report Server")

REPORT_DIR = Path(__file__).parent / "reports"


@mcp.tool()
def save_report(filename: str, content: str) -> dict:
    """Save a Markdown research report in the reports directory."""

    # Restrict filenames to simple Markdown filenames
    if (
        not filename.endswith(".md")
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        return {"error": "Invalid report filename"}

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / filename
    report_path.write_text(content, encoding="utf-8")

    return {
        "status": "saved",
        "filename": filename,
        "path": str(report_path)
    }


@mcp.tool()
def list_reports() -> list:
    """List existing Markdown research reports."""

    REPORT_DIR.mkdir(exist_ok=True)

    return [
        path.name
        for path in REPORT_DIR.glob("*.md")
        if path.is_file()
    ]


if __name__ == "__main__":
    mcp.run()
