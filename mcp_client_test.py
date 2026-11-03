
import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def main():
    project_dir = Path(__file__).parent

    # Tell the MCP client how to launch our server
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(project_dir / "mcp_server.py")]
    )

    # Connect to the MCP server
    async with Client(server) as client:

        print("\nConnected to MCP server!")

        # Discover available tools
        result = await client.list_tools()

        print("\nDiscovered tools:")

        for tool in result.tools:
            print(f"\nName: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Input schema: {tool.input_schema}")

        # Call a discovered tool
        print("\nCalling open_website...")

        response = await client.call_tool(
            "open_website",
            {"url": "https://example.com"}
        )

        print("\nTool response:")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
