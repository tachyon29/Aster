
import asyncio
import sys
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters


async def main():

    project_dir = Path(__file__).parent

    servers = {
        "browser": "mcp_server.py",
        "reports": "report_server.py"
    }

    async with AsyncExitStack() as stack:

        clients = {}
        all_tools = []

        # Connect to every MCP server
        for server_name, filename in servers.items():

            params = StdioServerParameters(
                command=sys.executable,
                args=[str(project_dir / filename)]
            )

            client = await stack.enter_async_context(
                Client(params)
            )

            clients[server_name] = client

            discovered = await client.list_tools()

            print(f"\n{server_name} server tools:")

            for tool in discovered.tools:

                print("-", tool.name)

                all_tools.append({
                    "name": tool.name,
                    "server": server_name,
                    "schema": tool
                })

        print("\nTotal discovered tools:", len(all_tools))

        # Test calling the report server
        report_client = clients["reports"]

        result = await report_client.call_tool(
            "save_report",
            {
                "filename": "test_report.md",
                "content": "# WebPilot\n\nMy first MCP report!"
            }
        )

        print("\nReport server response:", result)


if __name__ == "__main__":
    asyncio.run(main())
