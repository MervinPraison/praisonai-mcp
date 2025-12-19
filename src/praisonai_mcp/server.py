"""PraisonAI MCP Server implementation.

This module creates an MCP server that exposes all PraisonAI tools
for AI assistants like Claude Desktop and Cursor.
"""

import argparse
from typing import List, Optional, Callable

from .tools import ALL_TOOLS, CORE_TOOLS, FILE_TOOLS, AGENT_TOOLS, MEMORY_TOOLS, TODO_TOOLS, CODE_TOOLS


def create_server(
    name: str = "praisonai-mcp",
    include_tools: Optional[List[str]] = None,
    extra_tools: Optional[List[Callable]] = None,
    debug: bool = False
):
    """Create a PraisonAI MCP server with specified tools.
    
    Args:
        name: Name of the MCP server
        include_tools: List of tool category names to include (default: all)
                      Options: core, file, agent, memory, knowledge, todo, workflow, code, mcp, session
        extra_tools: Additional custom tool functions to register
        debug: Enable debug logging
    
    Returns:
        Configured ToolsMCPServer instance
    """
    from praisonaiagents.mcp import ToolsMCPServer
    
    # Tool categories
    tool_categories = {
        "core": CORE_TOOLS,
        "file": FILE_TOOLS,
        "agent": AGENT_TOOLS,
        "memory": MEMORY_TOOLS,
        "todo": TODO_TOOLS,
        "code": CODE_TOOLS,
    }
    
    # Select tools to include
    if include_tools:
        tools = []
        for category in include_tools:
            if category in tool_categories:
                tools.extend(tool_categories[category])
    else:
        tools = list(ALL_TOOLS)
    
    # Add extra tools
    if extra_tools:
        tools.extend(extra_tools)
    
    # Create server
    server = ToolsMCPServer(name=name, tools=tools, debug=debug)
    
    return server


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="PraisonAI MCP Server - Expose AI tools for Claude Desktop, Cursor, etc."
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use SSE transport instead of stdio"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE server (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for SSE server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=["core", "file", "agent", "memory", "todo", "code", "all"],
        default=["all"],
        help="Tool categories to enable (default: all)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Determine which tools to include
    include_tools = None
    if "all" not in args.categories:
        include_tools = args.categories
    
    # Create server
    server = create_server(
        include_tools=include_tools,
        debug=args.debug
    )
    
    # Print tool info
    tool_names = server.get_tool_names()
    print(f"🚀 PraisonAI MCP Server")
    print(f"📦 {len(tool_names)} tools available:")
    for name in tool_names:
        print(f"   • {name}")
    print()
    
    # Run server
    if args.sse:
        server.run_sse(host=args.host, port=args.port)
    else:
        server.run_stdio()


if __name__ == "__main__":
    main()
