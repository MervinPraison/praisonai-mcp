"""PraisonAI MCP Server - Expose PraisonAI tools as MCP server.

This module provides an MCP server that exposes PraisonAI tools
for use with Claude Desktop, Cursor, and other MCP clients.

Usage:
    # Run as stdio server (for Claude Desktop)
    python -m praisonai_mcp
    
    # Run as SSE server (for web clients)
    python -m praisonai_mcp --sse --port 8080
"""

from .server import main, create_server

__version__ = "0.1.0"
__all__ = ["main", "create_server"]
