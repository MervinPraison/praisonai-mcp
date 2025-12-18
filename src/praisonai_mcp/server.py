"""PraisonAI MCP Server implementation.

This module creates an MCP server that exposes useful tools
for AI assistants like Claude Desktop and Cursor.
"""

import argparse
import os
from typing import Dict, Any, List, Optional


# Built-in tools that are always available

def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web for information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        # Try Tavily first
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=max_results)
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:200]
                    }
                    for r in response.get("results", [])
                ]
            }
    except Exception:
        pass
    
    # Fallback to DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:200]
                    }
                    for r in results
                ]
            }
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3")
    
    Returns:
        The result of the calculation
    """
    import ast
    import operator
    
    # Safe operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            return operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Get the current date and time.
    
    Args:
        timezone: Timezone name (e.g., "UTC", "America/New_York", "Europe/London")
    
    Returns:
        Current date and time information
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
    except Exception:
        now = datetime.utcnow()
        timezone = "UTC"
    
    return {
        "timezone": timezone,
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


def read_file(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read the contents of a file.
    
    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)
    
    Returns:
        File contents and metadata
    """
    import os
    
    try:
        # Security: Only allow reading from current directory and subdirectories
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if not abs_path.startswith(cwd):
            return {"error": "Access denied: Can only read files in current directory"}
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "path": path,
            "size": len(content),
            "lines": content.count('\n') + 1,
            "content": content[:10000]  # Limit content size
        }
    except Exception as e:
        return {"path": path, "error": str(e)}


def list_directory(path: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """List files and directories in a path.
    
    Args:
        path: Directory path to list (default: current directory)
        pattern: Glob pattern to filter files (default: *)
    
    Returns:
        List of files and directories with metadata
    """
    import os
    import glob
    
    try:
        # Security: Only allow listing current directory and subdirectories
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if not abs_path.startswith(cwd):
            return {"error": "Access denied: Can only list current directory"}
        
        full_pattern = os.path.join(path, pattern)
        items = glob.glob(full_pattern)
        
        result = []
        for item in items[:100]:  # Limit results
            stat = os.stat(item)
            result.append({
                "name": os.path.basename(item),
                "path": item,
                "is_dir": os.path.isdir(item),
                "size": stat.st_size if os.path.isfile(item) else None,
            })
        
        return {
            "path": path,
            "pattern": pattern,
            "count": len(result),
            "items": result
        }
    except Exception as e:
        return {"path": path, "error": str(e)}


def run_python(code: str) -> Dict[str, Any]:
    """Execute Python code and return the result.
    
    Args:
        code: Python code to execute
    
    Returns:
        Execution result or error
    """
    import io
    
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Create isolated namespace
        namespace = {"__builtins__": __builtins__}
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
        
        # Get the last expression value if any
        result = namespace.get("result", namespace.get("output", None))
        
        return {
            "success": True,
            "stdout": stdout_capture.getvalue()[:5000],
            "stderr": stderr_capture.getvalue()[:1000],
            "result": str(result) if result is not None else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": stdout_capture.getvalue()[:5000],
            "stderr": stderr_capture.getvalue()[:1000]
        }


def create_server(
    name: str = "praisonai-mcp",
    include_tools: Optional[List[str]] = None,
    extra_tools: Optional[List] = None,
    debug: bool = False
):
    """Create a PraisonAI MCP server with specified tools.
    
    Args:
        name: Name of the MCP server
        include_tools: List of built-in tool names to include (default: all)
        extra_tools: Additional custom tool functions to register
        debug: Enable debug logging
    
    Returns:
        Configured ToolsMCPServer instance
    """
    from praisonaiagents.mcp import ToolsMCPServer
    
    # All available built-in tools
    all_tools = {
        "search_web": search_web,
        "calculate": calculate,
        "get_current_time": get_current_time,
        "read_file": read_file,
        "list_directory": list_directory,
        "run_python": run_python,
    }
    
    # Select tools to include
    if include_tools:
        tools = [all_tools[name] for name in include_tools if name in all_tools]
    else:
        tools = list(all_tools.values())
    
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
        "--tools",
        type=str,
        nargs="+",
        help="Specific tools to enable (default: all)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Create server
    server = create_server(
        include_tools=args.tools,
        debug=args.debug
    )
    
    # Run server
    if args.sse:
        server.run_sse(host=args.host, port=args.port)
    else:
        server.run_stdio()


if __name__ == "__main__":
    main()
