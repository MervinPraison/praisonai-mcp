"""PraisonAI MCP Tools - All CLI commands as MCP tools.

This module exposes all PraisonAI CLI commands as MCP tools
for use with Claude Desktop, Cursor, and other MCP clients.
"""

import os
import subprocess
import json
from typing import Dict, Any, List, Optional


# =============================================================================
# CORE TOOLS - Basic utilities
# =============================================================================

def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web for information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
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


# =============================================================================
# FILE TOOLS - File operations
# =============================================================================

def read_file(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read the contents of a file.
    
    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)
    
    Returns:
        File contents and metadata
    """
    try:
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
            "content": content[:10000]
        }
    except Exception as e:
        return {"path": path, "error": str(e)}


def write_file(path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Write content to a file.
    
    Args:
        path: Path to the file to write
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
    
    Returns:
        Status of the write operation
    """
    try:
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if not abs_path.startswith(cwd):
            return {"error": "Access denied: Can only write files in current directory"}
        
        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return {
            "path": path,
            "size": len(content),
            "lines": content.count('\n') + 1,
            "success": True
        }
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def list_directory(path: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """List files and directories in a path.
    
    Args:
        path: Directory path to list (default: current directory)
        pattern: Glob pattern to filter files (default: *)
    
    Returns:
        List of files and directories with metadata
    """
    import glob
    
    try:
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if not abs_path.startswith(cwd):
            return {"error": "Access denied: Can only list current directory"}
        
        full_pattern = os.path.join(path, pattern)
        items = glob.glob(full_pattern)
        
        result = []
        for item in items[:100]:
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


# =============================================================================
# AGENT TOOLS - AI Agent operations
# =============================================================================

def run_agent(prompt: str, model: str = "gpt-4o-mini", verbose: bool = False) -> Dict[str, Any]:
    """Run a PraisonAI agent with a prompt.
    
    Args:
        prompt: The task or question for the agent
        model: LLM model to use (default: gpt-4o-mini)
        verbose: Enable verbose output
    
    Returns:
        Agent response
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(
            instructions="You are a helpful AI assistant.",
            llm=model,
            verbose=verbose
        )
        result = agent.start(prompt)
        
        return {
            "prompt": prompt,
            "model": model,
            "response": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"prompt": prompt, "error": str(e), "success": False}


def run_research(query: str, model: str = "gpt-4o-mini", verbose: bool = False) -> Dict[str, Any]:
    """Run deep research on a topic using PraisonAI.
    
    Args:
        query: Research query or topic
        model: LLM model to use (default: gpt-4o-mini)
        verbose: Enable verbose output
    
    Returns:
        Research results
    """
    try:
        from praisonaiagents import Agent
        from praisonaiagents.tools import duckduckgo_search
        
        agent = Agent(
            instructions="""You are a research assistant. 
            Search for information and provide comprehensive, well-structured answers.
            Include sources and citations where possible.""",
            llm=model,
            tools=[duckduckgo_search],
            verbose=verbose
        )
        result = agent.start(query)
        
        return {
            "query": query,
            "model": model,
            "research": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def generate_agents_yaml(topic: str, framework: str = "praisonai") -> Dict[str, Any]:
    """Generate an agents.yaml file for a given topic.
    
    Args:
        topic: Topic or task description for agent generation
        framework: Framework to use (praisonai, crewai, autogen)
    
    Returns:
        Generated YAML content and file path
    """
    try:
        from praisonai.auto import AutoGenerator
        
        generator = AutoGenerator(
            topic=topic,
            framework=framework,
            agent_file="agents.yaml"
        )
        file_path = generator.generate()
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return {
            "topic": topic,
            "framework": framework,
            "file_path": file_path,
            "content": content,
            "success": True
        }
    except ImportError:
        return {"error": "praisonai not installed. Run: pip install praisonai"}
    except Exception as e:
        return {"topic": topic, "error": str(e), "success": False}


# =============================================================================
# MEMORY TOOLS - Memory management
# =============================================================================

def memory_add(content: str, user_id: str = "default") -> Dict[str, Any]:
    """Add a memory to the agent's memory store.
    
    Args:
        content: Memory content to store
        user_id: User ID for memory isolation
    
    Returns:
        Status of the memory operation
    """
    try:
        from praisonaiagents.memory import Memory
        
        memory = Memory(user_id=user_id)
        memory.add(content)
        
        return {
            "content": content[:100] + "..." if len(content) > 100 else content,
            "user_id": user_id,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


def memory_search(query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
    """Search memories for relevant content.
    
    Args:
        query: Search query
        user_id: User ID for memory isolation
        limit: Maximum number of results
    
    Returns:
        Matching memories
    """
    try:
        from praisonaiagents.memory import Memory
        
        memory = Memory(user_id=user_id)
        results = memory.search(query, limit=limit)
        
        return {
            "query": query,
            "user_id": user_id,
            "results": results,
            "count": len(results),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def memory_list(user_id: str = "default") -> Dict[str, Any]:
    """List all memories for a user.
    
    Args:
        user_id: User ID for memory isolation
    
    Returns:
        All stored memories
    """
    try:
        from praisonaiagents.memory import Memory
        
        memory = Memory(user_id=user_id)
        all_memories = memory.get_all()
        
        return {
            "user_id": user_id,
            "memories": all_memories,
            "count": len(all_memories),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


def memory_clear(user_id: str = "default") -> Dict[str, Any]:
    """Clear all memories for a user.
    
    Args:
        user_id: User ID for memory isolation
    
    Returns:
        Status of the clear operation
    """
    try:
        from praisonaiagents.memory import Memory
        
        memory = Memory(user_id=user_id)
        memory.clear()
        
        return {
            "user_id": user_id,
            "cleared": True,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# KNOWLEDGE TOOLS - Knowledge base operations
# =============================================================================

def knowledge_add(content: str, source: str = "manual") -> Dict[str, Any]:
    """Add content to the knowledge base.
    
    Args:
        content: Content to add to knowledge base
        source: Source identifier for the content
    
    Returns:
        Status of the operation
    """
    try:
        from praisonaiagents.knowledge import Knowledge
        
        kb = Knowledge()
        kb.add(content, source=source)
        
        return {
            "content": content[:100] + "..." if len(content) > 100 else content,
            "source": source,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


def knowledge_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search the knowledge base.
    
    Args:
        query: Search query
        limit: Maximum number of results
    
    Returns:
        Matching knowledge entries
    """
    try:
        from praisonaiagents.knowledge import Knowledge
        
        kb = Knowledge()
        results = kb.search(query, limit=limit)
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


# =============================================================================
# TODO TOOLS - Task management
# =============================================================================

def todo_add(task: str, priority: str = "medium") -> Dict[str, Any]:
    """Add a task to the todo list.
    
    Args:
        task: Task description
        priority: Priority level (low, medium, high)
    
    Returns:
        Created task details
    """
    try:
        from praisonaiagents.tools import todo_add as _todo_add
        result = _todo_add(task, priority)
        return {"task": task, "priority": priority, "result": result, "success": True}
    except ImportError:
        # Fallback to simple file-based todo
        todo_file = os.path.expanduser("~/.praisonai_todo.json")
        try:
            with open(todo_file, 'r') as f:
                todos = json.load(f)
        except:
            todos = []
        
        todo = {
            "id": len(todos) + 1,
            "task": task,
            "priority": priority,
            "status": "pending"
        }
        todos.append(todo)
        
        with open(todo_file, 'w') as f:
            json.dump(todos, f, indent=2)
        
        return {"task": task, "priority": priority, "id": todo["id"], "success": True}
    except Exception as e:
        return {"task": task, "error": str(e), "success": False}


def todo_list(status: str = "all") -> Dict[str, Any]:
    """List all tasks in the todo list.
    
    Args:
        status: Filter by status (all, pending, completed)
    
    Returns:
        List of tasks
    """
    try:
        from praisonaiagents.tools import todo_list as _todo_list
        result = _todo_list(status)
        return {"status": status, "tasks": result, "success": True}
    except ImportError:
        todo_file = os.path.expanduser("~/.praisonai_todo.json")
        try:
            with open(todo_file, 'r') as f:
                todos = json.load(f)
        except:
            todos = []
        
        if status != "all":
            todos = [t for t in todos if t.get("status") == status]
        
        return {"status": status, "tasks": todos, "count": len(todos), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def todo_complete(task_id: int) -> Dict[str, Any]:
    """Mark a task as completed.
    
    Args:
        task_id: ID of the task to complete
    
    Returns:
        Updated task details
    """
    try:
        from praisonaiagents.tools import todo_complete as _todo_complete
        result = _todo_complete(task_id)
        return {"task_id": task_id, "result": result, "success": True}
    except ImportError:
        todo_file = os.path.expanduser("~/.praisonai_todo.json")
        try:
            with open(todo_file, 'r') as f:
                todos = json.load(f)
        except:
            return {"task_id": task_id, "error": "No todos found", "success": False}
        
        for todo in todos:
            if todo.get("id") == task_id:
                todo["status"] = "completed"
                with open(todo_file, 'w') as f:
                    json.dump(todos, f, indent=2)
                return {"task_id": task_id, "task": todo["task"], "success": True}
        
        return {"task_id": task_id, "error": "Task not found", "success": False}
    except Exception as e:
        return {"task_id": task_id, "error": str(e), "success": False}


# =============================================================================
# WORKFLOW TOOLS - Workflow management
# =============================================================================

def workflow_run(steps: str, variables: Dict[str, str] = None) -> Dict[str, Any]:
    """Run a workflow with specified steps.
    
    Args:
        steps: Workflow steps in format 'step1:action1,step2:action2'
        variables: Variables to pass to the workflow
    
    Returns:
        Workflow execution results
    """
    try:
        from praisonaiagents import Agent
        
        step_list = [s.strip() for s in steps.split(',')]
        results = []
        
        agent = Agent(instructions="Execute workflow steps efficiently.")
        
        for step in step_list:
            if ':' in step:
                name, action = step.split(':', 1)
            else:
                name, action = step, step
            
            # Replace variables
            if variables:
                for key, value in variables.items():
                    action = action.replace(f"{{{key}}}", value)
            
            result = agent.start(action)
            results.append({
                "step": name,
                "action": action,
                "result": str(result)
            })
        
        return {
            "steps": step_list,
            "variables": variables,
            "results": results,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"steps": steps, "error": str(e), "success": False}


# =============================================================================
# CODE TOOLS - Code operations
# =============================================================================

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
        namespace = {"__builtins__": __builtins__}
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
        
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


def run_shell(command: str, cwd: str = None) -> Dict[str, Any]:
    """Execute a shell command.
    
    Args:
        command: Shell command to execute
        cwd: Working directory for the command
    
    Returns:
        Command output
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60
        )
        
        return {
            "command": command,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "Command timed out", "success": False}
    except Exception as e:
        return {"command": command, "error": str(e), "success": False}


def git_commit(message: str = None, all_files: bool = True) -> Dict[str, Any]:
    """Create a git commit with an optional AI-generated message.
    
    Args:
        message: Commit message (if None, AI generates one)
        all_files: Stage all changed files before commit
    
    Returns:
        Commit result
    """
    try:
        if all_files:
            subprocess.run(["git", "add", "-A"], capture_output=True)
        
        # Get diff for AI message generation
        if not message:
            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                capture_output=True,
                text=True
            )
            
            if diff_result.stdout.strip():
                try:
                    from praisonaiagents import Agent
                    agent = Agent(instructions="Generate a concise git commit message based on the diff.")
                    message = str(agent.start(f"Generate commit message for:\n{diff_result.stdout}"))
                except:
                    message = "Update files"
            else:
                return {"error": "No changes to commit", "success": False}
        
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True
        )
        
        return {
            "message": message,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# MCP TOOLS - MCP server operations
# =============================================================================

def mcp_list_servers() -> Dict[str, Any]:
    """List available MCP servers.
    
    Returns:
        List of known MCP servers
    """
    servers = [
        {"name": "filesystem", "command": "npx -y @modelcontextprotocol/server-filesystem"},
        {"name": "memory", "command": "npx -y @modelcontextprotocol/server-memory"},
        {"name": "brave-search", "command": "npx -y @modelcontextprotocol/server-brave-search"},
        {"name": "github", "command": "npx -y @modelcontextprotocol/server-github"},
        {"name": "sqlite", "command": "npx -y @modelcontextprotocol/server-sqlite"},
        {"name": "postgres", "command": "npx -y @modelcontextprotocol/server-postgres"},
        {"name": "puppeteer", "command": "npx -y @modelcontextprotocol/server-puppeteer"},
        {"name": "time", "command": "uvx mcp-server-time"},
        {"name": "fetch", "command": "uvx mcp-server-fetch"},
    ]
    
    return {
        "servers": servers,
        "count": len(servers),
        "success": True
    }


def mcp_connect(command: str, env: Dict[str, str] = None) -> Dict[str, Any]:
    """Connect to an MCP server and list its tools.
    
    Args:
        command: MCP server command (e.g., 'npx -y @modelcontextprotocol/server-filesystem .')
        env: Environment variables for the server
    
    Returns:
        Available tools from the MCP server
    """
    try:
        from praisonaiagents.mcp import MCP
        
        mcp = MCP(command, env=env)
        tools = mcp.get_tools()
        
        tool_info = []
        for tool in tools:
            tool_info.append({
                "name": getattr(tool, '__name__', str(tool)),
                "description": getattr(tool, '__doc__', '')[:100] if hasattr(tool, '__doc__') else ''
            })
        
        return {
            "command": command,
            "tools": tool_info,
            "count": len(tool_info),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"command": command, "error": str(e), "success": False}


# =============================================================================
# SESSION TOOLS - Session management
# =============================================================================

def session_save(name: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Save the current session.
    
    Args:
        name: Session name
        data: Optional data to save with the session
    
    Returns:
        Session save status
    """
    try:
        session_dir = os.path.expanduser("~/.praisonai/sessions")
        os.makedirs(session_dir, exist_ok=True)
        
        session_file = os.path.join(session_dir, f"{name}.json")
        
        session_data = {
            "name": name,
            "timestamp": get_current_time()["datetime"],
            "data": data or {}
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return {
            "name": name,
            "file": session_file,
            "success": True
        }
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


def session_load(name: str) -> Dict[str, Any]:
    """Load a saved session.
    
    Args:
        name: Session name to load
    
    Returns:
        Session data
    """
    try:
        session_dir = os.path.expanduser("~/.praisonai/sessions")
        session_file = os.path.join(session_dir, f"{name}.json")
        
        if not os.path.exists(session_file):
            return {"name": name, "error": "Session not found", "success": False}
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return {
            "name": name,
            "data": session_data,
            "success": True
        }
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


def session_list() -> Dict[str, Any]:
    """List all saved sessions.
    
    Returns:
        List of session names
    """
    try:
        session_dir = os.path.expanduser("~/.praisonai/sessions")
        
        if not os.path.exists(session_dir):
            return {"sessions": [], "count": 0, "success": True}
        
        sessions = []
        for f in os.listdir(session_dir):
            if f.endswith('.json'):
                sessions.append(f[:-5])  # Remove .json extension
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# TOOLS REGISTRY - List of all available tools
# =============================================================================

# Core tools
CORE_TOOLS = [
    search_web,
    calculate,
    get_current_time,
]

# File tools
FILE_TOOLS = [
    read_file,
    write_file,
    list_directory,
]

# Agent tools
AGENT_TOOLS = [
    run_agent,
    run_research,
    generate_agents_yaml,
]

# Memory tools
MEMORY_TOOLS = [
    memory_add,
    memory_search,
    memory_list,
    memory_clear,
]

# Knowledge tools
KNOWLEDGE_TOOLS = [
    knowledge_add,
    knowledge_search,
]

# Todo tools
TODO_TOOLS = [
    todo_add,
    todo_list,
    todo_complete,
]

# Workflow tools
WORKFLOW_TOOLS = [
    workflow_run,
]

# Code tools
CODE_TOOLS = [
    run_python,
    run_shell,
    git_commit,
]

# MCP tools
MCP_TOOLS = [
    mcp_list_servers,
    mcp_connect,
]

# Session tools
SESSION_TOOLS = [
    session_save,
    session_load,
    session_list,
]

# All tools combined
ALL_TOOLS = (
    CORE_TOOLS +
    FILE_TOOLS +
    AGENT_TOOLS +
    MEMORY_TOOLS +
    KNOWLEDGE_TOOLS +
    TODO_TOOLS +
    WORKFLOW_TOOLS +
    CODE_TOOLS +
    MCP_TOOLS +
    SESSION_TOOLS
)


# =============================================================================
# WORKFLOW ADVANCED TOOLS - Pipeline patterns
# =============================================================================

def workflow_create(name: str, steps: List[str]) -> Dict[str, Any]:
    """Create a workflow with multiple steps.
    
    Args:
        name: Workflow name
        steps: List of step descriptions
    
    Returns:
        Created workflow details
    """
    try:
        from praisonaiagents import Workflow, WorkflowStep
        
        workflow_steps = []
        for i, step_desc in enumerate(steps):
            workflow_steps.append(WorkflowStep(
                name=f"step_{i+1}",
                description=step_desc
            ))
        
        workflow = Workflow(name=name, steps=workflow_steps)
        
        return {
            "name": name,
            "steps": steps,
            "step_count": len(steps),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


def workflow_from_yaml(yaml_content: str) -> Dict[str, Any]:
    """Create a workflow from YAML definition.
    
    Args:
        yaml_content: YAML workflow definition
    
    Returns:
        Parsed workflow details
    """
    try:
        from praisonaiagents.workflows import YAMLWorkflowParser
        import yaml
        
        config = yaml.safe_load(yaml_content)
        parser = YAMLWorkflowParser()
        workflow = parser.parse(config)
        
        return {
            "workflow": str(workflow),
            "parsed": True,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# PLANNING TOOLS - Plan creation and execution
# =============================================================================

def plan_create(goal: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Create a plan for achieving a goal.
    
    Args:
        goal: The goal to achieve
        model: LLM model to use for planning
    
    Returns:
        Generated plan with steps
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(
            instructions="""You are a planning assistant. Create a detailed step-by-step plan.
            Format each step as: 1. [Step description]
            Be specific and actionable.""",
            llm=model
        )
        
        result = agent.start(f"Create a detailed plan for: {goal}")
        
        return {
            "goal": goal,
            "plan": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"goal": goal, "error": str(e), "success": False}


def plan_execute(plan: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Execute a plan step by step.
    
    Args:
        plan: The plan to execute (text with numbered steps)
        model: LLM model to use for execution
    
    Returns:
        Execution results for each step
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(
            instructions="Execute each step of the plan and report results.",
            llm=model
        )
        
        result = agent.start(f"Execute this plan:\n{plan}")
        
        return {
            "plan": plan[:200] + "..." if len(plan) > 200 else plan,
            "execution_result": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# GUARDRAIL TOOLS - Output validation
# =============================================================================

def guardrail_validate(content: str, rules: str) -> Dict[str, Any]:
    """Validate content against guardrail rules.
    
    Args:
        content: Content to validate
        rules: Validation rules description
    
    Returns:
        Validation result
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(
            instructions=f"""You are a content validator. Check if the content follows these rules:
            {rules}
            
            Respond with:
            - PASS: if content follows all rules
            - FAIL: if content violates any rules, explain which ones""",
            llm="gpt-4o-mini"
        )
        
        result = agent.start(f"Validate this content:\n{content}")
        result_str = str(result)
        
        return {
            "content": content[:100] + "..." if len(content) > 100 else content,
            "rules": rules,
            "result": result_str,
            "passed": "PASS" in result_str.upper(),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# DEEP RESEARCH TOOLS - Advanced research
# =============================================================================

def deep_research(query: str, max_depth: int = 3, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Perform deep research on a topic with multiple iterations.
    
    Args:
        query: Research query
        max_depth: Maximum research depth/iterations
        model: LLM model to use
    
    Returns:
        Comprehensive research results
    """
    try:
        from praisonaiagents import DeepResearchAgent
        
        agent = DeepResearchAgent(
            model=model,
            max_iterations=max_depth
        )
        
        result = agent.research(query)
        
        return {
            "query": query,
            "research": result.content if hasattr(result, 'content') else str(result),
            "citations": result.citations if hasattr(result, 'citations') else [],
            "success": True
        }
    except ImportError:
        # Fallback to regular research
        return run_research(query, model)
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


# =============================================================================
# CONTEXT TOOLS - Repository analysis
# =============================================================================

def analyze_repository(url: str, goal: str) -> Dict[str, Any]:
    """Analyze a repository for a specific goal.
    
    Args:
        url: Repository URL (GitHub, GitLab, etc.)
        goal: Analysis goal or question
    
    Returns:
        Repository analysis results
    """
    try:
        from praisonaiagents import ContextAgent, create_context_agent
        
        agent = create_context_agent(url=url)
        result = agent.analyze(goal)
        
        return {
            "url": url,
            "goal": goal,
            "analysis": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


def fast_context_search(path: str, query: str) -> Dict[str, Any]:
    """Search codebase for relevant context.
    
    Args:
        path: Path to search in
        query: Search query
    
    Returns:
        Relevant code snippets and files
    """
    try:
        from praisonaiagents import FastContext
        
        fc = FastContext(path)
        results = fc.search(query)
        
        return {
            "path": path,
            "query": query,
            "results": [
                {
                    "file": r.file_path,
                    "lines": f"{r.start_line}-{r.end_line}",
                    "snippet": r.content[:200]
                }
                for r in results[:10]
            ],
            "count": len(results),
            "success": True
        }
    except ImportError:
        return {"error": "FastContext not available"}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


# =============================================================================
# SEARCH TOOLS - Various search providers
# =============================================================================

def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search using Tavily API.
    
    Args:
        query: Search query
        max_results: Maximum results
    
    Returns:
        Search results
    """
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set", "success": False}
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        
        return {
            "query": query,
            "results": response.get("results", []),
            "success": True
        }
    except ImportError:
        return {"error": "tavily not installed. Run: pip install tavily-python"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def duckduckgo_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search using DuckDuckGo.
    
    Args:
        query: Search query
        max_results: Maximum results
    
    Returns:
        Search results
    """
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
                    "snippet": r.get("body", "")
                }
                for r in results
            ],
            "success": True
        }
    except ImportError:
        return {"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


# =============================================================================
# FINANCE TOOLS - Stock and financial data
# =============================================================================

def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get current stock price.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL)
    
    Returns:
        Current stock price and info
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        return {
            "symbol": symbol,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency", "USD"),
            "name": info.get("shortName", symbol),
            "change": info.get("regularMarketChange"),
            "change_percent": info.get("regularMarketChangePercent"),
            "success": True
        }
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "success": False}


def get_stock_history(symbol: str, period: str = "1mo") -> Dict[str, Any]:
    """Get historical stock data.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Historical price data
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        data = []
        for date, row in hist.tail(10).iterrows():
            data.append({
                "date": str(date.date()),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"])
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data": data,
            "success": True
        }
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "success": False}


# =============================================================================
# IMAGE TOOLS - Image analysis
# =============================================================================

def analyze_image(image_path: str, question: str = "Describe this image") -> Dict[str, Any]:
    """Analyze an image using vision model.
    
    Args:
        image_path: Path to image file
        question: Question about the image
    
    Returns:
        Image analysis result
    """
    try:
        from praisonaiagents import ImageAgent
        
        agent = ImageAgent()
        result = agent.analyze(image_path, question)
        
        return {
            "image": image_path,
            "question": question,
            "analysis": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"image": image_path, "error": str(e), "success": False}


# =============================================================================
# QUERY REWRITING TOOLS - Query optimization
# =============================================================================

def rewrite_query(query: str, strategy: str = "auto") -> Dict[str, Any]:
    """Rewrite a query for better search results.
    
    Args:
        query: Original query
        strategy: Rewrite strategy (auto, expand, simplify, technical)
    
    Returns:
        Rewritten query
    """
    try:
        from praisonaiagents import QueryRewriterAgent, RewriteStrategy
        
        strategy_map = {
            "auto": RewriteStrategy.AUTO,
            "expand": RewriteStrategy.EXPAND,
            "simplify": RewriteStrategy.SIMPLIFY,
            "technical": RewriteStrategy.TECHNICAL
        }
        
        agent = QueryRewriterAgent()
        result = agent.rewrite(query, strategy=strategy_map.get(strategy, RewriteStrategy.AUTO))
        
        return {
            "original": query,
            "rewritten": result.primary_query,
            "alternatives": result.alternative_queries if hasattr(result, 'alternative_queries') else [],
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def expand_prompt(prompt: str) -> Dict[str, Any]:
    """Expand a short prompt into a detailed one.
    
    Args:
        prompt: Short prompt to expand
    
    Returns:
        Expanded detailed prompt
    """
    try:
        from praisonaiagents import PromptExpanderAgent, ExpandStrategy
        
        agent = PromptExpanderAgent()
        result = agent.expand(prompt, strategy=ExpandStrategy.AUTO)
        
        return {
            "original": prompt,
            "expanded": result.expanded_prompt,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"prompt": prompt, "error": str(e), "success": False}


# =============================================================================
# RULES TOOLS - Rules management
# =============================================================================

def rules_list() -> Dict[str, Any]:
    """List all defined rules.
    
    Returns:
        List of rules
    """
    try:
        rules_dir = os.path.expanduser("~/.praisonai/rules")
        
        if not os.path.exists(rules_dir):
            return {"rules": [], "count": 0, "success": True}
        
        rules = []
        for f in os.listdir(rules_dir):
            if f.endswith('.txt') or f.endswith('.md'):
                rules.append(f[:-4] if f.endswith('.txt') else f[:-3])
        
        return {"rules": rules, "count": len(rules), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def rules_add(name: str, content: str) -> Dict[str, Any]:
    """Add a new rule.
    
    Args:
        name: Rule name
        content: Rule content
    
    Returns:
        Status of rule creation
    """
    try:
        rules_dir = os.path.expanduser("~/.praisonai/rules")
        os.makedirs(rules_dir, exist_ok=True)
        
        rule_file = os.path.join(rules_dir, f"{name}.txt")
        with open(rule_file, 'w') as f:
            f.write(content)
        
        return {"name": name, "file": rule_file, "success": True}
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


def rules_get(name: str) -> Dict[str, Any]:
    """Get a specific rule.
    
    Args:
        name: Rule name
    
    Returns:
        Rule content
    """
    try:
        rules_dir = os.path.expanduser("~/.praisonai/rules")
        
        for ext in ['.txt', '.md']:
            rule_file = os.path.join(rules_dir, f"{name}{ext}")
            if os.path.exists(rule_file):
                with open(rule_file, 'r') as f:
                    content = f.read()
                return {"name": name, "content": content, "success": True}
        
        return {"name": name, "error": "Rule not found", "success": False}
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


# =============================================================================
# HOOKS TOOLS - Event hooks
# =============================================================================

def hooks_list() -> Dict[str, Any]:
    """List available hooks.
    
    Returns:
        List of hook types
    """
    hooks = [
        {"name": "on_start", "description": "Called when agent starts"},
        {"name": "on_end", "description": "Called when agent completes"},
        {"name": "on_tool_call", "description": "Called before tool execution"},
        {"name": "on_tool_result", "description": "Called after tool execution"},
        {"name": "on_error", "description": "Called on error"},
        {"name": "on_message", "description": "Called on each message"},
    ]
    
    return {"hooks": hooks, "count": len(hooks), "success": True}


# =============================================================================
# DOCS TOOLS - Documentation
# =============================================================================

def docs_search(query: str) -> Dict[str, Any]:
    """Search PraisonAI documentation.
    
    Args:
        query: Search query
    
    Returns:
        Relevant documentation
    """
    docs_url = "https://docs.praison.ai"
    
    # Common documentation topics
    topics = {
        "agent": f"{docs_url}/agents",
        "mcp": f"{docs_url}/mcp",
        "workflow": f"{docs_url}/workflows",
        "memory": f"{docs_url}/memory",
        "tools": f"{docs_url}/tools",
        "knowledge": f"{docs_url}/knowledge",
        "research": f"{docs_url}/research",
        "planning": f"{docs_url}/planning",
    }
    
    query_lower = query.lower()
    relevant = []
    for topic, url in topics.items():
        if topic in query_lower:
            relevant.append({"topic": topic, "url": url})
    
    if not relevant:
        relevant = [{"topic": "main", "url": docs_url}]
    
    return {
        "query": query,
        "results": relevant,
        "docs_url": docs_url,
        "success": True
    }


# =============================================================================
# UPDATED TOOLS REGISTRY
# =============================================================================

# Workflow advanced tools
WORKFLOW_ADVANCED_TOOLS = [
    workflow_create,
    workflow_from_yaml,
]

# Planning tools
PLANNING_TOOLS = [
    plan_create,
    plan_execute,
]

# Guardrail tools
GUARDRAIL_TOOLS = [
    guardrail_validate,
]

# Research advanced tools
RESEARCH_TOOLS = [
    deep_research,
]

# Context tools
CONTEXT_TOOLS = [
    analyze_repository,
    fast_context_search,
]

# Search provider tools
SEARCH_PROVIDER_TOOLS = [
    tavily_search,
    duckduckgo_search,
]

# Finance tools
FINANCE_TOOLS = [
    get_stock_price,
    get_stock_history,
]

# Image tools
IMAGE_TOOLS = [
    analyze_image,
]

# Query tools
QUERY_TOOLS = [
    rewrite_query,
    expand_prompt,
]

# Rules tools
RULES_TOOLS = [
    rules_list,
    rules_add,
    rules_get,
]

# Hooks tools
HOOKS_TOOLS = [
    hooks_list,
]

# Docs tools
DOCS_TOOLS = [
    docs_search,
]

# Update ALL_TOOLS to include new tools
ALL_TOOLS = (
    CORE_TOOLS +
    FILE_TOOLS +
    AGENT_TOOLS +
    MEMORY_TOOLS +
    KNOWLEDGE_TOOLS +
    TODO_TOOLS +
    WORKFLOW_TOOLS +
    CODE_TOOLS +
    MCP_TOOLS +
    SESSION_TOOLS +
    WORKFLOW_ADVANCED_TOOLS +
    PLANNING_TOOLS +
    GUARDRAIL_TOOLS +
    RESEARCH_TOOLS +
    CONTEXT_TOOLS +
    SEARCH_PROVIDER_TOOLS +
    FINANCE_TOOLS +
    IMAGE_TOOLS +
    QUERY_TOOLS +
    RULES_TOOLS +
    HOOKS_TOOLS +
    DOCS_TOOLS
)
