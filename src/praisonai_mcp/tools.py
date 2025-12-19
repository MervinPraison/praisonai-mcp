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
