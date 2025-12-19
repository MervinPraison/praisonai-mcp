# PraisonAI MCP Server

<p align="center">
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://static.pepy.tech/badge/praisonaiagents" alt="Downloads" /></a>
  <a href="https://pypi.org/project/praisonaiagents/"><img src="https://img.shields.io/pypi/v/praisonaiagents" alt="PyPI" /></a>
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai"><img src="https://img.shields.io/badge/MCP-Registry-blue" alt="MCP Registry" /></a>
</p>

An MCP server that exposes [PraisonAI](https://github.com/MervinPraison/PraisonAI) tools and agents as MCP tools for use with Claude Desktop, Cursor, and other MCP clients.

## Features

- 🛠️ **47 Built-in Tools** - Agents, memory, knowledge, workflows, planning, and more
- 🤖 **AI Agents as Tools** - Run PraisonAI agents directly from MCP
- 🔌 **Multiple Transports** - stdio (Claude Desktop) and SSE (web clients)
- ⚡ **Easy Setup** - Works with `uvx` or `pip install`

## Installation

### Using uvx (Recommended)

```bash
uvx praisonai-mcp
```

### Using pip

```bash
pip install praisonai-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"]
    }
  }
}
```

## Available Tools (47 Total)

### 🔧 Core Tools (3)
| Tool | Description |
|------|-------------|
| `search_web` | Search the web using Tavily or DuckDuckGo |
| `calculate` | Evaluate mathematical expressions safely |
| `get_current_time` | Get current date/time in any timezone |

### 📁 File Tools (3)
| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file |
| `write_file` | Write content to a file |
| `list_directory` | List files in a directory |

### 🤖 Agent Tools (3)
| Tool | Description |
|------|-------------|
| `run_agent` | Run a PraisonAI agent with a prompt |
| `run_research` | Deep research on any topic |
| `generate_agents_yaml` | Generate agents.yaml for a topic |

### 🧠 Memory Tools (4)
| Tool | Description |
|------|-------------|
| `memory_add` | Add content to memory store |
| `memory_search` | Search memories |
| `memory_list` | List all memories |
| `memory_clear` | Clear all memories |

### 📚 Knowledge Tools (2)
| Tool | Description |
|------|-------------|
| `knowledge_add` | Add to knowledge base |
| `knowledge_search` | Search knowledge base |

### ✅ Todo Tools (3)
| Tool | Description |
|------|-------------|
| `todo_add` | Add a task to todo list |
| `todo_list` | List all tasks |
| `todo_complete` | Mark task as completed |

### 🔄 Workflow Tools (3)
| Tool | Description |
|------|-------------|
| `workflow_run` | Run multi-step workflows |
| `workflow_create` | Create a workflow with steps |
| `workflow_from_yaml` | Create workflow from YAML |

### 💻 Code Tools (3)
| Tool | Description |
|------|-------------|
| `run_python` | Execute Python code |
| `run_shell` | Execute shell commands |
| `git_commit` | Create git commits (AI-generated messages) |

### 🔌 MCP Tools (2)
| Tool | Description |
|------|-------------|
| `mcp_list_servers` | List available MCP servers |
| `mcp_connect` | Connect to an MCP server |

### 💾 Session Tools (3)
| Tool | Description |
|------|-------------|
| `session_save` | Save current session |
| `session_load` | Load a saved session |
| `session_list` | List all sessions |

### 📋 Planning Tools (2)
| Tool | Description |
|------|-------------|
| `plan_create` | Create a plan for a goal |
| `plan_execute` | Execute a plan step by step |

### 🛡️ Guardrail Tools (1)
| Tool | Description |
|------|-------------|
| `guardrail_validate` | Validate content against rules |

### 🔬 Research Tools (1)
| Tool | Description |
|------|-------------|
| `deep_research` | Deep research with multiple iterations |

### 🔍 Context Tools (2)
| Tool | Description |
|------|-------------|
| `analyze_repository` | Analyze a repository for a goal |
| `fast_context_search` | Search codebase for context |

### 🌐 Search Provider Tools (2)
| Tool | Description |
|------|-------------|
| `tavily_search` | Search using Tavily API |
| `duckduckgo_search` | Search using DuckDuckGo |

### 📈 Finance Tools (2)
| Tool | Description |
|------|-------------|
| `get_stock_price` | Get current stock price |
| `get_stock_history` | Get historical stock data |

### 🖼️ Image Tools (1)
| Tool | Description |
|------|-------------|
| `analyze_image` | Analyze image using vision model |

### ✍️ Query Tools (2)
| Tool | Description |
|------|-------------|
| `rewrite_query` | Rewrite query for better results |
| `expand_prompt` | Expand short prompt to detailed |

### 📜 Rules Tools (3)
| Tool | Description |
|------|-------------|
| `rules_list` | List all defined rules |
| `rules_add` | Add a new rule |
| `rules_get` | Get a specific rule |

### 🪝 Hooks Tools (1)
| Tool | Description |
|------|-------------|
| `hooks_list` | List available hooks |

### 📖 Docs Tools (1)
| Tool | Description |
|------|-------------|
| `docs_search` | Search PraisonAI documentation |

## Running as SSE Server

For web clients or remote access:

```bash
python -m praisonai_mcp --sse --port 8080
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for AI-powered tools |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `BRAVE_API_KEY` | Brave Search API key |

## Related Projects

- [PraisonAI](https://github.com/MervinPraison/PraisonAI) - AI Agents Framework
- [PraisonAI Agents](https://pypi.org/project/praisonaiagents/) - Lightweight agents package
- [MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai) - Official MCP Registry listing

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- 📖 [Documentation](https://docs.praison.ai/mcp)
- 🐛 [Issues](https://github.com/MervinPraison/praisonai-mcp/issues)
- 💬 [Discussions](https://github.com/MervinPraison/PraisonAI/discussions)
