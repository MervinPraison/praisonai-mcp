# PraisonAI MCP Server

<p align="center">
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://static.pepy.tech/badge/praisonaiagents" alt="Downloads" /></a>
  <a href="https://pypi.org/project/praisonaiagents/"><img src="https://img.shields.io/pypi/v/praisonaiagents" alt="PyPI" /></a>
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai"><img src="https://img.shields.io/badge/MCP-Registry-blue" alt="MCP Registry" /></a>
</p>

An MCP server that exposes [PraisonAI](https://github.com/MervinPraison/PraisonAI) tools and agents as MCP tools for use with Claude Desktop, Cursor, and other MCP clients.

## Features

- 🛠️ **70 Built-in Tools** - Complete coverage of all PraisonAI features
- 🤖 **AI Agents as Tools** - Run PraisonAI agents directly from MCP
- 🔌 **Multiple Transports** - stdio (Claude Desktop) and SSE (web clients)
- ⚡ **Easy Setup** - Works with `uvx` or `pip install`

## Installation

```bash
# Using uvx (Recommended)
uvx praisonai-mcp

# Using pip
pip install praisonai-mcp
```

## Usage with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

## Available Tools (70 Total)

### 🔧 Core Tools
| Tool | Description |
|------|-------------|
| `search_web` | Search the web using Tavily or DuckDuckGo |
| `calculate` | Evaluate mathematical expressions |
| `get_current_time` | Get current date/time in any timezone |

### 📁 File Tools
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `list_directory` | List directory contents |

### 🤖 Agent Tools
| Tool | Description |
|------|-------------|
| `run_agent` | Run a PraisonAI agent |
| `run_research` | Deep research on any topic |
| `generate_agents_yaml` | Generate agents.yaml |
| `run_auto_agents` | Run auto-generated agents |
| `run_handoff` | Run with agent handoff |

### 🧠 Memory Tools
| Tool | Description |
|------|-------------|
| `memory_add` | Add to memory store |
| `memory_search` | Search memories |
| `memory_list` | List all memories |
| `memory_clear` | Clear memories |
| `auto_extract_memories` | Auto-extract memories from text |

### 📚 Knowledge Tools
| Tool | Description |
|------|-------------|
| `knowledge_add` | Add to knowledge base |
| `knowledge_search` | Search knowledge base |

### ✅ Todo Tools
| Tool | Description |
|------|-------------|
| `todo_add` | Add task |
| `todo_list` | List tasks |
| `todo_complete` | Complete task |

### 🔄 Workflow Tools
| Tool | Description |
|------|-------------|
| `workflow_run` | Run workflow |
| `workflow_create` | Create workflow |
| `workflow_from_yaml` | Create from YAML |
| `export_to_n8n` | Export to n8n format |

### 💻 Code Tools
| Tool | Description |
|------|-------------|
| `run_python` | Execute Python code |
| `run_shell` | Execute shell commands |
| `git_commit` | Create git commits |
| `code_apply_diff` | Apply SEARCH/REPLACE diff |
| `code_search_replace` | Search and replace in file |

### 📊 Data Format Tools
| Tool | Description |
|------|-------------|
| `read_csv` | Read CSV file |
| `write_csv` | Write CSV file |
| `read_json_file` | Read JSON file |
| `write_json_file` | Write JSON file |
| `read_yaml_file` | Read YAML file |
| `write_yaml_file` | Write YAML file |

### 🌐 Search Tools
| Tool | Description |
|------|-------------|
| `tavily_search` | Search using Tavily |
| `duckduckgo_search` | Search using DuckDuckGo |
| `wikipedia_search` | Search Wikipedia |
| `arxiv_search` | Search arXiv papers |
| `web_crawl` | Crawl and extract web content |

### 📈 Finance Tools
| Tool | Description |
|------|-------------|
| `get_stock_price` | Get current stock price |
| `get_stock_history` | Get historical stock data |

### 🧮 Calculator Tools
| Tool | Description |
|------|-------------|
| `solve_equation` | Solve mathematical equations |
| `convert_units` | Convert between units |
| `calculate_statistics` | Calculate statistics |

### 💾 Session Tools
| Tool | Description |
|------|-------------|
| `session_save` | Save session |
| `session_load` | Load session |
| `session_list` | List sessions |

### 📋 Planning Tools
| Tool | Description |
|------|-------------|
| `plan_create` | Create a plan |
| `plan_execute` | Execute a plan |

### 🛡️ Guardrail Tools
| Tool | Description |
|------|-------------|
| `guardrail_validate` | Validate content |

### 🔬 Research Tools
| Tool | Description |
|------|-------------|
| `deep_research` | Deep research with iterations |

### 🔍 Context Tools
| Tool | Description |
|------|-------------|
| `analyze_repository` | Analyze repository |
| `fast_context_search` | Search codebase |

### 🖼️ Image Tools
| Tool | Description |
|------|-------------|
| `analyze_image` | Analyze image |

### ✍️ Query Tools
| Tool | Description |
|------|-------------|
| `rewrite_query` | Rewrite query |
| `expand_prompt` | Expand prompt |

### 📜 Rules Tools
| Tool | Description |
|------|-------------|
| `rules_list` | List rules |
| `rules_add` | Add rule |
| `rules_get` | Get rule |

### 🔌 MCP Tools
| Tool | Description |
|------|-------------|
| `mcp_list_servers` | List MCP servers |
| `mcp_connect` | Connect to MCP server |

### 🖥️ System Tools
| Tool | Description |
|------|-------------|
| `list_processes` | List running processes |
| `get_system_info` | Get system information |

### 📊 Telemetry Tools
| Tool | Description |
|------|-------------|
| `track_metrics` | Track metrics event |
| `get_metrics` | Get tracked metrics |

### 🎯 Router Tools
| Tool | Description |
|------|-------------|
| `select_model` | Select best model for task |

### 🪝 Hooks & Docs
| Tool | Description |
|------|-------------|
| `hooks_list` | List available hooks |
| `docs_search` | Search documentation |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `TAVILY_API_KEY` | Tavily search API key |
| `BRAVE_API_KEY` | Brave Search API key |

## Related Projects

- [PraisonAI](https://github.com/MervinPraison/PraisonAI) - AI Agents Framework
- [MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai)

## License

MIT License
