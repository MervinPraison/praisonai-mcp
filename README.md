# PraisonAI MCP Server

<p align="center">
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://static.pepy.tech/badge/praisonaiagents" alt="Downloads" /></a>
  <a href="https://pypi.org/project/praisonaiagents/"><img src="https://img.shields.io/pypi/v/praisonaiagents" alt="PyPI" /></a>
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai"><img src="https://img.shields.io/badge/MCP-Registry-blue" alt="MCP Registry" /></a>
</p>

An MCP server that exposes [PraisonAI](https://github.com/MervinPraison/PraisonAI) AI agents and tools for use with Claude Desktop, Cursor, VS Code, Windsurf, and other MCP clients.

## Features

- 🤖 **AI Agents as Tools** - Run PraisonAI agents directly from MCP
- 🔄 **Workflow Orchestration** - Multi-step agent workflows
- 🛠️ **64 Built-in Tools** - Complete coverage of all PraisonAI features
- ⚡ **Easy Setup** - Works with `uvx` or `pip install`

## Installation

```bash
# Using uvx (Recommended)
uvx praisonai-mcp

# Using pip
pip install praisonai-mcp
```

---

## Available Tools (64 Total)

### 🤖 Agent Tools (Primary)

The core tools for running AI agents:

| Tool | Description |
|------|-------------|
| `run_agent` | Run a PraisonAI agent with a prompt |
| `run_research` | Deep research on any topic |
| `run_auto_agents` | Auto-generate and run agents for a task |
| `run_handoff` | Run task with agent handoff/delegation |
| `generate_agents_yaml` | Generate agents.yaml configuration |

### 🔄 Workflow Tools

Orchestrate multi-step agent workflows:

| Tool | Description |
|------|-------------|
| `workflow_run` | Run a multi-step workflow |
| `workflow_create` | Create a new workflow |
| `workflow_from_yaml` | Create workflow from YAML |
| `export_to_n8n` | Export workflow to n8n format |

---

### 🌐 Search Tools (13 tools)

Unified web search with automatic fallback across multiple providers:

| Tool | Description |
|------|-------------|
| `search_web` | **Unified search** - Auto-fallback across providers |
| `get_search_providers` | List available providers and their status |
| **Tavily** | |
| `tavily_search` | AI-powered search (requires `TAVILY_API_KEY`) |
| `tavily_extract` | Extract content from URLs |
| **Exa** | |
| `exa_search` | Semantic search (requires `EXA_API_KEY`) |
| `exa_search_contents` | Search with full content retrieval |
| `exa_find_similar` | Find similar pages to a URL |
| **You.com** | |
| `ydc_search` | AI search with LLM-ready snippets (requires `YDC_API_KEY`) |
| `ydc_news` | Live news search |
| **Free Providers** | |
| `duckduckgo_search` | DuckDuckGo search (no API key) |
| `wikipedia_search` | Wikipedia search |
| `arxiv_search` | arXiv academic papers |
| `searxng_search` | Self-hosted SearxNG meta search |

### 🕷️ Crawl & Scrape Tools

Web crawling and content extraction:

| Tool | Description |
|------|-------------|
| `crawl4ai_scrape` | Scrape webpage using Crawl4AI |
| `crawl4ai_extract` | Extract structured data with Crawl4AI |
| `scrape_page` | Scrape webpage and extract text |
| `extract_links` | Extract all links from a webpage |
| `web_crawl` | Crawl website and extract content |

---

### 📦 Supporting Tools

#### 🧠 Memory & Knowledge
| Tool | Description |
|------|-------------|
| `memory_add` | Add to memory store |
| `memory_search` | Search memories |
| `memory_list` | List all memories |
| `memory_clear` | Clear memories |
| `auto_extract_memories` | Auto-extract memories from text |
| `knowledge_add` | Add to knowledge base |
| `knowledge_search` | Search knowledge base |

#### 📋 Planning & Research
| Tool | Description |
|------|-------------|
| `plan_create` | Create a plan for a goal |
| `plan_execute` | Execute a plan step by step |
| `deep_research` | Deep research with iterations |
| `analyze_repository` | Analyze a repository |
| `fast_context_search` | Search codebase for context |

#### 💻 Code & Execution
| Tool | Description |
|------|-------------|
| `run_python` | Execute Python code |
| `run_shell` | Execute shell commands |
| `git_commit` | Create git commits |
| `code_apply_diff` | Apply SEARCH/REPLACE diff |
| `code_search_replace` | Search and replace in file |

#### 📁 File Operations
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `list_directory` | List directory contents |
| `read_csv` | Read CSV file |
| `write_csv` | Write CSV file |
| `read_json_file` | Read JSON file |
| `write_json_file` | Write JSON file |
| `read_yaml_file` | Read YAML file |
| `write_yaml_file` | Write YAML file |

#### 🧮 Utilities
| Tool | Description |
|------|-------------|
| `calculate` | Evaluate math expressions |
| `get_current_time` | Get current date/time |
| `solve_equation` | Solve math equations |
| `convert_units` | Convert between units |
| `calculate_statistics` | Calculate statistics |

#### 📈 Finance
| Tool | Description |
|------|-------------|
| `get_stock_price` | Get current stock price |
| `get_stock_history` | Get historical stock data |

#### 🖼️ Image & Query
| Tool | Description |
|------|-------------|
| `analyze_image` | Analyze image using vision |
| `rewrite_query` | Rewrite query for better results |
| `expand_prompt` | Expand short prompt to detailed |

#### ✅ Task Management
| Tool | Description |
|------|-------------|
| `todo_add` | Add task to todo list |
| `todo_list` | List all tasks |
| `todo_complete` | Mark task as completed |

#### 💾 Session & State
| Tool | Description |
|------|-------------|
| `session_save` | Save current session |
| `session_load` | Load a saved session |
| `session_list` | List all sessions |

#### 📜 Rules & Guardrails
| Tool | Description |
|------|-------------|
| `rules_list` | List all defined rules |
| `rules_add` | Add a new rule |
| `rules_get` | Get a specific rule |
| `guardrail_validate` | Validate content against rules |

#### 🖥️ System & Telemetry
| Tool | Description |
|------|-------------|
| `list_processes` | List running processes |
| `get_system_info` | Get system information |
| `track_metrics` | Track metrics event |
| `get_metrics` | Get tracked metrics |
| `select_model` | Select best model for task |

#### 🔌 MCP & Hooks
| Tool | Description |
|------|-------------|
| `mcp_list_servers` | List MCP servers |
| `mcp_connect` | Connect to MCP server |
| `hooks_list` | List available hooks |
| `docs_search` | Search documentation |

---

## MCP Client Configurations

### Claude Desktop

**Config file:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "TAVILY_API_KEY": "your-tavily-api-key"
      }
    }
  }
}
```

### VS Code (GitHub Copilot)

**Config file:** `.vscode/mcp.json`

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "openai-key",
      "description": "OpenAI API Key",
      "password": true
    }
  ],
  "servers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "${input:openai-key}"
      }
    }
  }
}
```

### Cursor

**Config file:** `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

### Windsurf

**Config file:** `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

### Cline (VS Code Extension)

Open Command Palette → "Cline: MCP Servers" → Add:

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

### Continue

**Config file:** `~/.continue/config.json`

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "uvx",
          "args": ["praisonai-mcp"]
        }
      }
    ]
  }
}
```

### Zed

**Config file:** `~/.config/zed/settings.json`

```json
{
  "context_servers": {
    "praisonai": {
      "command": {
        "path": "uvx",
        "args": ["praisonai-mcp"]
      }
    }
  }
}
```

### Claude Code (CLI)

```bash
claude mcp add praisonai -- uvx praisonai-mcp
```

---

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | Agent tools |
| `TAVILY_API_KEY` | Tavily search API key | tavily_search, tavily_extract |
| `EXA_API_KEY` | Exa search API key | exa_search, exa_search_contents, exa_find_similar |
| `YDC_API_KEY` | You.com API key | ydc_search, ydc_news |
| `SEARXNG_URL` | SearxNG instance URL | searxng_search (optional) |

---

## Running as SSE Server

```bash
python -m praisonai_mcp --sse --port 8080
```

---

## Links

- 📖 [Documentation](https://docs.praison.ai/mcp)
- 🐙 [PraisonAI](https://github.com/MervinPraison/PraisonAI)
- 📦 [MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai)

## License

MIT License
