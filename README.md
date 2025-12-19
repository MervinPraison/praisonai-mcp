# PraisonAI MCP Server

<p align="center">
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://static.pepy.tech/badge/praisonaiagents" alt="Downloads" /></a>
  <a href="https://pypi.org/project/praisonaiagents/"><img src="https://img.shields.io/pypi/v/praisonaiagents" alt="PyPI" /></a>
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai"><img src="https://img.shields.io/badge/MCP-Registry-blue" alt="MCP Registry" /></a>
</p>

An MCP server that exposes [PraisonAI](https://github.com/MervinPraison/PraisonAI) tools and agents as MCP tools for use with Claude Desktop, Cursor, VS Code, Windsurf, and other MCP clients.

## Features

- 🛠️ **70 Built-in Tools** - Complete coverage of all PraisonAI features
- 🤖 **AI Agents as Tools** - Run PraisonAI agents directly from MCP
- 🔌 **Multiple Transports** - stdio and SSE support
- ⚡ **Easy Setup** - Works with `uvx` or `pip install`

## Installation

```bash
# Using uvx (Recommended)
uvx praisonai-mcp

# Using pip
pip install praisonai-mcp
```

---

## MCP Client Configurations

### Claude Desktop

**Config file location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

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

---

### VS Code (GitHub Copilot)

**Config file location:** `.vscode/mcp.json` in your workspace or user settings

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "openai-key",
      "description": "OpenAI API Key",
      "password": true
    },
    {
      "type": "promptString",
      "id": "tavily-key",
      "description": "Tavily API Key",
      "password": true
    }
  ],
  "servers": {
    "praisonai": {
      "command": "uvx",
      "args": ["praisonai-mcp"],
      "env": {
        "OPENAI_API_KEY": "${input:openai-key}",
        "TAVILY_API_KEY": "${input:tavily-key}"
      }
    }
  }
}
```

> **Note:** VS Code securely prompts for API keys on first use and stores them for subsequent sessions.

---

### Cursor

**Config file location:** `~/.cursor/mcp.json`

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

---

### Windsurf

**Config file location:** `~/.codeium/windsurf/mcp_config.json`

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

> **Tip:** You can also add MCP servers via Windsurf Settings > Cascade > Plugins.

---

### Cline (VS Code Extension)

**Config file location:** Managed via Cline settings in VS Code

1. Open VS Code Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Search for "Cline: MCP Servers"
3. Add configuration:

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

---

### Continue (VS Code/JetBrains)

**Config file location:** `~/.continue/config.json`

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "uvx",
          "args": ["praisonai-mcp"]
        },
        "env": {
          "OPENAI_API_KEY": "your-openai-api-key",
          "TAVILY_API_KEY": "your-tavily-api-key"
        }
      }
    ]
  }
}
```

---

### Zed

**Config file location:** `~/.config/zed/settings.json`

```json
{
  "context_servers": {
    "praisonai": {
      "command": {
        "path": "uvx",
        "args": ["praisonai-mcp"]
      },
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "TAVILY_API_KEY": "your-tavily-api-key"
      }
    }
  }
}
```

---

### Claude Code (CLI)

```bash
# Add the MCP server
claude mcp add praisonai -- uvx praisonai-mcp

# With environment variables
OPENAI_API_KEY=your-key claude mcp add praisonai -- uvx praisonai-mcp
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI-powered tools | For agent tools |
| `TAVILY_API_KEY` | Tavily API key for web search | For tavily_search |
| `BRAVE_API_KEY` | Brave Search API key | Optional |

---

## Available Tools (70 Total)

### 🔧 Core Tools
`search_web`, `calculate`, `get_current_time`

### 📁 File Tools
`read_file`, `write_file`, `list_directory`

### 🤖 Agent Tools
`run_agent`, `run_research`, `generate_agents_yaml`, `run_auto_agents`, `run_handoff`

### 🧠 Memory Tools
`memory_add`, `memory_search`, `memory_list`, `memory_clear`, `auto_extract_memories`

### 📚 Knowledge Tools
`knowledge_add`, `knowledge_search`

### ✅ Todo Tools
`todo_add`, `todo_list`, `todo_complete`

### 🔄 Workflow Tools
`workflow_run`, `workflow_create`, `workflow_from_yaml`, `export_to_n8n`

### 💻 Code Tools
`run_python`, `run_shell`, `git_commit`, `code_apply_diff`, `code_search_replace`

### 📊 Data Format Tools
`read_csv`, `write_csv`, `read_json_file`, `write_json_file`, `read_yaml_file`, `write_yaml_file`

### 🌐 Search Tools
`tavily_search`, `duckduckgo_search`, `wikipedia_search`, `arxiv_search`, `web_crawl`

### 📈 Finance Tools
`get_stock_price`, `get_stock_history`

### 🧮 Calculator Tools
`solve_equation`, `convert_units`, `calculate_statistics`

### 💾 Session Tools
`session_save`, `session_load`, `session_list`

### 📋 Planning Tools
`plan_create`, `plan_execute`

### 🛡️ Guardrail Tools
`guardrail_validate`

### 🔬 Research Tools
`deep_research`

### 🔍 Context Tools
`analyze_repository`, `fast_context_search`

### 🖼️ Image Tools
`analyze_image`

### ✍️ Query Tools
`rewrite_query`, `expand_prompt`

### 📜 Rules Tools
`rules_list`, `rules_add`, `rules_get`

### 🔌 MCP Tools
`mcp_list_servers`, `mcp_connect`

### 🖥️ System Tools
`list_processes`, `get_system_info`

### 📊 Telemetry Tools
`track_metrics`, `get_metrics`

### 🎯 Router Tools
`select_model`

### 🪝 Hooks & Docs
`hooks_list`, `docs_search`

---

## Running as SSE Server

For web clients or remote access:

```bash
python -m praisonai_mcp --sse --port 8080
```

---

## Related Projects

- [PraisonAI](https://github.com/MervinPraison/PraisonAI) - AI Agents Framework
- [PraisonAI Agents](https://pypi.org/project/praisonaiagents/) - Lightweight agents package
- [MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai) - Official MCP Registry listing

## License

MIT License

## Links

- 📖 [Documentation](https://docs.praison.ai/mcp)
- 🐛 [Issues](https://github.com/MervinPraison/praisonai-mcp/issues)
- 💬 [Discussions](https://github.com/MervinPraison/PraisonAI/discussions)
