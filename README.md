# PraisonAI MCP Server

<p align="center">
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://static.pepy.tech/badge/praisonaiagents" alt="Downloads" /></a>
  <a href="https://pypi.org/project/praisonaiagents/"><img src="https://img.shields.io/pypi/v/praisonaiagents" alt="PyPI" /></a>
  <a href="https://github.com/MervinPraison/PraisonAI"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.MervinPraison/praisonai"><img src="https://img.shields.io/badge/MCP-Registry-blue" alt="MCP Registry" /></a>
</p>

An MCP server that exposes [PraisonAI](https://github.com/MervinPraison/PraisonAI) tools and agents as MCP tools for use with Claude Desktop, Cursor, and other MCP clients.

## Features

- 🛠️ **100+ Built-in Tools** - Web search, file operations, calculations, and more
- 🤖 **AI Agents as Tools** - Expose PraisonAI agents as MCP tools
- 🔌 **Multiple Transports** - stdio (Claude Desktop) and SSE (web clients)
- ⚡ **Easy Setup** - Works with `uvx` or `pip install`

## Installation

### Using uvx (Recommended)

```bash
uvx praisonai-mcp
```

### Using pip

```bash
pip install praisonaiagents
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

Or with pip installation:

```json
{
  "mcpServers": {
    "praisonai": {
      "command": "python",
      "args": ["-m", "praisonai_mcp"]
    }
  }
}
```

## Available Tools

### Built-in Tools

| Tool | Description |
|------|-------------|
| `search_web` | Search the web using Tavily, DuckDuckGo, or other providers |
| `calculate` | Evaluate mathematical expressions |
| `get_weather` | Get weather information for a city |
| `read_file` | Read contents of a file |
| `write_file` | Write content to a file |
| `list_files` | List files in a directory |

### Custom Tools

Create your own tools by defining Python functions:

```python
from praisonaiagents.mcp import ToolsMCPServer

def my_custom_tool(query: str) -> str:
    """Search for information.
    
    Args:
        query: The search query
    
    Returns:
        Search results
    """
    return f"Results for: {query}"

server = ToolsMCPServer(name="my-tools")
server.register_tool(my_custom_tool)
server.run()
```

## Running as SSE Server

For web clients or remote access:

```bash
python -m praisonai_mcp --sse --port 8080
```

Then connect from any MCP client:

```python
from praisonaiagents import Agent, MCP

agent = Agent(
    tools=MCP("http://localhost:8080/sse")
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for AI-powered tools |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `BRAVE_API_KEY` | Brave Search API key |

## Development

```bash
# Clone the repository
git clone https://github.com/MervinPraison/praisonai-mcp.git
cd praisonai-mcp

# Install dependencies
pip install -e .

# Run the server
python -m praisonai_mcp
```

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
