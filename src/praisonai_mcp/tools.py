"""PraisonAI MCP Tools - Thin wrappers around praisonaiagents.

This module exposes PraisonAI tools via direct imports from praisonaiagents,
avoiding code duplication and staying in sync with the main package.
"""

from typing import Dict, Any, List
import os
import json

# =============================================================================
# AGENT TOOLS - Primary tools for running AI agents
# =============================================================================

def run_agent(prompt: str, model: str = "gpt-4o-mini", instructions: str = None) -> Dict[str, Any]:
    """Run a PraisonAI agent with a prompt.
    
    Args:
        prompt: The task or question for the agent
        model: LLM model to use (default: gpt-4o-mini)
        instructions: Optional system instructions
    
    Returns:
        Agent response
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(
            instructions=instructions or "You are a helpful AI assistant.",
            llm=model
        )
        result = agent.start(prompt)
        
        return {
            "prompt": prompt,
            "result": str(result),
            "model": model,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed. Run: pip install praisonaiagents"}
    except Exception as e:
        return {"prompt": prompt, "error": str(e), "success": False}


def run_research(topic: str, max_iterations: int = 3) -> Dict[str, Any]:
    """Run deep research on a topic using DeepResearchAgent.
    
    Args:
        topic: Research topic
        max_iterations: Maximum research iterations
    
    Returns:
        Research findings
    """
    try:
        from praisonaiagents import DeepResearchAgent
        
        agent = DeepResearchAgent(max_iterations=max_iterations)
        result = agent.start(topic)
        
        return {
            "topic": topic,
            "result": str(result),
            "iterations": max_iterations,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed"}
    except Exception as e:
        return {"topic": topic, "error": str(e), "success": False}


def run_auto_agents(topic: str, num_agents: int = 3) -> Dict[str, Any]:
    """Auto-generate and run agents for a topic.
    
    Args:
        topic: Topic or task for agents
        num_agents: Number of agents to generate
    
    Returns:
        Result from auto agents
    """
    try:
        from praisonaiagents import AutoAgents
        
        agents = AutoAgents(topic=topic, num_agents=num_agents)
        result = agents.run()
        
        return {
            "topic": topic,
            "num_agents": num_agents,
            "result": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed"}
    except Exception as e:
        return {"topic": topic, "error": str(e), "success": False}


def run_handoff(task: str, agents: List[str]) -> Dict[str, Any]:
    """Run task with agent handoff/delegation.
    
    Args:
        task: Task to complete
        agents: List of agent roles for handoff
    
    Returns:
        Result from handoff execution
    """
    try:
        from praisonaiagents import Agent
        
        agent_list = []
        for role in agents:
            agent_list.append(Agent(
                name=role,
                instructions=f"You are a {role}. Complete tasks related to your expertise."
            ))
        
        primary = agent_list[0]
        if len(agent_list) > 1:
            primary.handoffs = agent_list[1:]
        
        result = primary.start(task)
        
        return {
            "task": task,
            "agents": agents,
            "result": str(result),
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed"}
    except Exception as e:
        return {"task": task, "error": str(e), "success": False}


def generate_agents_yaml(topic: str) -> Dict[str, Any]:
    """Generate agents.yaml configuration for a topic.
    
    Args:
        topic: Topic to generate agents for
    
    Returns:
        Generated YAML configuration
    """
    try:
        from praisonaiagents import AutoAgents
        
        agents = AutoAgents(topic=topic)
        yaml_content = agents.generate_yaml()
        
        return {
            "topic": topic,
            "yaml": yaml_content,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed"}
    except Exception as e:
        return {"topic": topic, "error": str(e), "success": False}


# =============================================================================
# WORKFLOW TOOLS
# =============================================================================

def workflow_run(steps: List[str], name: str = "workflow") -> Dict[str, Any]:
    """Run a multi-step workflow.
    
    Args:
        steps: List of workflow steps
        name: Workflow name
    
    Returns:
        Workflow execution results
    """
    try:
        from praisonaiagents import Agent
        
        agent = Agent(instructions="Execute each step in order.")
        results = []
        
        for i, step in enumerate(steps):
            result = agent.start(step)
            results.append({"step": i + 1, "task": step, "result": str(result)})
        
        return {
            "name": name,
            "steps": len(steps),
            "results": results,
            "success": True
        }
    except ImportError:
        return {"error": "praisonaiagents not installed"}
    except Exception as e:
        return {"name": name, "error": str(e), "success": False}


def workflow_create(name: str, steps: List[str]) -> Dict[str, Any]:
    """Create a workflow definition.
    
    Args:
        name: Workflow name
        steps: List of step descriptions
    
    Returns:
        Workflow definition
    """
    return {
        "name": name,
        "steps": [{"id": i + 1, "task": step} for i, step in enumerate(steps)],
        "created": True,
        "success": True
    }


def workflow_from_yaml(yaml_content: str) -> Dict[str, Any]:
    """Create workflow from YAML configuration.
    
    Args:
        yaml_content: YAML workflow definition
    
    Returns:
        Parsed workflow
    """
    try:
        import yaml
        workflow = yaml.safe_load(yaml_content)
        return {"workflow": workflow, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def export_to_n8n(workflow_name: str, steps: List[str]) -> Dict[str, Any]:
    """Export workflow to n8n format.
    
    Args:
        workflow_name: Name of the workflow
        steps: List of workflow steps
    
    Returns:
        n8n workflow JSON
    """
    nodes = [{"id": "start", "name": "Start", "type": "n8n-nodes-base.start", "position": [250, 300]}]
    connections = {}
    
    prev_node = "start"
    for i, step in enumerate(steps):
        node_id = f"step_{i+1}"
        nodes.append({
            "id": node_id,
            "name": step[:30],
            "type": "n8n-nodes-base.httpRequest",
            "position": [250 + (i+1) * 200, 300],
        })
        if prev_node not in connections:
            connections[prev_node] = {"main": [[]]}
        connections[prev_node]["main"][0].append({"node": node_id, "type": "main", "index": 0})
        prev_node = node_id
    
    return {
        "workflow_name": workflow_name,
        "n8n_workflow": {"name": workflow_name, "nodes": nodes, "connections": connections},
        "node_count": len(nodes),
        "success": True
    }


# =============================================================================
# SEARCH TOOLS - Unified web search with multiple providers
# =============================================================================

def search_web(query: str, max_results: int = 5, providers: List[str] = None) -> Dict[str, Any]:
    """Search the web using multiple providers with automatic fallback.
    
    Tries each search provider in order until one succeeds. Provider priority:
    1. Tavily (requires TAVILY_API_KEY)
    2. Exa (requires EXA_API_KEY)
    3. You.com (requires YDC_API_KEY)
    4. DuckDuckGo (no API key needed)
    5. SearxNG (requires SEARXNG_URL)
    
    Args:
        query: Search query
        max_results: Maximum results (default: 5)
        providers: Optional list of providers to try in order
                  Options: ["tavily", "exa", "youdotcom", "duckduckgo", "searxng"]
    
    Returns:
        Search results with provider info
    """
    try:
        from praisonaiagents.tools.web_search import search_web as _search_web
        results = _search_web(query, max_results=max_results, providers=providers)
        
        # Check if results contain error
        if results and isinstance(results, list) and "error" in results[0]:
            return {"query": query, "error": results[0]["error"], "success": False}
        
        provider = results[0].get("provider", "unknown") if results else "unknown"
        return {
            "query": query,
            "results": results,
            "provider": provider,
            "count": len(results),
            "success": True
        }
    except ImportError:
        # Fallback to DuckDuckGo
        try:
            from duckduckgo_search import DDGS
            ddgs = DDGS()
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "provider": "duckduckgo"
                })
            return {"query": query, "results": results, "provider": "duckduckgo", "success": True}
        except Exception:
            return {"query": query, "error": "No search providers available", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def get_search_providers() -> Dict[str, Any]:
    """Get list of available search providers and their status.
    
    Returns:
        List of providers with availability status
    """
    try:
        from praisonaiagents.tools.web_search import get_available_providers
        providers = get_available_providers()
        return {"providers": providers, "count": len(providers), "success": True}
    except ImportError:
        # Manual check
        providers = []
        
        # Check Tavily
        providers.append({
            "name": "tavily",
            "available": bool(os.environ.get("TAVILY_API_KEY")),
            "reason": None if os.environ.get("TAVILY_API_KEY") else "TAVILY_API_KEY not set"
        })
        
        # Check Exa
        providers.append({
            "name": "exa",
            "available": bool(os.environ.get("EXA_API_KEY")),
            "reason": None if os.environ.get("EXA_API_KEY") else "EXA_API_KEY not set"
        })
        
        # Check You.com
        providers.append({
            "name": "youdotcom",
            "available": bool(os.environ.get("YDC_API_KEY")),
            "reason": None if os.environ.get("YDC_API_KEY") else "YDC_API_KEY not set"
        })
        
        # DuckDuckGo (always available if package installed)
        try:
            from duckduckgo_search import DDGS  # noqa
            providers.append({"name": "duckduckgo", "available": True, "reason": None})
        except ImportError:
            providers.append({"name": "duckduckgo", "available": False, "reason": "duckduckgo_search not installed"})
        
        return {"providers": providers, "count": len(providers), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """Search using Tavily API.
    
    Tavily provides AI-powered web search optimized for LLM applications.
    Requires TAVILY_API_KEY environment variable.
    
    Args:
        query: Search query
        max_results: Maximum results (default: 5)
        search_depth: "basic" or "advanced" (default: basic)
    
    Returns:
        Tavily search results
    """
    try:
        from praisonaiagents.tools.tavily_tools import tavily_search as _tavily
        results = _tavily(query, max_results=max_results, search_depth=search_depth)
        return {"query": query, "results": results, "provider": "tavily", "success": True}
    except ImportError:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set. Get one at https://tavily.com", "success": False}
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=max_results, search_depth=search_depth)
            return {"query": query, "results": response.get("results", []), "provider": "tavily", "success": True}
        except Exception as e:
            return {"error": f"Tavily error: {str(e)}", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def tavily_extract(urls: List[str]) -> Dict[str, Any]:
    """Extract content from URLs using Tavily.
    
    Args:
        urls: List of URLs to extract content from
    
    Returns:
        Extracted content
    """
    try:
        from praisonaiagents.tools.tavily_tools import tavily_extract as _extract
        results = _extract(urls)
        return {"urls": urls, "results": results, "success": True}
    except ImportError:
        return {"error": "tavily-python not installed. Run: pip install tavily-python", "success": False}
    except Exception as e:
        return {"urls": urls, "error": str(e), "success": False}


def exa_search(query: str, num_results: int = 10, search_type: str = "auto") -> Dict[str, Any]:
    """Search using Exa API.
    
    Exa provides AI-powered web search with semantic understanding.
    Requires EXA_API_KEY environment variable.
    
    Args:
        query: Search query
        num_results: Number of results (default: 10)
        search_type: "auto", "neural", "fast", or "deep" (default: auto)
    
    Returns:
        Exa search results
    """
    try:
        from praisonaiagents.tools.exa_tools import exa_search as _exa
        results = _exa(query, num_results=num_results, type=search_type)
        return {"query": query, "results": results, "provider": "exa", "success": True}
    except ImportError:
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return {"error": "EXA_API_KEY not set. Get one at https://exa.ai", "success": False}
        try:
            from exa_py import Exa
            client = Exa(api_key)
            response = client.search(query, num_results=num_results, type=search_type)
            results = [{"url": r.url, "title": getattr(r, "title", "")} for r in response.results]
            return {"query": query, "results": results, "provider": "exa", "success": True}
        except Exception as e:
            return {"error": f"Exa error: {str(e)}", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def exa_search_contents(query: str, num_results: int = 5, text: bool = True) -> Dict[str, Any]:
    """Search Exa and retrieve content from results.
    
    Args:
        query: Search query
        num_results: Number of results (default: 5)
        text: Include full text content (default: True)
    
    Returns:
        Search results with content
    """
    try:
        from praisonaiagents.tools.exa_tools import exa_search_contents as _exa_contents
        results = _exa_contents(query, num_results=num_results, text=text)
        return {"query": query, "results": results, "provider": "exa", "success": True}
    except ImportError:
        return {"error": "exa_py not installed. Run: pip install exa_py", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def exa_find_similar(url: str, num_results: int = 10) -> Dict[str, Any]:
    """Find similar pages to a given URL using Exa.
    
    Args:
        url: URL to find similar pages for
        num_results: Number of results (default: 10)
    
    Returns:
        Similar pages
    """
    try:
        from praisonaiagents.tools.exa_tools import exa_find_similar as _find_similar
        results = _find_similar(url, num_results=num_results)
        return {"url": url, "results": results, "provider": "exa", "success": True}
    except ImportError:
        return {"error": "exa_py not installed. Run: pip install exa_py", "success": False}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


def ydc_search(query: str, count: int = 10) -> Dict[str, Any]:
    """Search using You.com API.
    
    You.com provides AI-powered search with LLM-ready snippets.
    Requires YDC_API_KEY environment variable.
    
    Args:
        query: Search query
        count: Number of results (default: 10)
    
    Returns:
        You.com search results
    """
    try:
        from praisonaiagents.tools.youdotcom_tools import ydc_search as _ydc
        results = _ydc(query, count=count)
        return {"query": query, "results": results, "provider": "youdotcom", "success": True}
    except ImportError:
        api_key = os.environ.get("YDC_API_KEY")
        if not api_key:
            return {"error": "YDC_API_KEY not set. Get one at https://you.com/api", "success": False}
        try:
            from youdotcom import You
            client = You(api_key_auth=api_key)
            response = client.search.unified(query=query, count=count)
            results = []
            if hasattr(response, 'results'):
                web_results = getattr(response.results, 'web', []) or []
                for r in web_results[:count]:
                    results.append({
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "snippet": getattr(r, "description", "")
                    })
            return {"query": query, "results": results, "provider": "youdotcom", "success": True}
        except Exception as e:
            return {"error": f"You.com error: {str(e)}", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def ydc_news(query: str, count: int = 10) -> Dict[str, Any]:
    """Get live news from You.com.
    
    Args:
        query: News search query
        count: Number of results (default: 10)
    
    Returns:
        News results
    """
    try:
        from praisonaiagents.tools.youdotcom_tools import ydc_news as _ydc_news
        results = _ydc_news(query, count=count)
        return {"query": query, "results": results, "provider": "youdotcom", "success": True}
    except ImportError:
        return {"error": "youdotcom not installed. Run: pip install youdotcom", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def duckduckgo_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search using DuckDuckGo (no API key required).
    
    Args:
        query: Search query
        max_results: Maximum results (default: 5)
    
    Returns:
        DuckDuckGo search results
    """
    try:
        from praisonaiagents.tools.duckduckgo_tools import internet_search
        results = internet_search(query, max_results=max_results)
        return {"query": query, "results": results, "provider": "duckduckgo", "success": True}
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            ddgs = DDGS()
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return {"query": query, "results": results, "provider": "duckduckgo", "success": True}
        except ImportError:
            return {"error": "duckduckgo_search not installed. Run: pip install duckduckgo-search", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def wikipedia_search(query: str, limit: int = 3) -> Dict[str, Any]:
    """Search Wikipedia.
    
    Args:
        query: Search query
        limit: Maximum results (default: 3)
    
    Returns:
        Wikipedia results
    """
    try:
        from praisonaiagents.tools.wikipedia_tools import wiki_search
        results = wiki_search(query, limit=limit)
        return {"query": query, "results": results, "provider": "wikipedia", "success": True}
    except ImportError:
        return {"error": "wikipedia package not installed. Run: pip install wikipedia", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def arxiv_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search arXiv for academic papers.
    
    Args:
        query: Search query
        max_results: Maximum results (default: 5)
    
    Returns:
        arXiv paper results
    """
    try:
        from praisonaiagents.tools.arxiv_tools import search_arxiv
        results = search_arxiv(query, max_results=max_results)
        return {"query": query, "results": results, "provider": "arxiv", "success": True}
    except ImportError:
        return {"error": "arxiv package not installed. Run: pip install arxiv", "success": False}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


def searxng_search(query: str, max_results: int = 5, searxng_url: str = None) -> Dict[str, Any]:
    """Search using SearxNG (self-hosted meta search engine).
    
    Args:
        query: Search query
        max_results: Maximum results (default: 5)
        searxng_url: SearxNG instance URL (default: SEARXNG_URL env var or localhost:32768)
    
    Returns:
        SearxNG search results
    """
    try:
        import requests
        url = searxng_url or os.environ.get("SEARXNG_URL", "http://localhost:32768/search")
        
        params = {
            'q': query,
            'format': 'json',
            'engines': 'google,bing,duckduckgo',
            'safesearch': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        raw_results = response.json().get('results', [])
        results = []
        for r in raw_results[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })
        
        return {"query": query, "results": results, "provider": "searxng", "success": True}
    except Exception as e:
        return {"query": query, "error": str(e), "success": False}


# =============================================================================
# CRAWL & SCRAPE TOOLS
# =============================================================================

def scrape_page(url: str) -> Dict[str, Any]:
    """Scrape a webpage and extract text content.
    
    Args:
        url: URL to scrape
    
    Returns:
        Page content
    """
    try:
        from praisonaiagents.tools.spider_tools import scrape_page as _scrape
        result = _scrape(url)
        return {"url": url, "content": result, "success": True}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


def extract_links(url: str) -> Dict[str, Any]:
    """Extract all links from a webpage.
    
    Args:
        url: URL to extract links from
    
    Returns:
        List of links
    """
    try:
        from praisonaiagents.tools.spider_tools import extract_links as _extract
        links = _extract(url)
        return {"url": url, "links": links, "count": len(links) if links else 0, "success": True}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


def web_crawl(url: str, max_pages: int = 5) -> Dict[str, Any]:
    """Crawl a website.
    
    Args:
        url: Starting URL
        max_pages: Maximum pages to crawl
    
    Returns:
        Crawled content
    """
    try:
        from praisonaiagents.tools.spider_tools import crawl as _crawl
        result = _crawl(url, max_pages=max_pages)
        return {"url": url, "result": result, "success": True}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


def crawl4ai_scrape(url: str) -> Dict[str, Any]:
    """Scrape webpage using Crawl4AI.
    
    Args:
        url: URL to scrape
    
    Returns:
        Scraped content
    """
    try:
        from praisonaiagents.tools.crawl4ai_tools import crawl4ai
        result = crawl4ai(url)
        return {"url": url, "content": result, "success": True}
    except ImportError:
        return {"error": "crawl4ai not installed. Run: pip install crawl4ai"}
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


# =============================================================================
# MEMORY TOOLS
# =============================================================================

def memory_add(content: str, user_id: str = "default") -> Dict[str, Any]:
    """Add content to memory store.
    
    Args:
        content: Content to remember
        user_id: User ID for isolation
    
    Returns:
        Memory storage status
    """
    try:
        from praisonaiagents import Memory
        memory = Memory(user_id=user_id)
        memory.add(content)
        return {"content": content[:100], "user_id": user_id, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def memory_search(query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
    """Search memories.
    
    Args:
        query: Search query
        user_id: User ID
        limit: Maximum results
    
    Returns:
        Matching memories
    """
    try:
        from praisonaiagents import Memory
        memory = Memory(user_id=user_id)
        results = memory.search(query, limit=limit)
        return {"query": query, "results": results, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def memory_list(user_id: str = "default") -> Dict[str, Any]:
    """List all memories.
    
    Args:
        user_id: User ID
    
    Returns:
        All memories
    """
    try:
        from praisonaiagents import Memory
        memory = Memory(user_id=user_id)
        memories = memory.get_all()
        return {"memories": memories, "count": len(memories) if memories else 0, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def memory_clear(user_id: str = "default") -> Dict[str, Any]:
    """Clear all memories.
    
    Args:
        user_id: User ID
    
    Returns:
        Clear status
    """
    try:
        from praisonaiagents import Memory
        memory = Memory(user_id=user_id)
        memory.clear()
        return {"user_id": user_id, "cleared": True, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# KNOWLEDGE TOOLS
# =============================================================================

def knowledge_add(content: str, source: str = "manual") -> Dict[str, Any]:
    """Add to knowledge base.
    
    Args:
        content: Content to add
        source: Source identifier
    
    Returns:
        Add status
    """
    try:
        from praisonaiagents import Knowledge
        kb = Knowledge()
        kb.add(content, source=source)
        return {"content": content[:100], "source": source, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def knowledge_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search knowledge base.
    
    Args:
        query: Search query
        limit: Maximum results
    
    Returns:
        Matching knowledge
    """
    try:
        from praisonaiagents import Knowledge
        kb = Knowledge()
        results = kb.search(query, limit=limit)
        return {"query": query, "results": results, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# SESSION TOOLS
# =============================================================================

def session_save(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Save current session.
    
    Args:
        name: Session name
        data: Session data
    
    Returns:
        Save status
    """
    try:
        from praisonaiagents import Session
        session = Session(name=name)
        session.save(data)
        return {"name": name, "success": True}
    except Exception as e:
        # Fallback to file-based
        session_dir = os.path.expanduser("~/.praisonai/sessions")
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, f"{name}.json"), 'w') as f:
            json.dump(data, f)
        return {"name": name, "success": True}


def session_load(name: str) -> Dict[str, Any]:
    """Load a saved session.
    
    Args:
        name: Session name
    
    Returns:
        Session data
    """
    try:
        from praisonaiagents import Session
        session = Session(name=name)
        data = session.load()
        return {"name": name, "data": data, "success": True}
    except Exception as e:
        # Fallback
        session_file = os.path.expanduser(f"~/.praisonai/sessions/{name}.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                return {"name": name, "data": json.load(f), "success": True}
        return {"name": name, "error": "Session not found", "success": False}


def session_list() -> Dict[str, Any]:
    """List all sessions.
    
    Returns:
        List of sessions
    """
    session_dir = os.path.expanduser("~/.praisonai/sessions")
    if not os.path.exists(session_dir):
        return {"sessions": [], "count": 0, "success": True}
    
    sessions = [f.replace('.json', '') for f in os.listdir(session_dir) if f.endswith('.json')]
    return {"sessions": sessions, "count": len(sessions), "success": True}


# =============================================================================
# FILE TOOLS
# =============================================================================

def read_file(path: str) -> Dict[str, Any]:
    """Read file contents.
    
    Args:
        path: File path
    
    Returns:
        File content
    """
    try:
        from praisonaiagents.tools.file_tools import FileTools
        ft = FileTools()
        content = ft.read_file(path)
        return {"path": path, "content": content, "success": True}
    except ImportError:
        try:
            with open(path, 'r') as f:
                content = f.read()
            return {"path": path, "content": content, "success": True}
        except Exception as e:
            return {"path": path, "error": str(e), "success": False}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Write content to file.
    
    Args:
        path: File path
        content: Content to write
    
    Returns:
        Write status
    """
    try:
        from praisonaiagents.tools.file_tools import FileTools
        ft = FileTools()
        ft.write_file(path, content)
        return {"path": path, "success": True}
    except ImportError:
        try:
            with open(path, 'w') as f:
                f.write(content)
            return {"path": path, "success": True}
        except Exception as e:
            return {"path": path, "error": str(e), "success": False}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def list_directory(path: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """List directory contents.
    
    Args:
        path: Directory path
        pattern: Glob pattern
    
    Returns:
        Directory listing
    """
    try:
        from praisonaiagents.tools.file_tools import FileTools
        ft = FileTools()
        items = ft.list_files(path, pattern=pattern)
        return {"path": path, "items": items, "success": True}
    except ImportError:
        import glob
        items = glob.glob(os.path.join(path, pattern))
        return {"path": path, "items": items, "count": len(items), "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def read_csv(path: str, limit: int = 100) -> Dict[str, Any]:
    """Read CSV file.
    
    Args:
        path: CSV file path
        limit: Maximum rows
    
    Returns:
        CSV data
    """
    try:
        from praisonaiagents.tools.csv_tools import CSVTools
        ct = CSVTools()
        data = ct.read_csv(path, limit=limit)
        return {"path": path, "data": data, "success": True}
    except ImportError:
        import csv
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for i, row in enumerate(reader) if i < limit]
        return {"path": path, "rows": rows, "count": len(rows), "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def write_csv(path: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Write CSV file.
    
    Args:
        path: CSV file path
        data: Data to write
    
    Returns:
        Write status
    """
    try:
        from praisonaiagents.tools.csv_tools import CSVTools
        ct = CSVTools()
        ct.write_csv(path, data)
        return {"path": path, "rows": len(data), "success": True}
    except ImportError:
        import csv
        if data:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        return {"path": path, "rows": len(data), "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def read_json_file(path: str) -> Dict[str, Any]:
    """Read JSON file.
    
    Args:
        path: JSON file path
    
    Returns:
        JSON data
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return {"path": path, "data": data, "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def write_json_file(path: str, data: Any) -> Dict[str, Any]:
    """Write JSON file.
    
    Args:
        path: JSON file path
        data: Data to write
    
    Returns:
        Write status
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return {"path": path, "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def read_yaml_file(path: str) -> Dict[str, Any]:
    """Read YAML file.
    
    Args:
        path: YAML file path
    
    Returns:
        YAML data
    """
    try:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {"path": path, "data": data, "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


def write_yaml_file(path: str, data: Any) -> Dict[str, Any]:
    """Write YAML file.
    
    Args:
        path: YAML file path
        data: Data to write
    
    Returns:
        Write status
    """
    try:
        import yaml
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return {"path": path, "success": True}
    except Exception as e:
        return {"path": path, "error": str(e), "success": False}


# =============================================================================
# CODE EXECUTION TOOLS
# =============================================================================

def run_python(code: str) -> Dict[str, Any]:
    """Execute Python code.
    
    Args:
        code: Python code to execute
    
    Returns:
        Execution result
    """
    try:
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        result = local_vars.get('result', str(local_vars) if local_vars else "Executed")
        return {"code": code[:100], "result": str(result), "success": True}
    except Exception as e:
        return {"code": code[:100], "error": str(e), "success": False}


def run_shell(command: str, cwd: str = None) -> Dict[str, Any]:
    """Execute shell command.
    
    Args:
        command: Shell command
        cwd: Working directory
    
    Returns:
        Command output
    """
    try:
        import subprocess
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=cwd
        )
        return {
            "command": command,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except Exception as e:
        return {"command": command, "error": str(e), "success": False}


def git_commit(message: str = None, path: str = ".") -> Dict[str, Any]:
    """Create a git commit.
    
    Args:
        message: Commit message (auto-generated if not provided)
        path: Repository path
    
    Returns:
        Commit status
    """
    try:
        import subprocess
        
        # Stage all changes
        subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True)
        
        # Generate message if not provided
        if not message:
            diff = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                cwd=path, capture_output=True, text=True
            )
            message = f"Auto-commit: {diff.stdout[:50]}" if diff.stdout else "Auto-commit"
        
        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=path, capture_output=True, text=True
        )
        
        return {
            "message": message,
            "output": result.stdout,
            "success": result.returncode == 0
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# UTILITY TOOLS
# =============================================================================

def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate mathematical expression.
    
    Args:
        expression: Math expression
    
    Returns:
        Calculation result
    """
    try:
        import ast
        import operator
        
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_node(node.operand))
            raise ValueError(f"Unsupported: {type(node)}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        return {"expression": expression, "result": result, "success": True}
    except Exception as e:
        return {"expression": expression, "error": str(e), "success": False}


def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Get current date/time.
    
    Args:
        timezone: Timezone name
    
    Returns:
        Current datetime
    """
    from datetime import datetime
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except:
        now = datetime.utcnow()
        timezone = "UTC"
    
    return {
        "datetime": now.isoformat(),
        "timezone": timezone,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "success": True
    }


def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get current stock price.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Stock price info
    """
    try:
        from praisonaiagents.tools.yfinance_tools import YFinanceTools
        yf = YFinanceTools()
        price = yf.get_stock_price(symbol)
        return {"symbol": symbol, "price": price, "success": True}
    except ImportError:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("regularMarketPrice"),
                "currency": info.get("currency"),
                "success": True
            }
        except:
            return {"symbol": symbol, "error": "yfinance not available", "success": False}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "success": False}


def get_stock_history(symbol: str, period: str = "1mo") -> Dict[str, Any]:
    """Get historical stock data.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 1y)
    
    Returns:
        Historical data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        data = hist.tail(10).to_dict('records') if not hist.empty else []
        return {"symbol": symbol, "period": period, "data": data, "success": True}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "success": False}


def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        System info
    """
    try:
        import platform
        import psutil
        return {
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "success": True
        }
    except ImportError:
        import platform
        return {"platform": platform.system(), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def list_processes(filter_name: str = None) -> Dict[str, Any]:
    """List running processes.
    
    Args:
        filter_name: Filter by name
    
    Returns:
        Process list
    """
    try:
        import psutil
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                info = p.info
                if filter_name and filter_name.lower() not in info['name'].lower():
                    continue
                procs.append(info)
            except:
                pass
        return {"processes": procs[:50], "count": len(procs), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# PLANNING & GUARDRAILS
# =============================================================================

def plan_create(goal: str) -> Dict[str, Any]:
    """Create a plan for a goal.
    
    Args:
        goal: Goal to plan for
    
    Returns:
        Generated plan
    """
    try:
        from praisonaiagents import Agent
        agent = Agent(instructions="Create a detailed step-by-step plan.")
        result = agent.start(f"Create a plan for: {goal}")
        return {"goal": goal, "plan": str(result), "success": True}
    except Exception as e:
        return {"goal": goal, "error": str(e), "success": False}


def guardrail_validate(content: str, rules: str) -> Dict[str, Any]:
    """Validate content against rules.
    
    Args:
        content: Content to validate
        rules: Validation rules
    
    Returns:
        Validation result
    """
    try:
        from praisonaiagents import LLMGuardrail
        guardrail = LLMGuardrail(rules=rules)
        result = guardrail.validate(content)
        return {"content": content[:100], "valid": result.is_valid, "success": True}
    except Exception as e:
        return {"content": content[:100], "error": str(e), "success": False}


# =============================================================================
# TODO TOOLS
# =============================================================================

_todos = []

def todo_add(task: str, priority: str = "medium") -> Dict[str, Any]:
    """Add task to todo list.
    
    Args:
        task: Task description
        priority: Priority level
    
    Returns:
        Add status
    """
    todo = {"id": len(_todos) + 1, "task": task, "priority": priority, "done": False}
    _todos.append(todo)
    return {"todo": todo, "success": True}


def todo_list() -> Dict[str, Any]:
    """List all tasks.
    
    Returns:
        All todos
    """
    return {"tasks": _todos, "count": len(_todos), "success": True}


def todo_complete(task_id: int) -> Dict[str, Any]:
    """Mark task as completed.
    
    Args:
        task_id: Task ID
    
    Returns:
        Completion status
    """
    for todo in _todos:
        if todo["id"] == task_id:
            todo["done"] = True
            return {"todo": todo, "success": True}
    return {"error": f"Task {task_id} not found", "success": False}


# =============================================================================
# MCP & DOCS TOOLS
# =============================================================================

def mcp_list_servers() -> Dict[str, Any]:
    """List available MCP servers.
    
    Returns:
        List of MCP servers
    """
    servers = [
        {"name": "filesystem", "package": "@modelcontextprotocol/server-filesystem"},
        {"name": "github", "package": "@modelcontextprotocol/server-github"},
        {"name": "postgres", "package": "@modelcontextprotocol/server-postgres"},
        {"name": "brave-search", "package": "@modelcontextprotocol/server-brave-search"},
        {"name": "memory", "package": "@modelcontextprotocol/server-memory"},
        {"name": "puppeteer", "package": "@modelcontextprotocol/server-puppeteer"},
    ]
    return {"servers": servers, "count": len(servers), "success": True}


def docs_search(query: str) -> Dict[str, Any]:
    """Search PraisonAI documentation.
    
    Args:
        query: Search query
    
    Returns:
        Documentation results
    """
    docs = {
        "agent": "https://docs.praison.ai/agents",
        "workflow": "https://docs.praison.ai/workflows",
        "memory": "https://docs.praison.ai/memory",
        "knowledge": "https://docs.praison.ai/knowledge",
        "tools": "https://docs.praison.ai/tools",
        "mcp": "https://docs.praison.ai/mcp",
    }
    
    results = []
    for topic, url in docs.items():
        if query.lower() in topic:
            results.append({"topic": topic, "url": url})
    
    return {"query": query, "results": results or [{"topic": "docs", "url": "https://docs.praison.ai"}], "success": True}


def hooks_list() -> Dict[str, Any]:
    """List available hooks.
    
    Returns:
        Available hooks
    """
    hooks = ["on_start", "on_complete", "on_error", "on_tool_call", "on_message", "on_step"]
    return {"hooks": hooks, "count": len(hooks), "success": True}


def rules_list() -> Dict[str, Any]:
    """List all rules.
    
    Returns:
        All rules
    """
    rules_dir = os.path.expanduser("~/.praisonai/rules")
    if not os.path.exists(rules_dir):
        return {"rules": [], "count": 0, "success": True}
    rules = [f.replace('.txt', '') for f in os.listdir(rules_dir) if f.endswith('.txt')]
    return {"rules": rules, "count": len(rules), "success": True}


def rules_add(name: str, content: str) -> Dict[str, Any]:
    """Add a rule.
    
    Args:
        name: Rule name
        content: Rule content
    
    Returns:
        Add status
    """
    rules_dir = os.path.expanduser("~/.praisonai/rules")
    os.makedirs(rules_dir, exist_ok=True)
    with open(os.path.join(rules_dir, f"{name}.txt"), 'w') as f:
        f.write(content)
    return {"name": name, "success": True}


def rules_get(name: str) -> Dict[str, Any]:
    """Get a rule.
    
    Args:
        name: Rule name
    
    Returns:
        Rule content
    """
    rule_file = os.path.expanduser(f"~/.praisonai/rules/{name}.txt")
    if os.path.exists(rule_file):
        with open(rule_file, 'r') as f:
            return {"name": name, "content": f.read(), "success": True}
    return {"name": name, "error": "Rule not found", "success": False}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Agent Tools (Primary)
AGENT_TOOLS = [run_agent, run_research, run_auto_agents, run_handoff, generate_agents_yaml]

# Workflow Tools
WORKFLOW_TOOLS = [workflow_run, workflow_create, workflow_from_yaml, export_to_n8n]

# Search Tools
SEARCH_TOOLS = [
    search_web, get_search_providers,
    tavily_search, tavily_extract,
    exa_search, exa_search_contents, exa_find_similar,
    ydc_search, ydc_news,
    duckduckgo_search, wikipedia_search, arxiv_search, searxng_search
]

# Crawl Tools
CRAWL_TOOLS = [scrape_page, extract_links, web_crawl, crawl4ai_scrape]

# Memory Tools
MEMORY_TOOLS = [memory_add, memory_search, memory_list, memory_clear]

# Knowledge Tools
KNOWLEDGE_TOOLS = [knowledge_add, knowledge_search]

# Session Tools
SESSION_TOOLS = [session_save, session_load, session_list]

# File Tools
FILE_TOOLS = [read_file, write_file, list_directory, read_csv, write_csv, read_json_file, write_json_file, read_yaml_file, write_yaml_file]

# Code Tools
CODE_TOOLS = [run_python, run_shell, git_commit]

# Utility Tools
UTILITY_TOOLS = [calculate, get_current_time, get_stock_price, get_stock_history, get_system_info, list_processes]

# Planning Tools
PLANNING_TOOLS = [plan_create, guardrail_validate]

# Todo Tools
TODO_TOOLS = [todo_add, todo_list, todo_complete]

# MCP & Docs Tools
MCP_TOOLS = [mcp_list_servers, docs_search, hooks_list, rules_list, rules_add, rules_get]

# All Tools
ALL_TOOLS = (
    AGENT_TOOLS +
    WORKFLOW_TOOLS +
    SEARCH_TOOLS +
    CRAWL_TOOLS +
    MEMORY_TOOLS +
    KNOWLEDGE_TOOLS +
    SESSION_TOOLS +
    FILE_TOOLS +
    CODE_TOOLS +
    UTILITY_TOOLS +
    PLANNING_TOOLS +
    TODO_TOOLS +
    MCP_TOOLS
)
