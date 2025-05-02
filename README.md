# Graphistry MCP Server

A Model Context Protocol (MCP) server for integrating Graphistry's graph visualization capabilities with LLM workflows, focusing on advanced graph insights and investigations for network analysis, threat detection, and pattern discovery.

Developed by the Graphistry Community.

## Features

- GPU-accelerated graph visualization via Graphistry
- Advanced pattern discovery and relationship analysis
- Streamable HTTP interface for resumable connections
- Support for various graph data formats (Pandas, NetworkX, edge lists)
- Interactive graph visualization and exploration
- Layout control for different visualization types
- Network investigation and anomaly detection capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graphistry-mcp.git
cd graphistry-mcp

# Run the setup script which uses uv for dependency management
./setup-graphistry-mcp.sh
```

## Usage

### Starting the server

The server can be run in two modes:

1. Standard stdio mode (for typical MCP clients):
```bash
./start-graphistry-mcp.sh
```

2. HTTP mode (for web-based clients or testing):
```bash
./start-graphistry-mcp.sh --http 8080
```

### Available Tools

The Graphistry MCP server provides the following tools for graph insights and investigations:

1. **visualize_graph** - Create a graph visualization from different data formats:
   - Supports pandas, networkx, and edge_list formats
   - Customizable node and edge attributes
   - Returns a unique graph ID and visualization URL
   - Reveals patterns and connections in complex data

2. **get_graph_info** - Retrieve information about a stored graph:
   - Access metadata for previously created visualizations
   - Get the visualization URL for sharing
   - Analyze graph metrics and statistics

3. **apply_layout** - Change the layout algorithm for a graph:
   - Force directed layout for natural clustering
   - Radial layout for hierarchical relationship investigation
   - Circle layout for symmetry analysis
   - Grid layout for structural comparisons

4. **detect_patterns** - Identify interesting patterns within graphs:
   - Community detection for network segmentation
   - Path finding between key entities
   - Centrality metrics for key node identification
   - Anomaly detection for outlier identification

### Example Client Usage

```python
import asyncio
from mcp.client import Client
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client() as client:
        # List available tools
        tools = await client.list_tools()
        
        # Create a simple graph
        result = await client.call_tool("visualize_graph", {
            "data_format": "edge_list",
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
                {"source": "C", "target": "A"}
            ],
            "title": "Triangle Graph"
        })
        
        print(f"Graph created: {result['url']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Docker Usage

You can also run the server in a Docker container:

```bash
# Build the Docker image
docker build -t graphistry-mcp .

# Run the container
docker run -p 8080:8080 graphistry-mcp
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .

# Type check
mypy src/
```

## License

MIT