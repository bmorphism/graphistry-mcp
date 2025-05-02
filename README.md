# Graphistry FastMCP Server

A [Model-Control-Program (MCP)](https://github.com/llmOS/mcp) server for integrating Graphistry's graph visualization capabilities with LLM workflows, focusing on advanced graph insights and investigations for network analysis, threat detection, and pattern discovery.

Built using [FastMCP](https://github.com/jlowin/fastmcp), this implementation provides a streamlined API with robust HTTP streaming support.

Developed by the Graphistry Community.

## 🚨 Important: Graphistry Registration Required

**This MCP server requires a free Graphistry account to use visualization features.**

1. Sign up for a free account at [hub.graphistry.com](https://hub.graphistry.com)
2. Set your credentials as environment variables before starting the server:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

Without these credentials, certain visualization features will be limited.

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
git clone https://github.com/bmorphism/graphistry-mcp.git
cd graphistry-mcp

# Run the setup script which uses uv for dependency management
./setup-graphistry-mcp.sh

# Set up your Graphistry credentials (obtained from hub.graphistry.com)
export GRAPHISTRY_USERNAME=your_username
export GRAPHISTRY_PASSWORD=your_password
```

## Usage

### Graphistry Account Setup

Before using this server, you must:

1. Register for a free account at [hub.graphistry.com](https://hub.graphistry.com)
2. Set your credentials as environment variables:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

These credentials enable the server to create and access GPU-accelerated visualizations.

### Starting the server

The server can be run in two modes:

1. Standard stdio mode (for typical MCP clients):
```bash
# Make sure your Graphistry credentials are set before running
./start-graphistry-mcp.sh
```

2. HTTP mode (for web-based clients or testing):
```bash
# Make sure your Graphistry credentials are set before running
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

### Advanced Visualization Features

The following additional visualization capabilities are available when properly authenticated with Graphistry:

1. **interactive_exploration** - Interact with live visualizations:
   - Pan, zoom, and explore complex graph structures
   - Click nodes and edges to reveal detailed information
   - Filter and highlight specific patterns within the visualization

2. **graph_embedding** - Generate embeddings for graph analysis:
   - Visualize graph embeddings in 2D or 3D space
   - Identify clusters and patterns through dimensional reduction
   - Compare similarity between different graph structures

3. **time_series_analysis** - Visualize how graphs evolve over time:
   - Play back temporal graph changes as animations
   - Track entity relationships as they form and dissolve
   - Identify patterns in temporal network dynamics

4. **annotation_tools** - Add context to your visualizations:
   - Highlight important nodes or relationships
   - Add explanatory text to visualizations
   - Create shareable, annotated graph stories

### Example Client Usage

```python
import asyncio
import os
from mcp.client import Client
from mcp.client.stdio import stdio_client

# Set up Graphistry credentials before starting client
# These should match the credentials used on your Graphistry account
os.environ["GRAPHISTRY_USERNAME"] = "your_graphistry_username"  # Replace with your username
os.environ["GRAPHISTRY_PASSWORD"] = "your_graphistry_password"  # Replace with your password

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
        print("Open this URL in your browser to view your visualization")

if __name__ == "__main__":
    asyncio.run(main())
```

> **Note**: The visualization URL will only work if you have properly set up your Graphistry credentials and have an active account at [hub.graphistry.com](https://hub.graphistry.com).

## Docker Usage

You can also run the server in a Docker container with your Graphistry credentials:

```bash
# Build the Docker image
docker build -t graphistry-mcp .

# Run the container with environment variables
docker run -p 8080:8080 \
  -e GRAPHISTRY_USERNAME=your_username \
  -e GRAPHISTRY_PASSWORD=your_password \
  graphistry-mcp
```

These environment variables will be passed to the container, allowing it to authenticate with Graphistry's services.

## Development

```bash
# Install development dependencies
uvx pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .

# Type check
mypy src/
```

## Graphistry Authentication Details

### Account Registration

1. Visit [hub.graphistry.com](https://hub.graphistry.com) and sign up for a free account
2. After registration, you'll have access to Graphistry's GPU-accelerated visualization platform
3. Keep your username and password secure for use with this MCP server

### Credential Management

There are several ways to provide your Graphistry credentials:

1. **Environment variables** (recommended):
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

2. **Configuration file**:
   Create a `.env` file in the project root:
   ```
   GRAPHISTRY_USERNAME=your_username
   GRAPHISTRY_PASSWORD=your_password
   ```

3. **Command-line arguments** (when starting the server):
   ```bash
   ./start-graphistry-mcp.sh --graphistry-username=your_username --graphistry-password=your_password
   ```

### Troubleshooting Authentication

If you encounter authentication issues:

1. Verify your credentials at [hub.graphistry.com](https://hub.graphistry.com)
2. Check for typos in your username or password
3. Ensure your account has been activated (check your email)
4. Look for error messages in the server logs that might indicate authentication problems

### Rate Limits and Usage

Free Graphistry accounts have certain usage limitations. For high-volume or production usage, consider upgrading to a paid plan at [graphistry.com/plans](https://www.graphistry.com/plans).

## License

MIT