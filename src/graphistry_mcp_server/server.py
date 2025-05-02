"""
Graphistry MCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import graphistry
import pandas as pd
import networkx as nx
from mcp.server import Server
from mcp.server import stdio
import mcp.types as mcp_types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("graphistry-mcp-server")

# Initialize state
graph_cache = {}

@server.list_tools
async def list_tools() -> List[Dict[str, Any]]:
    """List available Graphistry visualization tools."""
    return [
        {
            "name": "visualize_graph",
            "description": "Visualize a graph using Graphistry's GPU-accelerated renderer",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_format": {
                        "type": "string",
                        "description": "The format of the input data (pandas, networkx, edge_list)",
                        "enum": ["pandas", "networkx", "edge_list"]
                    },
                    "nodes": {
                        "type": "array",
                        "description": "List of nodes (required for edge_list format)",
                        "items": {
                            "type": "object"
                        }
                    },
                    "edges": {
                        "type": "array",
                        "description": "List of edges (required for edge_list format)",
                        "items": {
                            "type": "object"
                        }
                    },
                    "node_id": {
                        "type": "string",
                        "description": "Column name for node IDs (for pandas format)"
                    },
                    "source": {
                        "type": "string",
                        "description": "Column name for edge source (for pandas format)"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Column name for edge destination (for pandas format)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the visualization"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description for the visualization"
                    }
                },
                "required": ["data_format"]
            }
        },
        {
            "name": "get_graph_info",
            "description": "Get information about a stored graph visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {
                        "type": "string",
                        "description": "ID of the graph to retrieve information for"
                    }
                },
                "required": ["graph_id"]
            }
        },
        {
            "name": "apply_layout",
            "description": "Apply a layout algorithm to a graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {
                        "type": "string",
                        "description": "ID of the graph to apply layout to"
                    },
                    "layout": {
                        "type": "string",
                        "description": "Layout algorithm to apply",
                        "enum": ["force_directed", "radial", "circle", "grid"]
                    }
                },
                "required": ["graph_id", "layout"]
            }
        }
    ]

@server.call_tool
async def call_tool(name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool calls for Graphistry visualization."""
    logger.info(f"Tool call: {name} with parameters: {parameters}")
    
    if name == "visualize_graph":
        return await handle_visualize_graph(parameters)
    elif name == "get_graph_info":
        return await handle_get_graph_info(parameters)
    elif name == "apply_layout":
        return await handle_apply_layout(parameters)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_visualize_graph(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle visualization of graph data."""
    data_format = parameters.get("data_format")
    title = parameters.get("title", "Graph Visualization")
    description = parameters.get("description", "")
    
    g = graphistry.graph()
    
    if data_format == "pandas":
        # Convert the edge data to pandas DataFrame
        if "edges" in parameters:
            edges_df = pd.DataFrame(parameters["edges"])
            source = parameters.get("source", "src")
            destination = parameters.get("destination", "dst")
            g = g.edges(edges_df, source=source, destination=destination)
        
        # Add nodes if provided
        if "nodes" in parameters:
            nodes_df = pd.DataFrame(parameters["nodes"])
            node_id = parameters.get("node_id", "id")
            g = g.nodes(nodes_df, node_id)
    
    elif data_format == "networkx":
        # Create a NetworkX graph from the provided data
        nx_graph = nx.Graph()
        
        if "nodes" in parameters:
            for node in parameters["nodes"]:
                nx_graph.add_node(node["id"], **{k:v for k,v in node.items() if k != "id"})
        
        if "edges" in parameters:
            for edge in parameters["edges"]:
                nx_graph.add_edge(
                    edge["source"], 
                    edge["target"], 
                    **{k:v for k,v in edge.items() if k not in ["source", "target"]}
                )
        
        g = g.networkx(nx_graph)
    
    elif data_format == "edge_list":
        # Create from simple edge list
        if "edges" not in parameters:
            raise ValueError("Edge list format requires 'edges' parameter")
            
        edges = parameters["edges"]
        edges_df = pd.DataFrame(edges)
        g = g.edges(edges_df)
    
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    # Apply settings
    g = g.settings(url_params={'play': 7000, 'strongGravity': True})
    
    # Generate a unique ID for this graph
    import uuid
    graph_id = str(uuid.uuid4())
    
    # Store the graph in our cache
    graph_cache[graph_id] = {
        "graph": g,
        "title": title,
        "description": description,
        "created_at": pd.Timestamp.now().isoformat(),
        "url": g.plot(render=False)
    }
    
    return {
        "graph_id": graph_id,
        "title": title,
        "description": description,
        "url": graph_cache[graph_id]["url"],
        "message": "Graph visualization created successfully"
    }

async def handle_get_graph_info(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Get information about a stored graph."""
    graph_id = parameters.get("graph_id")
    if not graph_id or graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    graph_info = graph_cache[graph_id]
    return {
        "graph_id": graph_id,
        "title": graph_info["title"],
        "description": graph_info["description"],
        "created_at": graph_info["created_at"],
        "url": graph_info["url"]
    }

async def handle_apply_layout(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a layout algorithm to a graph."""
    graph_id = parameters.get("graph_id")
    layout = parameters.get("layout")
    
    if not graph_id or graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    if not layout:
        raise ValueError("Layout algorithm must be specified")
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
    # Apply the requested layout
    layout_param = {}
    if layout == "force_directed":
        layout_param = {'play': 7000, 'strongGravity': True}
    elif layout == "radial":
        layout_param = {'play': 0, 'layout': 'radial'}
    elif layout == "circle":
        layout_param = {'play': 0, 'layout': 'circle'}
    elif layout == "grid":
        layout_param = {'play': 0, 'layout': 'grid'}
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    g = g.settings(url_params=layout_param)
    
    # Update the graph in our cache
    graph_cache[graph_id] = {
        **graph_info,
        "graph": g,
        "url": g.plot(render=False)
    }
    
    return {
        "graph_id": graph_id,
        "title": graph_info["title"],
        "layout": layout,
        "url": graph_cache[graph_id]["url"],
        "message": f"Applied {layout} layout to graph"
    }

async def main() -> None:
    """Run the server with streamable HTTP support."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        from mcp.server.streamable_http import streamable_http_server
        
        port = 8080
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                logger.error(f"Invalid port: {sys.argv[2]}, using default 8080")
        
        logger.info(f"Starting Graphistry MCP server on HTTP port {port}")
        await streamable_http_server(server, host="0.0.0.0", port=port)
    else:
        # Default to stdio for CLI usage
        logger.info("Starting Graphistry MCP server with stdio transport")
        async with stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())