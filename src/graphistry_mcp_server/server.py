"""
Graphistry MCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
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
    """List available Graphistry visualization tools for graph insights and investigations."""
    return [
        {
            "name": "visualize_graph",
            "description": "Visualize a graph using Graphistry's GPU-accelerated renderer for pattern discovery and relationship analysis",
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
            "description": "Get information about a stored graph visualization including metrics and statistics",
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
            "description": "Apply a layout algorithm to a graph for different analysis perspectives",
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
        },
        {
            "name": "detect_patterns",
            "description": "Identify interesting patterns, communities, and anomalies within graphs",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {
                        "type": "string",
                        "description": "ID of the graph to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of pattern analysis to perform",
                        "enum": ["community_detection", "centrality", "path_finding", "anomaly_detection"]
                    },
                    "options": {
                        "type": "object",
                        "description": "Additional options for the analysis"
                    }
                },
                "required": ["graph_id", "analysis_type"]
            }
        }
    ]

@server.call_tool
async def call_tool(name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool calls for Graphistry visualization and analytics."""
    logger.info(f"Tool call: {name} with parameters: {parameters}")
    
    if name == "visualize_graph":
        return await handle_visualize_graph(parameters)
    elif name == "get_graph_info":
        return await handle_get_graph_info(parameters)
    elif name == "apply_layout":
        return await handle_apply_layout(parameters)
    elif name == "detect_patterns":
        return await handle_detect_patterns(parameters)
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
    
    # Get the graph object
    g = graph_info["graph"]
    
    # Try to extract basic graph metrics if possible
    metrics = {}
    try:
        # If we can get a networkx graph, compute some metrics
        if hasattr(g, '_nx'):
            nx_graph = g._nx
            metrics = {
                "num_nodes": nx_graph.number_of_nodes(),
                "num_edges": nx_graph.number_of_edges(),
                "density": nx.density(nx_graph),
                "is_connected": nx.is_connected(nx_graph) if nx_graph.number_of_nodes() > 0 else False,
                "average_clustering": nx.average_clustering(nx_graph) if nx_graph.number_of_nodes() > 0 else 0,
            }
    except Exception as e:
        logger.warning(f"Could not compute graph metrics: {e}")
    
    return {
        "graph_id": graph_id,
        "title": graph_info["title"],
        "description": graph_info["description"],
        "created_at": graph_info["created_at"],
        "url": graph_info["url"],
        "metrics": metrics
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

async def handle_detect_patterns(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Detect patterns, communities, and anomalies in a graph."""
    graph_id = parameters.get("graph_id")
    analysis_type = parameters.get("analysis_type")
    options = parameters.get("options", {})
    
    if not graph_id or graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    if not analysis_type:
        raise ValueError("Analysis type must be specified")
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
    # Extract NetworkX graph for analysis
    nx_graph = None
    if hasattr(g, '_nx'):
        nx_graph = g._nx
    else:
        # Try to get a networkx graph from edges
        try:
            edges_df = g._edges
            nx_graph = nx.from_pandas_edgelist(edges_df)
        except:
            raise ValueError("Could not extract a NetworkX graph for analysis")
    
    result = {
        "graph_id": graph_id,
        "analysis_type": analysis_type,
    }
    
    # Perform the requested analysis
    if analysis_type == "community_detection":
        # Detect communities using appropriate algorithm
        algorithm = options.get("algorithm", "louvain")
        if algorithm == "louvain":
            try:
                from community import best_partition
                partition = best_partition(nx_graph)
                # Convert partition to a format for visualization
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)
                
                # Count nodes in each community
                community_sizes = {comm_id: len(nodes) for comm_id, nodes in communities.items()}
                
                result["communities"] = communities
                result["community_sizes"] = community_sizes
                result["num_communities"] = len(communities)
                
                # Update graph visualization with community colors
                node_attrs = {node: {"community": comm} for node, comm in partition.items()}
                for node, attrs in node_attrs.items():
                    if nx_graph.has_node(node):
                        nx_graph.nodes[node].update(attrs)
                
                # Update the graph in cache with community information
                g = g.networkx(nx_graph)
                graph_cache[graph_id]["graph"] = g
                graph_cache[graph_id]["url"] = g.plot(render=False)
                result["url"] = graph_cache[graph_id]["url"]
                
            except ImportError:
                result["error"] = "Community detection requires the python-louvain package"
        else:
            result["error"] = f"Unsupported community detection algorithm: {algorithm}"
    
    elif analysis_type == "centrality":
        # Calculate various centrality metrics
        try:
            # Get top N nodes by different centrality measures
            top_n = options.get("top_n", 5)
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(nx_graph)
            degree_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(nx_graph)
            betweenness_central_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(nx_graph)
            closeness_central_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            result["centrality"] = {
                "degree": {str(node): value for node, value in degree_central_nodes},
                "betweenness": {str(node): value for node, value in betweenness_central_nodes},
                "closeness": {str(node): value for node, value in closeness_central_nodes}
            }
            
            # Update node attributes with centrality measures
            for node in nx_graph.nodes():
                nx_graph.nodes[node]["degree_centrality"] = degree_centrality.get(node, 0)
                nx_graph.nodes[node]["betweenness_centrality"] = betweenness_centrality.get(node, 0)
                nx_graph.nodes[node]["closeness_centrality"] = closeness_centrality.get(node, 0)
            
            # Update the graph in cache with centrality information
            g = g.networkx(nx_graph)
            graph_cache[graph_id]["graph"] = g
            graph_cache[graph_id]["url"] = g.plot(render=False)
            result["url"] = graph_cache[graph_id]["url"]
            
        except Exception as e:
            result["error"] = f"Error calculating centrality: {str(e)}"
    
    elif analysis_type == "path_finding":
        # Find paths between nodes
        source = options.get("source")
        target = options.get("target")
        
        if not source or not target:
            result["error"] = "Both source and target nodes are required for path finding"
        else:
            try:
                if nx.has_path(nx_graph, source, target):
                    shortest_path = nx.shortest_path(nx_graph, source, target)
                    result["path"] = shortest_path
                    result["path_length"] = len(shortest_path) - 1
                    
                    # Highlight the path in the visualization
                    path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
                    for u, v in nx_graph.edges():
                        nx_graph.edges[u, v]["on_path"] = False
                    
                    for u, v in path_edges:
                        if nx_graph.has_edge(u, v):
                            nx_graph.edges[u, v]["on_path"] = True
                    
                    # Update the graph in cache with path information
                    g = g.networkx(nx_graph)
                    graph_cache[graph_id]["graph"] = g
                    graph_cache[graph_id]["url"] = g.plot(render=False)
                    result["url"] = graph_cache[graph_id]["url"]
                else:
                    result["path_exists"] = False
                    result["message"] = f"No path exists between {source} and {target}"
            except Exception as e:
                result["error"] = f"Error finding path: {str(e)}"
    
    elif analysis_type == "anomaly_detection":
        # Detect anomalies in the graph
        try:
            # Calculate various metrics to identify anomalies
            degrees = dict(nx_graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees)
            std_degree = (sum((d - avg_degree) ** 2 for d in degrees.values()) / len(degrees)) ** 0.5
            
            # Identify nodes with unusually high or low degree
            high_degree_threshold = avg_degree + 2 * std_degree
            low_degree_threshold = max(0, avg_degree - 2 * std_degree)
            
            high_degree_nodes = {node: deg for node, deg in degrees.items() if deg > high_degree_threshold}
            isolated_nodes = {node: deg for node, deg in degrees.items() if deg == 0}
            
            result["anomalies"] = {
                "high_degree_nodes": high_degree_nodes,
                "isolated_nodes": isolated_nodes,
                "avg_degree": avg_degree,
                "std_degree": std_degree
            }
            
            # Mark anomalous nodes in the graph
            for node in nx_graph.nodes():
                degree = degrees.get(node, 0)
                nx_graph.nodes[node]["anomaly"] = "high_degree" if degree > high_degree_threshold else ("isolated" if degree == 0 else "normal")
            
            # Update the graph in cache with anomaly information
            g = g.networkx(nx_graph)
            graph_cache[graph_id]["graph"] = g
            graph_cache[graph_id]["url"] = g.plot(render=False)
            result["url"] = graph_cache[graph_id]["url"]
            
        except Exception as e:
            result["error"] = f"Error detecting anomalies: {str(e)}"
    
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    return result

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