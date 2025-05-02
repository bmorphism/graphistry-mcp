"""
Graphistry FastMCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from pathlib import Path

import pandas as pd
import networkx as nx
from fastmcp import FastMCP
from pydantic import Field, BaseModel

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment variables from {env_path.absolute()}")
except ImportError:
    pass

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize graphistry client
try:
    import graphistry
    HAS_GRAPHISTRY = True
    logger.info("Graphistry package available")
    
    # Check for graphistry credentials in environment
    import os
    GRAPHISTRY_USERNAME = os.environ.get("GRAPHISTRY_USERNAME")
    GRAPHISTRY_PASSWORD = os.environ.get("GRAPHISTRY_PASSWORD")
    
    if GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD:
        try:
            graphistry.register(
                api=3,
                protocol="https",
                server="hub.graphistry.com",
                username=GRAPHISTRY_USERNAME,
                password=GRAPHISTRY_PASSWORD
            )
            logger.info("✅ Graphistry client registered successfully with credentials")
        except Exception as e:
            logger.warning(f"❌ Failed to register Graphistry client: {str(e)}")
            logger.warning("Please check your Graphistry credentials and ensure your account is active")
    else:
        logger.warning("⚠️  Graphistry credentials not found in environment variables")
        logger.warning("Please set GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD environment variables")
        logger.warning("Visit https://hub.graphistry.com to sign up for a free account")
except ImportError:
    HAS_GRAPHISTRY = False
    logger.warning("❌ Graphistry package not found, visualization capabilities will be limited")
    logger.warning("Install the graphistry package with: uvx pip install graphistry")

# Logger is already configured above

# Initialize the FastMCP server
mcp = FastMCP(
    "Graphistry Graph Visualization",
    dependencies=[
        "graphistry",
        "pandas",
        "networkx"
    ],
)

# Initialize in-memory graph cache
graph_cache: Dict[str, Dict[str, Any]] = {}


class GraphFormat(str, Literal["pandas", "networkx", "edge_list"]):
    """The format of the input graph data."""
    pass


class Node(BaseModel):
    """A node in a graph with optional attributes."""
    id: str
    attrs: Dict[str, Any] = Field(default_factory=dict, 
                                  description="Additional node attributes")


class Edge(BaseModel):
    """An edge in a graph with optional attributes."""
    source: str
    target: str
    attrs: Dict[str, Any] = Field(default_factory=dict,
                                 description="Additional edge attributes")


class LayoutOptions(TypedDict, total=False):
    """Options for layout algorithms."""
    play: int
    strongGravity: bool
    layout: str


# Tools Implementation

@mcp.tool()
def visualize_graph(
    data_format: Annotated[GraphFormat, 
                          Field(description="The format of the input data")],
    nodes: Annotated[Optional[List[Node]], 
                    Field(default=None, 
                          description="List of nodes (required for edge_list format)")] = None,
    edges: Annotated[Optional[List[Edge]], 
                    Field(default=None, 
                          description="List of edges (required for edge_list format)")] = None,
    node_id: Annotated[Optional[str], 
                      Field(default=None, 
                            description="Column name for node IDs (for pandas format)")] = None,
    source: Annotated[Optional[str], 
                     Field(default=None, 
                           description="Column name for edge source (for pandas format)")] = None,
    destination: Annotated[Optional[str], 
                          Field(default=None, 
                                description="Column name for edge destination (for pandas format)")] = None,
    title: Annotated[Optional[str], 
                    Field(default="Graph Visualization", 
                          description="Title for the visualization")] = "Graph Visualization",
    description: Annotated[Optional[str], 
                          Field(default="", 
                                description="Description for the visualization")] = "",
) -> Dict[str, Any]:
    """
    Visualize a graph using Graphistry's GPU-accelerated renderer for pattern discovery 
    and relationship analysis.
    """
    logger.info(f"Visualizing graph in {data_format} format")
    
    g = graphistry.graph()
    
    if data_format == "pandas":
        # Convert edge data to pandas DataFrame
        if edges:
            edges_data = [{"source": e.source, "target": e.target, **e.attrs} for e in edges]
            edges_df = pd.DataFrame(edges_data)
            src = source or "source"
            dst = destination or "target"
            g = g.edges(edges_df, source=src, destination=dst)
        
        # Add nodes if provided
        if nodes:
            nodes_data = [{"id": n.id, **n.attrs} for n in nodes]
            nodes_df = pd.DataFrame(nodes_data)
            nid = node_id or "id"
            g = g.nodes(nodes_df, nid)
    
    elif data_format == "networkx":
        # Create a NetworkX graph from provided nodes and edges
        nx_graph = nx.Graph()
        
        if nodes:
            for node in nodes:
                nx_graph.add_node(node.id, **node.attrs)
        
        if edges:
            for edge in edges:
                nx_graph.add_edge(edge.source, edge.target, **edge.attrs)
        
        g = g.networkx(nx_graph)
    
    elif data_format == "edge_list":
        # Create from simple edge list
        if not edges:
            raise ValueError("Edge list format requires edges parameter")
            
        edges_data = [{"source": e.source, "target": e.target, **e.attrs} for e in edges]
        edges_df = pd.DataFrame(edges_data)
        g = g.edges(edges_df)
    
    else:
        # This should be caught by pydantic but we'll include a check anyway
        raise ValueError(f"Unsupported data format: {data_format}")
    
    # Apply default settings
    g = g.settings(url_params={'play': 7000, 'strongGravity': True})
    
    # Generate a unique ID for this graph
    graph_id = str(uuid.uuid4())
    
    # Store the graph in our cache
    graph_cache[graph_id] = {
        "graph": g,
        "title": title,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "url": g.plot(render=False)
    }
    
    return {
        "graph_id": graph_id,
        "title": title,
        "description": description,
        "url": graph_cache[graph_id]["url"],
        "message": "Graph visualization created successfully"
    }


@mcp.tool()
def get_graph_info(
    graph_id: Annotated[str, Field(description="ID of the graph to retrieve information for")]
) -> Dict[str, Any]:
    """
    Get information about a stored graph visualization including metrics and statistics.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    graph_info = graph_cache[graph_id]
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


@mcp.tool()
def apply_layout(
    graph_id: Annotated[str, Field(description="ID of the graph to apply layout to")],
    layout: Annotated[str, Field(description="Layout algorithm to apply", 
                               enum=["force_directed", "radial", "circle", "grid"])]
) -> Dict[str, Any]:
    """
    Apply a layout algorithm to a graph for different analysis perspectives.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
    # Apply the requested layout
    layout_param: LayoutOptions = {}
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
    
    # Update the graph in cache
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


class AnalysisType(str, Literal["community_detection", "centrality", "path_finding", "anomaly_detection"]):
    """The type of graph analysis to perform."""
    pass


class AnalysisOptions(BaseModel):
    """Options for graph analysis."""
    algorithm: Optional[str] = Field(default="louvain", 
                                   description="Algorithm to use for community detection")
    top_n: Optional[int] = Field(default=5, 
                               description="Number of top nodes to return for centrality")
    source: Optional[str] = Field(default=None, 
                                description="Source node for path finding")
    target: Optional[str] = Field(default=None, 
                                description="Target node for path finding")


@mcp.tool()
def detect_patterns(
    graph_id: Annotated[str, Field(description="ID of the graph to analyze")],
    analysis_type: Annotated[AnalysisType, 
                           Field(description="Type of pattern analysis to perform")],
    options: Annotated[Optional[AnalysisOptions], 
                      Field(default=None, 
                            description="Additional options for the analysis")] = None
) -> Dict[str, Any]:
    """
    Identify interesting patterns, communities, and anomalies within graphs.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    if options is None:
        options = AnalysisOptions()
    
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
        except Exception:
            raise ValueError("Could not extract a NetworkX graph for analysis")
    
    result = {
        "graph_id": graph_id,
        "analysis_type": analysis_type,
    }
    
    # Perform the requested analysis
    if analysis_type == "community_detection":
        # Detect communities using appropriate algorithm
        algorithm = options.algorithm
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
            top_n = options.top_n
            
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
        source = options.source
        target = options.target
        
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
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            std_degree = (sum((d - avg_degree) ** 2 for d in degrees.values()) / len(degrees)) ** 0.5 if degrees else 0
            
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
        # This should be caught by pydantic but we'll include a check anyway
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    return result


def main() -> None:
    """
    Run the FastMCP server in the appropriate mode based on command line arguments.
    """
    # Check Graphistry registration status before starting server
    if HAS_GRAPHISTRY:
        if GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD:
            logger.info("Starting server with Graphistry credentials configured")
        else:
            logger.warning("⚠️  Starting server WITHOUT Graphistry credentials")
            logger.warning("For full functionality, please sign up at https://hub.graphistry.com")
            logger.warning("and set GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD environment variables")
    else:
        logger.warning("⚠️  Server starting without Graphistry package installed")
        logger.warning("Visualization capabilities will be limited")
    
    # Determine server mode from command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        port = 8080
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                logger.error(f"Invalid port: {sys.argv[2]}, using default 8080")
        
        logger.info(f"Starting Graphistry FastMCP server on HTTP port {port}")
        import uvicorn
        uvicorn.run(mcp.starlette, host="0.0.0.0", port=port)
    else:
        # Default mode uses stdio
        logger.info("Starting Graphistry FastMCP server with stdio transport")
        asyncio.run(mcp.run_stdio())


if __name__ == "__main__":
    main()