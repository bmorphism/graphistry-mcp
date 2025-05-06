"""
Tests for the Graphistry MCP server.
"""
import json
from typing import Dict, Any, List
import pytest
from graphistry_mcp_server.server import list_tools, call_tool


@pytest.mark.asyncio
async def test_list_tools() -> None:
    """Test that list_tools returns a list of valid tool definitions."""
    tools: List[Dict[str, Any]] = await list_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check that each tool has the required structure
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool


@pytest.mark.asyncio
async def test_visualize_graph_tool() -> None:
    """Test that the visualize_graph tool works with basic parameters."""
    # Create a simple test graph
    test_params: Dict[str, Any] = {
        "data_format": "edge_list",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"}
        ],
        "title": "Test Graph"
    }
    
    result: Dict[str, Any] = await call_tool("visualize_graph", test_params)
    assert "graph_id" in result
    assert "url" in result
    assert result["title"] == "Test Graph"


@pytest.mark.asyncio
async def test_get_graph_info_tool() -> None:
    """Test that the get_graph_info tool works after creating a graph."""
    # First create a graph
    create_params: Dict[str, Any] = {
        "data_format": "edge_list",
        "edges": [
            {"source": "1", "target": "2"},
        ],
        "title": "Info Test Graph"
    }
    
    create_result: Dict[str, Any] = await call_tool("visualize_graph", create_params)
    graph_id: str = create_result["graph_id"]
    
    # Now test getting info about it
    info_params: Dict[str, str] = {
        "graph_id": graph_id
    }
    
    info_result: Dict[str, Any] = await call_tool("get_graph_info", info_params)
    assert info_result["graph_id"] == graph_id
    assert info_result["title"] == "Info Test Graph"