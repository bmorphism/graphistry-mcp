#!/bin/bash
# Script to start the Graphistry MCP server

# Check if HTTP mode is requested
if [ "$1" == "--http" ]; then
    PORT=${2:-8080}
    echo "Starting Graphistry MCP server in HTTP mode on port $PORT"
    uvx python -m graphistry_mcp_server.server --http $PORT
else
    echo "Starting Graphistry MCP server in stdio mode"
    uvx python -m graphistry_mcp_server.server
fi