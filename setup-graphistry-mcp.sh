#!/bin/bash
# Setup script for Graphistry MCP server using uv

set -e

# Create a Python virtual environment
echo "Creating virtual environment..."
uv venv

# Install project in development mode
echo "Installing project dependencies..."
uv pip install -e ".[dev]"

# Make the start script executable
chmod +x start-graphistry-mcp.sh

echo "Setup complete! You can start the server using:"
echo "./start-graphistry-mcp.sh"
echo "Or for HTTP mode:"
echo "./start-graphistry-mcp.sh --http [port]"