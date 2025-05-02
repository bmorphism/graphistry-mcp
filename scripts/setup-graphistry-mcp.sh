#!/bin/bash
# Setup script for Graphistry MCP server using uv

set -e

# Create a Python virtual environment
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install project in development mode with all dependencies
echo "Installing dependencies with uv..."
uv pip install -e ".[dev]"

# Install graphistry
echo "Installing graphistry with uv..."
uv pip install graphistry

echo "Setup complete! You can now run the server using:"
echo "./start-graphistry-mcp.sh"