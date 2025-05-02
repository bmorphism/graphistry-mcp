import asyncio
from .server import server

__version__ = "0.1.0"

def main():
    """Run the Graphistry MCP server."""
    asyncio.run(server.main())

if __name__ == "__main__":
    main()
