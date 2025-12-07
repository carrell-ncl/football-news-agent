# /Users/carrell/Desktop/dev/agentic_news/mcp_dev/hello_lowlevel.py
import sys
import anyio
from mcp.server.lowlevel import Server
from mcp.transport.stdio import stdio_server  # <-- correct import in mcp 1.16.0

print(">>> hello_lowlevel starting", file=sys.stderr, flush=True)

server = Server("hello-lowlevel")

@server.tool()
async def ping() -> str:
    """Health check."""
    return "pong"

async def main():
    print(">>> hello_lowlevel entering stdio", file=sys.stderr, flush=True)
    transport = stdio_server()
    async with transport:
        await server.run(transport)
    print(">>> hello_lowlevel exited stdio", file=sys.stderr, flush=True)

if __name__ == "__main__":
    anyio.run(main)
