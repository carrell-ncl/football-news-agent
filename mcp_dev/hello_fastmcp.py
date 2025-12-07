import sys
from mcp.server.fastmcp import FastMCP

print(">>> hello_fastmcp starting", file=sys.stderr, flush=True)
mcp = FastMCP("hello-fastmcp")

@mcp.tool()
def ping() -> str:
    return "pong"

if __name__ == "__main__":
    print(">>> calling mcp.run()", file=sys.stderr, flush=True)
    mcp.run()
    print(">>> mcp.run() returned", file=sys.stderr, flush=True)
