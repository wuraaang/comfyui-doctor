"""MCP client for Comfy-Pilot integration.

Connects to Comfy-Pilot MCP server to get richer context about ComfyUI state:
- System status, node info, installed custom nodes
- Visual perception (screenshot of outputs)
- Install custom nodes and download models via MCP

Falls back gracefully when MCP is not available.
"""

import json
from typing import Optional, Any


class MCPConnection:
    """Connection to a Comfy-Pilot MCP server.

    Usage:
        mcp = MCPConnection(url="http://localhost:8189/mcp")
        if mcp.connect():
            status = mcp.get_system_status()
    """

    def __init__(self, url: str = "http://localhost:8189/mcp"):
        self.url = url.rstrip("/")
        self._connected = False
        self._tools: list[str] = []

    def connect(self) -> bool:
        """Try to connect to the MCP server. Returns True if connected."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.url}/tools",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                self._tools = [t.get("name", "") for t in data.get("tools", [])]
                self._connected = True
                return True
        except Exception:
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    def list_tools(self) -> list[str]:
        return self._tools

    def _call_tool(self, tool_name: str, arguments: Optional[dict] = None) -> dict:
        """Call a tool on the MCP server."""
        if not self._connected:
            return {"error": "Not connected"}
        try:
            import urllib.request
            payload = {
                "name": tool_name,
                "arguments": arguments or {},
            }
            req = urllib.request.Request(
                f"{self.url}/call",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    # ── Wrappers ─────────────────────────────────────────────────────

    def get_system_status(self) -> dict:
        """Get ComfyUI system status via MCP."""
        return self._call_tool("get_system_status")

    def get_node_info(self, node_type: str = "") -> dict:
        """Get info about a specific node type or all nodes."""
        args = {"node_type": node_type} if node_type else {}
        return self._call_tool("get_node_info", args)

    def search_custom_nodes(self, query: str) -> dict:
        """Search for custom nodes via ComfyUI-Manager MCP."""
        return self._call_tool("search_custom_nodes", {"query": query})

    def view_image(self, filename: str, subfolder: str = "") -> dict:
        """Get image data/description via MCP (visual perception)."""
        return self._call_tool("view_image", {
            "filename": filename,
            "subfolder": subfolder,
        })

    def install_custom_node(self, repo_url: str) -> dict:
        """Install a custom node via MCP (alternative to git clone)."""
        return self._call_tool("install_custom_node", {"repo_url": repo_url})

    def download_model(self, url: str, filename: str, folder: str) -> dict:
        """Download a model via MCP."""
        return self._call_tool("download_model", {
            "url": url,
            "filename": filename,
            "folder": folder,
        })

    def enrich_context(self) -> dict:
        """Get enriched context from MCP for LLM.

        Returns extra info like installed nodes, running queue, etc.
        """
        context = {}
        if not self._connected:
            return context

        status = self.get_system_status()
        if "error" not in status:
            context["mcp_system_status"] = status

        return context
