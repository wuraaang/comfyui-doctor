"""MCP client for Comfy-Pilot integration.

Connects to Comfy-Pilot REST endpoints on the ComfyUI server:
- /claude-code/mcp-status — check if Comfy-Pilot is installed
- /claude-code/workflow — get/set the current workflow
- /claude-code/graph-command — send graph manipulation commands
- /claude-code/platform — get platform info
- /claude-code/run-node — run workflow up to a specific node

Also wraps standard ComfyUI API endpoints:
- /system_stats — system info (GPU, VRAM, queue)
- /object_info — all registered node types
- /prompt — queue a workflow
- /history — execution history

Falls back gracefully when Comfy-Pilot is not available.
"""

import json
from typing import Optional, Any


class MCPConnection:
    """Connection to ComfyUI with optional Comfy-Pilot enhancement.

    Usage:
        mcp = MCPConnection(url="http://localhost:8188")
        if mcp.connect():
            status = mcp.get_system_status()
    """

    def __init__(self, url: str = "http://localhost:8188"):
        # Normalize: strip trailing slash, remove /mcp suffix for backward compat
        url = url.rstrip("/")
        if url.endswith("/mcp"):
            url = url[:-4]
        self.url = url
        self._connected = False
        self._has_comfy_pilot = False
        self._tools: list[str] = []

    def connect(self) -> bool:
        """Try to connect to ComfyUI and detect Comfy-Pilot."""
        try:
            import urllib.request

            # First, check if ComfyUI itself is reachable
            req = urllib.request.Request(
                f"{self.url}/system_stats",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if "system" not in data:
                    return False
                self._connected = True

            # Then check if Comfy-Pilot is installed
            try:
                req2 = urllib.request.Request(
                    f"{self.url}/claude-code/mcp-status",
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(req2, timeout=3) as resp2:
                    pilot_data = json.loads(resp2.read().decode())
                    self._has_comfy_pilot = pilot_data.get("connected", False)
                    tool_count = pilot_data.get("tools", 0)
                    self._tools = [
                        "get_workflow", "summarize_workflow", "get_node_types",
                        "get_node_info", "get_status", "run", "edit_graph",
                        "view_image", "search_custom_nodes", "install_custom_node",
                        "uninstall_custom_node", "update_custom_node",
                        "download_model", "center_on_node",
                    ][:tool_count]
            except Exception:
                # Comfy-Pilot not installed — still connected to ComfyUI
                self._has_comfy_pilot = False

            return True
        except Exception:
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    def has_comfy_pilot(self) -> bool:
        return self._has_comfy_pilot

    def list_tools(self) -> list[str]:
        return self._tools

    def _request(self, path: str, method: str = "GET",
                 data: Optional[dict] = None, timeout: int = 10) -> dict:
        """Make an HTTP request to ComfyUI."""
        if not self._connected:
            return {"error": "Not connected"}
        try:
            import urllib.request
            url = f"{self.url}{path}"
            req_data = json.dumps(data).encode() if data else None
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    # ── Standard ComfyUI API ────────────────────────────────────────

    def get_system_status(self) -> dict:
        """Get ComfyUI system stats (GPU, VRAM, queue)."""
        return self._request("/system_stats")

    def get_object_info(self, node_type: str = "") -> dict:
        """Get registered node types from ComfyUI."""
        if node_type:
            return self._request(f"/object_info/{node_type}")
        return self._request("/object_info")

    def get_queue(self) -> dict:
        """Get current prompt queue."""
        return self._request("/queue")

    def get_history(self, prompt_id: str = "") -> dict:
        """Get execution history."""
        if prompt_id:
            return self._request(f"/history/{prompt_id}")
        return self._request("/history")

    # ── Comfy-Pilot endpoints ───────────────────────────────────────

    def get_workflow(self) -> dict:
        """Get current workflow from Comfy-Pilot (requires browser open)."""
        if not self._has_comfy_pilot:
            return {"error": "Comfy-Pilot not available"}
        return self._request("/claude-code/workflow")

    def send_graph_command(self, action: str, params: Optional[dict] = None) -> dict:
        """Send a graph manipulation command via Comfy-Pilot."""
        if not self._has_comfy_pilot:
            return {"error": "Comfy-Pilot not available"}
        return self._request(
            "/claude-code/graph-command",
            method="POST",
            data={"action": action, "params": params or {}},
            timeout=10,
        )

    def get_platform_info(self) -> dict:
        """Get platform info from Comfy-Pilot."""
        if not self._has_comfy_pilot:
            return {"error": "Comfy-Pilot not available"}
        return self._request("/claude-code/platform")

    def run_node(self, node_id: str) -> dict:
        """Run workflow up to a specific node via Comfy-Pilot."""
        if not self._has_comfy_pilot:
            return {"error": "Comfy-Pilot not available"}
        return self._request(
            "/claude-code/run-node",
            method="POST",
            data={"node_id": node_id},
        )

    # ── Backward-compat wrappers (match old MCP tool names) ────────

    def get_node_info(self, node_type: str = "") -> dict:
        """Get info about a specific node type."""
        return self.get_object_info(node_type)

    def search_custom_nodes(self, query: str) -> dict:
        """Search custom nodes — uses Comfy-Pilot if available, else ComfyUI Manager API."""
        if self._has_comfy_pilot:
            return self.send_graph_command("search_custom_nodes", {"query": query})
        return {"error": "search_custom_nodes requires Comfy-Pilot"}

    def view_image(self, filename: str, subfolder: str = "") -> dict:
        """Get image from ComfyUI output."""
        params = {"filename": filename}
        if subfolder:
            params["subfolder"] = subfolder
        # Use standard ComfyUI /view endpoint
        try:
            import urllib.request
            import urllib.parse
            qs = urllib.parse.urlencode(params)
            url = f"{self.url}/view?{qs}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_type = resp.headers.get("Content-Type", "")
                return {
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": int(resp.headers.get("Content-Length", 0)),
                }
        except Exception as e:
            return {"error": str(e)}

    def install_custom_node(self, repo_url: str) -> dict:
        """Install a custom node — delegates to Comfy-Pilot if available."""
        if self._has_comfy_pilot:
            return self.send_graph_command("install_custom_node", {"repo_url": repo_url})
        return {"error": "install_custom_node requires Comfy-Pilot"}

    def download_model(self, url: str, filename: str, folder: str) -> dict:
        """Download a model — delegates to Comfy-Pilot if available."""
        if self._has_comfy_pilot:
            return self.send_graph_command("download_model", {
                "url": url, "filename": filename, "folder": folder,
            })
        return {"error": "download_model requires Comfy-Pilot"}

    # ── Context enrichment for LLM ──────────────────────────────────

    def enrich_context(self) -> dict:
        """Get enriched context from ComfyUI for LLM.

        Returns system stats, queue info, and Comfy-Pilot data if available.
        """
        context = {}
        if not self._connected:
            return context

        # System stats (always available)
        stats = self.get_system_status()
        if "error" not in stats:
            system = stats.get("system", {})
            devices = stats.get("devices", [])
            context["system"] = {
                "os": system.get("os", ""),
                "python_version": system.get("python_version", ""),
                "comfyui_version": system.get("comfyui_version", ""),
                "pytorch_version": system.get("pytorch_version", ""),
                "embedded_python": system.get("embedded_python", False),
            }
            if devices:
                context["gpu"] = {
                    "name": devices[0].get("name", ""),
                    "type": devices[0].get("type", ""),
                    "vram_total_mb": devices[0].get("vram_total", 0) // (1024 * 1024),
                    "vram_free_mb": devices[0].get("vram_free", 0) // (1024 * 1024),
                    "torch_vram_total_mb": devices[0].get("torch_vram_total", 0) // (1024 * 1024),
                    "torch_vram_free_mb": devices[0].get("torch_vram_free", 0) // (1024 * 1024),
                }

        # Queue info
        queue = self.get_queue()
        if "error" not in queue:
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            context["queue"] = {
                "running": len(running),
                "pending": len(pending),
            }

        # Comfy-Pilot specific
        if self._has_comfy_pilot:
            context["comfy_pilot"] = True
            platform = self.get_platform_info()
            if "error" not in platform:
                context["platform"] = platform

        return context
