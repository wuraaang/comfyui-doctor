"""ComfyUI HTTP API client."""

import json
import time
import urllib.request
import urllib.error
import socket
from typing import Any, Optional
from pathlib import Path


class ComfyAPI:
    """Thin wrapper around ComfyUI's HTTP API."""

    def __init__(self, url: str = "http://127.0.0.1:8188", timeout: int = 30):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._object_info_cache: Optional[dict] = None
        self._object_info_time: float = 0
        self._cache_ttl: int = 300  # 5 min

    # ── Low-level request ─────────────────────────────────────────────

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        url = f"{self.url}{endpoint}"
        t = timeout or self.timeout
        try:
            if data:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method=method,
                )
            else:
                req = urllib.request.Request(url, method=method)

            with urllib.request.urlopen(req, timeout=t) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"Connection failed: {e}"}
        except socket.timeout:
            return {"error": f"Timeout after {t}s"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from ComfyUI"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ── Health check ──────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if ComfyUI is reachable."""
        result = self._request("/system_stats", timeout=5)
        return "error" not in result

    def system_stats(self) -> dict:
        return self._request("/system_stats")

    # ── Object info (node registry) ──────────────────────────────────

    def object_info(self, force: bool = False) -> dict:
        """Get all registered node types. Cached for 5 min."""
        now = time.time()
        if (
            not force
            and self._object_info_cache
            and (now - self._object_info_time) < self._cache_ttl
        ):
            return self._object_info_cache

        result = self._request("/object_info", timeout=60)
        if "error" not in result:
            self._object_info_cache = result
            self._object_info_time = now
        return result

    def registered_node_types(self) -> set[str]:
        """Return set of all registered node class_types."""
        info = self.object_info()
        if "error" in info:
            return set()
        return set(info.keys())

    # ── Workflow execution ────────────────────────────────────────────

    def queue_prompt(self, workflow: dict, client_id: str = "") -> dict:
        """Queue a workflow (API format). Returns prompt_id or error."""
        payload = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        return self._request("/prompt", method="POST", data=payload)

    def get_queue(self) -> dict:
        return self._request("/queue")

    def get_history(self, prompt_id: str = "") -> dict:
        endpoint = f"/history/{prompt_id}" if prompt_id else "/history"
        return self._request(endpoint)

    def interrupt(self) -> dict:
        return self._request("/interrupt", method="POST")

    # ── Wait for completion ───────────────────────────────────────────

    def wait_for_prompt(
        self, prompt_id: str, timeout: int = 600, poll_interval: float = 2.0
    ) -> dict:
        """Poll until a prompt completes or fails. Returns history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self.get_history(prompt_id)
            if "error" in history:
                time.sleep(poll_interval)
                continue

            entry = history.get(prompt_id, {})
            status = entry.get("status", {})
            status_str = status.get("status_str", "")

            if status_str == "success":
                return {"status": "success", "outputs": entry.get("outputs", {})}
            elif status_str == "error":
                messages = status.get("messages", [])
                return {"status": "error", "messages": messages, "entry": entry}

            time.sleep(poll_interval)

        return {"status": "timeout", "error": f"Timed out after {timeout}s"}
