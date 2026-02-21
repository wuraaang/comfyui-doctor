"""LLM client for comfyui-doctor — Claude-powered error diagnosis and fixes.

This module provides:
- Dataclasses for LLM request/response
- LLMClient that talks to an Anthropic-compatible proxy
- System prompt specialized for ComfyUI debugging
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Any

from .doctor import FixAction


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    """A request to send to the LLM."""
    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.2


@dataclass
class LLMResponse:
    """A response from the LLM."""
    text: str = ""
    fix_actions: list[FixAction] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0
    reasoning: str = ""
    raw_response: Optional[Any] = None
    error: str = ""
    latency_ms: float = 0.0


# ── LLM Client ──────────────────────────────────────────────────────────

class LLMClient:
    """Client for Claude via an Anthropic-compatible proxy.

    Usage:
        client = LLMClient(proxy_url="https://proxy.example.com", auth_token="sk-...")
        if client.is_available():
            # Use client for diagnosis
    """

    def __init__(self, proxy_url: str, auth_token: str, model: str = "claude-sonnet-4-20250514"):
        self.proxy_url = proxy_url.rstrip("/")
        self.auth_token = auth_token
        self.model = model
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Anthropic SDK client."""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(
                base_url=self.proxy_url,
                api_key=self.auth_token,
            )
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        """Check if the LLM proxy is reachable. Fast-fail 2s timeout."""
        if self._client is None:
            return False
        try:
            import httpx
            resp = httpx.get(
                f"{self.proxy_url}/v1/models",
                headers={"Authorization": f"Bearer {self.auth_token}"},
                timeout=2.0,
            )
            return resp.status_code < 500
        except Exception:
            # Fallback: try a minimal API call
            try:
                import urllib.request
                import urllib.error
                req = urllib.request.Request(
                    f"{self.proxy_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    return resp.status < 500
            except Exception:
                return False

    def _build_system_prompt(self) -> str:
        """Build the system prompt specialized for ComfyUI debugging."""
        return """You are a ComfyUI expert assistant embedded in comfyui-doctor, a CLI tool that auto-fixes ComfyUI workflows.

Your role: diagnose errors and suggest precise, executable fixes.

## ComfyUI Knowledge

- ComfyUI is a node-based Stable Diffusion UI where workflows are DAGs of nodes
- Each node has a class_type (e.g. "KSampler", "CheckpointLoaderSimple"), inputs, and outputs
- Workflows in API format are JSON: {"node_id": {"class_type": "...", "inputs": {...}}}
- Custom nodes are git repos installed in ComfyUI/custom_nodes/
- Models go in ComfyUI/models/{folder}/ (checkpoints, loras, vae, controlnet, etc.)
- Common model formats: .safetensors (preferred), .ckpt, .pt, .pth, .bin

## Common Error Patterns

- "No module named X" → pip install X
- "CUDA out of memory" → reduce resolution, use VAEDecodeTiled, enable fp8
- "Node 'X' not found" → install custom node package, or restart ComfyUI after install
- "prompt_outputs_failed_validation" → missing/invalid input values
- HTTP 400 from /prompt → workflow structure error or missing nodes

## Response Format

When diagnosing errors, respond with a JSON block:
```json
{
  "diagnosis": "Brief explanation of the problem",
  "confidence": 0.0-1.0,
  "fix_actions": [
    {
      "description": "What this fix does",
      "commands": ["shell command to run"],
      "category": "install_node|install_pip|download_model|config_change|workflow_mutation"
    }
  ],
  "workflow_mutations": [
    {
      "node_id": "3",
      "field": "inputs.steps",
      "value": 15,
      "reason": "Reduce steps to lower VRAM usage"
    }
  ]
}
```

## Rules

1. Only suggest commands you're confident will work
2. Prefer the simplest fix (pip install > rebuild from source)
3. For unknown nodes, suggest the most likely GitHub repo URL
4. For models, suggest HuggingFace or CivitAI URLs when possible
5. Never suggest destructive commands (rm -rf, etc.)
6. If unsure, set confidence < 0.5 and explain your uncertainty"""

    def _build_context(self, workflow: dict, system_info: Optional[dict] = None) -> dict:
        """Build minimal context for the LLM from a workflow and system info.

        Returns a dict with summarized workflow info suitable for including
        in an LLM message.
        """
        # Summarize workflow: node types, connections, model references
        node_types = {}
        model_refs = []
        total_nodes = len(workflow)

        for node_id, node_data in workflow.items():
            ct = node_data.get("class_type", "Unknown")
            node_types[ct] = node_types.get(ct, 0) + 1

            # Extract model references from inputs
            inputs = node_data.get("inputs", {})
            for key, val in inputs.items():
                if isinstance(val, str) and (
                    val.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin"))
                    or "/" in val and any(val.endswith(ext) for ext in (".safetensors", ".ckpt", ".pt"))
                ):
                    model_refs.append({"node": ct, "input": key, "filename": val})

        context = {
            "total_nodes": total_nodes,
            "node_types": node_types,
            "model_references": model_refs,
        }

        if system_info:
            # Include relevant system info (GPU, VRAM, OS)
            context["system"] = {
                k: v for k, v in system_info.items()
                if k in ("os", "python_version", "devices", "comfyui")
            }

        return context
