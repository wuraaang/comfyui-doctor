"""LLM client for comfyui-doctor — Claude-powered error diagnosis and fixes.

This module provides:
- Dataclasses for LLM request/response
- LLMClient that talks to an Anthropic-compatible proxy
- 6 escalation methods: find_node_package, find_model, diagnose_error,
  diagnose_quality, fix_oom, and generic ask
- System prompt specialized for ComfyUI debugging
- Adaptive retry with failure history
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from .doctor import FixAction
from ..knowledge.node_map import NodePackage
from ..knowledge.model_map import ModelInfo


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


@dataclass
class WorkflowMutation:
    """A mutation to apply to a workflow."""
    node_id: str
    field: str  # e.g. "inputs.steps"
    value: Any
    reason: str = ""


# ── LLM Client ──────────────────────────────────────────────────────────

class LLMClient:
    """Client for Claude via an Anthropic-compatible proxy.

    Provides 6 escalation methods for Doctor:
    1. find_node_package — unknown node type
    2. find_model — unknown model
    3. diagnose_error — unrecognized runtime error
    4. diagnose_quality — blank/bad image
    5. fix_oom — CUDA OOM
    6. diagnose_error (generic) — version/API mismatch
    """

    def __init__(self, proxy_url: str, auth_token: str, model: str = "claude-sonnet-4-20250514"):
        self.proxy_url = proxy_url.rstrip("/")
        self.auth_token = auth_token
        self.model = model
        self._client = None
        self._failure_history: list[dict] = []  # Adaptive retry context
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
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{self.proxy_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    return resp.status < 500
            except Exception:
                return False

    # ── Core API call ────────────────────────────────────────────────

    def _call(self, messages: list[dict], system: str = "", max_tokens: int = 2048) -> LLMResponse:
        """Make a raw API call to Claude. Returns LLMResponse."""
        if self._client is None:
            return LLMResponse(error="Anthropic SDK not available")

        system_prompt = system or self._build_system_prompt()
        start = time.time()

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )
            latency = (time.time() - start) * 1000
            text = response.content[0].text if response.content else ""
            parsed = self._parse_response(text)
            return LLMResponse(
                text=text,
                fix_actions=parsed.get("fix_actions", []),
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("diagnosis", ""),
                raw_response=response,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return LLMResponse(error=str(e), latency_ms=latency)

    def _parse_response(self, text: str) -> dict:
        """Parse LLM response text into structured data.

        Extracts JSON blocks from the response and converts fix_actions
        into FixAction dataclasses.
        """
        # Try to find JSON block in response
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if not json_match:
            # Try bare JSON
            json_match = re.search(r'\{[^{}]*"fix_actions"[^{}]*\}', text, re.DOTALL)
            if not json_match:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if not json_match:
            return {"diagnosis": text, "confidence": 0.3}

        try:
            data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))
        except json.JSONDecodeError:
            return {"diagnosis": text, "confidence": 0.3}

        # Convert raw fix_actions dicts to FixAction dataclasses
        fix_actions = []
        for fa in data.get("fix_actions", []):
            fix_actions.append(FixAction(
                description=fa.get("description", "LLM suggested fix"),
                commands=fa.get("commands", []),
                category=fa.get("category", "config_change"),
                auto=True,
            ))
        data["fix_actions"] = fix_actions

        # Extract workflow mutations
        mutations = []
        for wm in data.get("workflow_mutations", []):
            mutations.append(WorkflowMutation(
                node_id=str(wm.get("node_id", "")),
                field=wm.get("field", ""),
                value=wm.get("value"),
                reason=wm.get("reason", ""),
            ))
        data["workflow_mutations"] = mutations

        return data

    # ── System prompt ────────────────────────────────────────────────

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

Always respond with a JSON block:
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

    # ── Context builder ──────────────────────────────────────────────

    def _build_context(self, workflow: dict, system_info: Optional[dict] = None,
                       installed_nodes: Optional[set] = None,
                       installed_models: Optional[list] = None,
                       failure_history: Optional[list] = None) -> dict:
        """Build minimal context for the LLM from a workflow and system info.

        Phase 4 enrichment: includes installed nodes, models on disk,
        and history of failed fix attempts.
        """
        node_types = {}
        model_refs = []
        total_nodes = len(workflow)

        for node_id, node_data in workflow.items():
            ct = node_data.get("class_type", "Unknown")
            node_types[ct] = node_types.get(ct, 0) + 1

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
            context["system"] = {
                k: v for k, v in system_info.items()
                if k in ("os", "python_version", "devices", "comfyui")
            }

        # Phase 4: Enriched context
        if installed_nodes:
            context["installed_node_types"] = len(installed_nodes)

        if installed_models:
            context["installed_models"] = installed_models[:20]  # Cap at 20

        # Adaptive retry: include failure history
        history = failure_history or self._failure_history
        if history:
            context["previous_attempts"] = history[-5:]  # Last 5 attempts

        return context

    def _record_failure(self, error_text: str, fix_attempted: str):
        """Record a failed fix for adaptive retry context."""
        self._failure_history.append({
            "error": error_text[:200],
            "fix_attempted": fix_attempted,
            "timestamp": time.time(),
        })

    # ── Escalation Point 1: Unknown Node Type ────────────────────────

    def find_node_package(self, class_type: str, context: Optional[dict] = None) -> Optional[NodePackage]:
        """Ask Claude to find the package for an unknown node type.

        Returns NodePackage if found, None otherwise.
        """
        msg = (
            f"I need to find the ComfyUI custom node package that provides the node type: `{class_type}`\n\n"
            f"Please respond with a JSON block:\n"
            f'```json\n{{"package_name": "RepoName", "repo_url": "https://github.com/user/repo", '
            f'"pip_deps": [], "confidence": 0.0-1.0}}\n```\n\n'
        )
        if context:
            msg += f"Workflow context: {json.dumps(context, indent=2)[:500]}"

        resp = self._call([{"role": "user", "content": msg}], max_tokens=512)

        if resp.error:
            return None

        # Parse the response for package info
        try:
            json_match = re.search(r'\{[^{}]*"package_name"[^{}]*\}', resp.text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```json\s*\n(.*?)\n```', resp.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))
                if data.get("confidence", 0) >= 0.5 and data.get("repo_url"):
                    return NodePackage(
                        package_name=data["package_name"],
                        repo_url=data["repo_url"],
                        node_types=[class_type],
                        pip_deps=data.get("pip_deps", []),
                        description=f"[LLM] {data.get('description', '')}",
                    )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    # ── Escalation Point 2: Unknown Model ────────────────────────────

    def find_model(self, filename: str, node_type: str = "", model_folder: str = "") -> Optional[ModelInfo]:
        """Ask Claude to find download info for an unknown model.

        Returns ModelInfo if found, None otherwise.
        """
        msg = (
            f"I need to find the download URL for a ComfyUI model:\n"
            f"- Filename: `{filename}`\n"
            f"- Used by node type: `{node_type}`\n"
            f"- Expected folder: `models/{model_folder}/`\n\n"
            f"Please respond with a JSON block:\n"
            f'```json\n{{"url": "https://...", "size_gb": 0.0, "model_folder": "{model_folder}", '
            f'"hf_token_required": false, "confidence": 0.0-1.0}}\n```\n\n'
            f"Check HuggingFace and CivitAI. If the filename is a variant (e.g. with _fp8, _0.9vae), "
            f"suggest the closest match."
        )

        resp = self._call([{"role": "user", "content": msg}], max_tokens=512)

        if resp.error:
            return None

        try:
            json_match = re.search(r'\{[^{}]*"url"[^{}]*\}', resp.text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```json\s*\n(.*?)\n```', resp.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))
                if data.get("confidence", 0) >= 0.5 and data.get("url"):
                    return ModelInfo(
                        url=data["url"],
                        filename=filename,
                        size_gb=data.get("size_gb", 0),
                        model_folder=data.get("model_folder", model_folder),
                        description=f"[LLM] Found by Claude",
                        hf_token_required=data.get("hf_token_required", False),
                    )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    # ── Escalation Point 3: Unrecognized Runtime Error ───────────────

    def diagnose_error(self, error_text: str, workflow: Optional[dict] = None,
                       system_info: Optional[dict] = None) -> LLMResponse:
        """Ask Claude to diagnose an unrecognized error.

        Returns LLMResponse with fix_actions.
        """
        context = self._build_context(workflow or {}, system_info) if workflow else {}

        msg = (
            f"ComfyUI workflow error that I couldn't match in my knowledge base:\n\n"
            f"```\n{error_text[:1500]}\n```\n\n"
        )
        if context:
            msg += f"Workflow context:\n```json\n{json.dumps(context, indent=2)[:800]}\n```\n\n"
        msg += "Diagnose this error and provide fix_actions with shell commands to fix it."

        resp = self._call([{"role": "user", "content": msg}])

        if resp.error:
            return resp

        # Record for adaptive retry
        if resp.fix_actions:
            for fa in resp.fix_actions:
                self._record_failure(error_text[:200], fa.description)

        return resp

    # ── Escalation Point 4: Quality Issue (blank image) ──────────────

    def diagnose_quality(self, quality_data: dict, workflow: Optional[dict] = None) -> LLMResponse:
        """Ask Claude to diagnose blank/bad output images.

        Returns LLMResponse with workflow_mutations to fix quality.
        """
        msg = (
            f"ComfyUI workflow produced a blank/solid image:\n\n"
            f"Quality data: {json.dumps(quality_data, indent=2)}\n\n"
        )
        if workflow:
            context = self._build_context(workflow)
            msg += f"Workflow context:\n```json\n{json.dumps(context, indent=2)[:800]}\n```\n\n"

        msg += (
            "Common causes of blank images:\n"
            "- CFG too high or too low\n"
            "- Wrong VAE (e.g. SDXL workflow with SD1.5 VAE)\n"
            "- Corrupted model file\n"
            "- Steps=0 or denoise=0\n"
            "- Wrong latent dimensions\n\n"
            "Suggest workflow_mutations to fix the quality issue."
        )

        return self._call([{"role": "user", "content": msg}])

    # ── Escalation Point 5: CUDA OOM ────────────────────────────────

    def fix_oom(self, workflow: dict, vram_gb: float = 0, error_text: str = "") -> LLMResponse:
        """Ask Claude to reduce VRAM usage of a workflow.

        Returns LLMResponse with workflow_mutations (lower resolution,
        VAEDecodeTiled, FP8, etc.).
        """
        context = self._build_context(workflow)

        msg = (
            f"ComfyUI ran out of GPU memory (CUDA OOM):\n\n"
            f"```\n{error_text[:500]}\n```\n\n"
            f"GPU VRAM: {vram_gb:.1f} GB\n"
            f"Workflow context:\n```json\n{json.dumps(context, indent=2)[:800]}\n```\n\n"
            f"Suggest workflow_mutations to reduce VRAM usage. Common fixes:\n"
            f"- Reduce resolution (width/height in EmptyLatentImage)\n"
            f"- Replace VAEDecode with VAEDecodeTiled\n"
            f"- Reduce batch_size to 1\n"
            f"- Lower steps\n"
            f"- Use FP8 checkpoint if available\n\n"
            f"Only suggest mutations that will fix the OOM without breaking the workflow."
        )

        return self._call([{"role": "user", "content": msg}])

    # ── Apply workflow mutations ─────────────────────────────────────

    @staticmethod
    def apply_mutations(workflow: dict, mutations: list[WorkflowMutation]) -> tuple[dict, list[str]]:
        """Apply workflow mutations from LLM response.

        Returns (mutated_workflow, list_of_changes_applied).
        """
        changes = []
        for mutation in mutations:
            node = workflow.get(mutation.node_id)
            if not node:
                continue

            # Parse field path (e.g. "inputs.steps" → node["inputs"]["steps"])
            parts = mutation.field.split(".")
            target = node
            for part in parts[:-1]:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    target = None
                    break

            if target is not None and isinstance(target, dict):
                key = parts[-1]
                old_val = target.get(key)
                target[key] = mutation.value
                desc = f"Node {mutation.node_id}: {mutation.field} = {mutation.value}"
                if mutation.reason:
                    desc += f" ({mutation.reason})"
                if old_val is not None:
                    desc += f" [was: {old_val}]"
                changes.append(desc)

        return workflow, changes
