"""Tests for LLM module — dataclasses, LLMClient, Doctor integration, escalation methods."""

import json
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comfyui_doctor.core.llm import LLMRequest, LLMResponse, LLMClient, WorkflowMutation
from comfyui_doctor.core.doctor import Doctor, FixAction


def test_llm_request_dataclass():
    """Test LLMRequest defaults and creation."""
    req = LLMRequest()
    assert req.messages == []
    assert req.system_prompt == ""
    assert req.max_tokens == 2048
    assert req.temperature == 0.2

    req2 = LLMRequest(
        messages=[{"role": "user", "content": "hello"}],
        system_prompt="You are helpful",
        max_tokens=1024,
        temperature=0.5,
    )
    assert len(req2.messages) == 1
    assert req2.max_tokens == 1024
    print("✅ test_llm_request_dataclass passed")


def test_llm_response_dataclass():
    """Test LLMResponse defaults and creation."""
    resp = LLMResponse()
    assert resp.text == ""
    assert resp.fix_actions == []
    assert resp.confidence == 0.0
    assert resp.error == ""

    resp2 = LLMResponse(
        text="Install the missing module",
        fix_actions=[FixAction(
            description="Install insightface",
            commands=["pip install insightface"],
            category="install_pip",
        )],
        confidence=0.9,
        reasoning="The error clearly says module not found",
    )
    assert len(resp2.fix_actions) == 1
    assert resp2.fix_actions[0].commands[0] == "pip install insightface"
    assert resp2.confidence == 0.9
    print("✅ test_llm_response_dataclass passed")


def test_workflow_mutation_dataclass():
    """Test WorkflowMutation dataclass."""
    m = WorkflowMutation(node_id="3", field="inputs.steps", value=15, reason="Reduce VRAM")
    assert m.node_id == "3"
    assert m.field == "inputs.steps"
    assert m.value == 15
    assert m.reason == "Reduce VRAM"
    print("✅ test_workflow_mutation_dataclass passed")


def test_llm_client_init():
    """Test LLMClient initialization (no real API call)."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake-token-for-testing",
    )
    assert client.proxy_url == "https://fake-proxy.example.com"
    assert client.auth_token == "sk-fake-token-for-testing"
    assert client.model == "claude-sonnet-4-20250514"
    assert client._failure_history == []
    print("✅ test_llm_client_init passed")


def test_llm_client_is_available_no_server():
    """Test is_available returns False when proxy is unreachable."""
    client = LLMClient(
        proxy_url="http://localhost:1",
        auth_token="sk-fake",
    )
    assert client.is_available() is False
    print("✅ test_llm_client_is_available_no_server passed")


def test_llm_client_build_system_prompt():
    """Test system prompt generation."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    prompt = client._build_system_prompt()
    assert "ComfyUI" in prompt
    assert "fix_actions" in prompt
    assert "CUDA out of memory" in prompt
    assert len(prompt) > 500
    print("✅ test_llm_client_build_system_prompt passed")


def test_llm_client_build_context():
    """Test context building from workflow."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a photo of a cat", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": 20, "cfg": 7.0,
                "model": ["1", 0], "positive": ["2", 0],
            },
        },
    }
    context = client._build_context(workflow)
    assert context["total_nodes"] == 3
    assert context["node_types"]["KSampler"] == 1
    assert len(context["model_references"]) >= 1
    print(f"✅ test_llm_client_build_context passed (nodes={context['total_nodes']})")


def test_llm_client_build_context_with_system_info():
    """Test context building with system info."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    workflow = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    system_info = {
        "os": "Linux",
        "python_version": "3.11.0",
        "devices": [{"name": "RTX 4090", "vram_total": 25769803776}],
        "comfyui": "running",
        "irrelevant_key": "should be filtered",
    }
    context = client._build_context(workflow, system_info)
    assert "system" in context
    assert context["system"]["os"] == "Linux"
    assert "irrelevant_key" not in context["system"]
    print("✅ test_llm_client_build_context_with_system_info passed")


def test_llm_client_build_context_enriched():
    """Test Phase 4: enriched context with installed nodes and failure history."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    workflow = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}

    # Record some failures
    client._record_failure("ModuleNotFoundError: xyz", "pip install xyz")

    context = client._build_context(
        workflow,
        installed_nodes={"KSampler", "VAEDecode", "CLIPTextEncode"},
        installed_models=["sd_xl_base_1.0.safetensors"],
    )
    assert context["installed_node_types"] == 3
    assert "sd_xl_base_1.0.safetensors" in context["installed_models"]
    assert len(context["previous_attempts"]) == 1
    assert "xyz" in context["previous_attempts"][0]["error"]
    print("✅ test_llm_client_build_context_enriched passed")


def test_llm_client_parse_response():
    """Test parsing structured JSON from LLM response."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )

    # Test with JSON code block
    text = """Here's my analysis:
```json
{
  "diagnosis": "Missing insightface module",
  "confidence": 0.9,
  "fix_actions": [
    {
      "description": "Install insightface",
      "commands": ["pip install insightface"],
      "category": "install_pip"
    }
  ],
  "workflow_mutations": []
}
```
"""
    parsed = client._parse_response(text)
    assert parsed["diagnosis"] == "Missing insightface module"
    assert parsed["confidence"] == 0.9
    assert len(parsed["fix_actions"]) == 1
    assert isinstance(parsed["fix_actions"][0], FixAction)
    assert parsed["fix_actions"][0].commands == ["pip install insightface"]
    print("✅ test_llm_client_parse_response passed")


def test_llm_client_parse_response_no_json():
    """Test parsing when no JSON is present."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    parsed = client._parse_response("I don't know what to do")
    assert parsed["confidence"] == 0.3
    assert parsed["diagnosis"] == "I don't know what to do"
    print("✅ test_llm_client_parse_response_no_json passed")


def test_apply_mutations():
    """Test workflow mutation application."""
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"steps": 30, "cfg": 7.0, "seed": 42},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
    }
    mutations = [
        WorkflowMutation(node_id="3", field="inputs.steps", value=15, reason="Reduce VRAM"),
        WorkflowMutation(node_id="5", field="inputs.width", value=768, reason="Lower resolution"),
        WorkflowMutation(node_id="5", field="inputs.height", value=768),
        WorkflowMutation(node_id="99", field="inputs.x", value=1),  # Non-existent node
    ]
    result, changes = LLMClient.apply_mutations(workflow, mutations)
    assert result["3"]["inputs"]["steps"] == 15
    assert result["5"]["inputs"]["width"] == 768
    assert result["5"]["inputs"]["height"] == 768
    assert len(changes) == 3  # 3 applied, 1 skipped (node 99)
    print(f"✅ test_apply_mutations passed ({len(changes)} changes)")


def test_doctor_with_llm_client():
    """Test Doctor accepts llm_client parameter."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    doctor = Doctor(
        comfyui_url="http://localhost:1",
        comfyui_path="/tmp/fake-comfyui",
        llm_client=client,
    )
    assert doctor.llm_client is client
    assert doctor.llm_client.proxy_url == "https://fake-proxy.example.com"
    print("✅ test_doctor_with_llm_client passed")


def test_doctor_without_llm_client():
    """Test Doctor works without llm_client (backward compat)."""
    doctor = Doctor(
        comfyui_url="http://localhost:1",
        comfyui_path="/tmp/fake-comfyui",
    )
    assert doctor.llm_client is None
    assert doctor.comfyui_path == "/tmp/fake-comfyui"
    assert doctor.max_retries == 3
    assert doctor.auto_fix is True
    assert doctor.verbose is False
    print("✅ test_doctor_without_llm_client passed")


def test_doctor_report_llm_fields():
    """Test DoctorReport has LLM-related fields."""
    from comfyui_doctor.core.doctor import DoctorReport
    report = DoctorReport(workflow_path="test.json")
    assert report.llm_suggestions == []
    assert report.escalated_to_llm is False
    report.escalated_to_llm = True
    report.llm_suggestions.append("Found node package via LLM")
    assert len(report.llm_suggestions) == 1
    print("✅ test_doctor_report_llm_fields passed")


def test_doctor_has_token_manager():
    """Test Doctor auto-creates TokenManager."""
    doctor = Doctor(
        comfyui_url="http://localhost:1",
        comfyui_path="/tmp/fake-comfyui",
    )
    assert doctor.token_manager is not None
    print("✅ test_doctor_has_token_manager passed")


def test_record_failure_history():
    """Test failure history recording for adaptive retry."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake",
    )
    assert len(client._failure_history) == 0
    client._record_failure("Error A", "Fix A")
    client._record_failure("Error B", "Fix B")
    assert len(client._failure_history) == 2
    assert client._failure_history[0]["error"] == "Error A"
    assert client._failure_history[1]["fix_attempted"] == "Fix B"
    print("✅ test_record_failure_history passed")


if __name__ == "__main__":
    test_llm_request_dataclass()
    test_llm_response_dataclass()
    test_workflow_mutation_dataclass()
    test_llm_client_init()
    test_llm_client_is_available_no_server()
    test_llm_client_build_system_prompt()
    test_llm_client_build_context()
    test_llm_client_build_context_with_system_info()
    test_llm_client_build_context_enriched()
    test_llm_client_parse_response()
    test_llm_client_parse_response_no_json()
    test_apply_mutations()
    test_doctor_with_llm_client()
    test_doctor_without_llm_client()
    test_doctor_report_llm_fields()
    test_doctor_has_token_manager()
    test_record_failure_history()
    print("\n🎉 All LLM tests passed!")
