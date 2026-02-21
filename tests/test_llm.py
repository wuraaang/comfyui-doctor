"""Tests for LLM module — dataclasses, LLMClient init, Doctor integration."""

import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comfyui_doctor.core.llm import LLMRequest, LLMResponse, LLMClient
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


def test_llm_client_init():
    """Test LLMClient initialization (no real API call)."""
    client = LLMClient(
        proxy_url="https://fake-proxy.example.com",
        auth_token="sk-fake-token-for-testing",
    )
    assert client.proxy_url == "https://fake-proxy.example.com"
    assert client.auth_token == "sk-fake-token-for-testing"
    assert client.model == "claude-sonnet-4-20250514"
    # Client may or may not be initialized depending on anthropic install
    print("✅ test_llm_client_init passed")


def test_llm_client_is_available_no_server():
    """Test is_available returns False when proxy is unreachable."""
    client = LLMClient(
        proxy_url="http://localhost:1",  # Port 1 won't be listening
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
    assert len(prompt) > 500  # Should be a substantial prompt
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
    assert context["node_types"]["CheckpointLoaderSimple"] == 1
    # Should detect model reference
    assert len(context["model_references"]) >= 1
    assert any("sd_xl_base" in ref["filename"] for ref in context["model_references"])
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
    # All existing functionality should work without LLM
    assert doctor.comfyui_path == "/tmp/fake-comfyui"
    assert doctor.max_retries == 3
    assert doctor.auto_fix is True
    print("✅ test_doctor_without_llm_client passed")


if __name__ == "__main__":
    test_llm_request_dataclass()
    test_llm_response_dataclass()
    test_llm_client_init()
    test_llm_client_is_available_no_server()
    test_llm_client_build_system_prompt()
    test_llm_client_build_context()
    test_llm_client_build_context_with_system_info()
    test_doctor_with_llm_client()
    test_doctor_without_llm_client()
    print("\n🎉 All LLM tests passed!")
