"""Tests for MCP client — connection, fallback, context enrichment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comfyui_doctor.core.mcp_client import MCPConnection


def test_mcp_connection_init():
    """Test MCPConnection defaults."""
    mcp = MCPConnection()
    assert mcp.url == "http://localhost:8188"
    assert mcp.is_connected() is False
    assert mcp.has_comfy_pilot() is False
    assert mcp.list_tools() == []
    print("✅ test_mcp_connection_init passed")


def test_mcp_connection_custom_url():
    """Test MCPConnection with custom URL."""
    mcp = MCPConnection(url="http://myhost:9999/")
    assert mcp.url == "http://myhost:9999"  # Trailing slash stripped
    print("✅ test_mcp_connection_custom_url passed")


def test_mcp_connection_legacy_mcp_url():
    """Test MCPConnection strips /mcp suffix for backward compat."""
    mcp = MCPConnection(url="http://localhost:8189/mcp")
    assert mcp.url == "http://localhost:8189"
    print("✅ test_mcp_connection_legacy_mcp_url passed")


def test_mcp_connect_unreachable():
    """Test connect fails gracefully when server is unreachable."""
    mcp = MCPConnection(url="http://localhost:1")
    result = mcp.connect()
    assert result is False
    assert mcp.is_connected() is False
    print("✅ test_mcp_connect_unreachable passed")


def test_mcp_request_not_connected():
    """Test requests return error when not connected."""
    mcp = MCPConnection()
    result = mcp._request("/system_stats")
    assert "error" in result
    assert result["error"] == "Not connected"
    print("✅ test_mcp_request_not_connected passed")


def test_mcp_enrich_context_not_connected():
    """Test enrich_context returns empty when not connected."""
    mcp = MCPConnection()
    context = mcp.enrich_context()
    assert context == {}
    print("✅ test_mcp_enrich_context_not_connected passed")


def test_mcp_comfy_pilot_methods_without_pilot():
    """Test Comfy-Pilot methods return error when pilot not installed."""
    mcp = MCPConnection()
    mcp._connected = True  # Simulate connected to ComfyUI but no pilot
    mcp._has_comfy_pilot = False
    assert "error" in mcp.get_workflow()
    assert "error" in mcp.send_graph_command("test")
    assert "error" in mcp.install_custom_node("url")
    assert "error" in mcp.download_model("url", "file", "folder")
    print("✅ test_mcp_comfy_pilot_methods_without_pilot passed")


def test_mcp_wrappers_exist():
    """Test that all wrapper methods exist."""
    mcp = MCPConnection()
    assert callable(mcp.get_system_status)
    assert callable(mcp.get_node_info)
    assert callable(mcp.search_custom_nodes)
    assert callable(mcp.view_image)
    assert callable(mcp.install_custom_node)
    assert callable(mcp.download_model)
    assert callable(mcp.get_workflow)
    assert callable(mcp.get_queue)
    assert callable(mcp.get_history)
    print("✅ test_mcp_wrappers_exist passed")


if __name__ == "__main__":
    test_mcp_connection_init()
    test_mcp_connection_custom_url()
    test_mcp_connection_legacy_mcp_url()
    test_mcp_connect_unreachable()
    test_mcp_request_not_connected()
    test_mcp_enrich_context_not_connected()
    test_mcp_comfy_pilot_methods_without_pilot()
    test_mcp_wrappers_exist()
    print("\n🎉 All MCP tests passed!")
