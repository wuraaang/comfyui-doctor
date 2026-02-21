"""Tests for auth module — AuthConfig, AuthManager, TokenManager."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comfyui_doctor.core.auth import (
    AuthConfig,
    AuthManager,
    TokenManager,
    ensure_config_dir,
    CONFIG_DIR,
)


def test_auth_config_defaults():
    """Test AuthConfig default values."""
    config = AuthConfig()
    assert config.proxy_url == ""
    assert config.token == ""
    assert config.expires_at is None
    assert config.user_email == ""
    print("✅ test_auth_config_defaults passed")


def test_auth_config_creation():
    """Test AuthConfig with values."""
    config = AuthConfig(
        proxy_url="https://proxy.example.com",
        token="sk-test-token",
        expires_at=1700000000.0,
        user_email="test@example.com",
    )
    assert config.proxy_url == "https://proxy.example.com"
    assert config.token == "sk-test-token"
    assert config.expires_at == 1700000000.0
    print("✅ test_auth_config_creation passed")


def test_ensure_config_dir():
    """Test config directory creation."""
    path = ensure_config_dir()
    assert path == CONFIG_DIR
    assert path.is_dir()
    print("✅ test_ensure_config_dir passed")


def test_auth_manager_save_load():
    """Test saving and loading auth config."""
    auth = AuthManager()
    config = AuthConfig(
        proxy_url="https://test.proxy.com",
        token="sk-test-123",
        expires_at=9999999999.0,
        user_email="user@test.com",
    )
    auth.save_config(config)

    # Load fresh
    auth2 = AuthManager()
    loaded = auth2.load_config()
    assert loaded.proxy_url == "https://test.proxy.com"
    assert loaded.token == "sk-test-123"
    assert loaded.user_email == "user@test.com"
    print("✅ test_auth_manager_save_load passed")


def test_auth_manager_is_authenticated():
    """Test authentication check."""
    auth = AuthManager()

    # Save valid config
    config = AuthConfig(
        proxy_url="https://proxy.com",
        token="sk-valid",
        expires_at=9999999999.0,
    )
    auth.save_config(config)
    assert auth.is_authenticated() is True

    # Save expired config
    config_expired = AuthConfig(
        proxy_url="https://proxy.com",
        token="sk-expired",
        expires_at=1.0,  # Way in the past
    )
    auth.save_config(config_expired)
    auth._config = None  # Force reload
    assert auth.is_authenticated() is False

    # Empty token
    config_empty = AuthConfig(proxy_url="https://proxy.com", token="")
    auth.save_config(config_empty)
    auth._config = None
    assert auth.is_authenticated() is False
    print("✅ test_auth_manager_is_authenticated passed")


def test_token_manager_set_get():
    """Test TokenManager set/get/delete."""
    tm = TokenManager()
    tm._tokens = None  # Reset cache

    tm.set_token("huggingface", "hf_test_token_123")
    assert tm.get_token("huggingface") == "hf_test_token_123"

    tm.set_token("civitai", "civ_test_456")
    assert tm.get_token("civitai") == "civ_test_456"

    # Unknown service returns empty
    assert tm.get_token("unknown_service") == ""

    # Delete
    tm.delete_token("huggingface")
    assert tm.get_token("huggingface") == ""
    print("✅ test_token_manager_set_get passed")


def test_token_manager_list_masked():
    """Test token masking for display."""
    tm = TokenManager()
    tm._tokens = None
    tm.set_token("huggingface", "hf_abcdefghijklmnop")
    tm.set_token("civitai", "short")

    masked = tm.list_tokens()
    assert "hf_a" in masked["huggingface"]  # First 4 chars
    assert "mnop" in masked["huggingface"]  # Last 4 chars
    assert masked["civitai"] == "***"  # Too short to mask
    print("✅ test_token_manager_list_masked passed")


def test_token_manager_env_fallback():
    """Test tokens fall back to environment variables when file has no value."""
    tm = TokenManager()
    tm._tokens = None
    # Clear any existing tokens for this test
    tm.save_tokens({})
    tm._tokens = None  # Reset cache after clear

    with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
        tokens = tm.load_tokens()
        assert tokens.get("huggingface") == "hf_from_env"
    print("✅ test_token_manager_env_fallback passed")


def test_token_manager_has_required():
    """Test has_required_token check."""
    tm = TokenManager()
    tm._tokens = None
    # Start with clean slate
    tm.save_tokens({})
    tm._tokens = None
    tm.set_token("huggingface", "hf_test")
    assert tm.has_required_token("huggingface") is True
    assert tm.has_required_token("civitai") is False
    print("✅ test_token_manager_has_required passed")


def test_token_manager_services():
    """Test services registry."""
    tm = TokenManager()
    assert "huggingface" in tm.SERVICES
    assert "civitai" in tm.SERVICES
    assert "url" in tm.SERVICES["huggingface"]
    assert "env_var" in tm.SERVICES["huggingface"]
    print("✅ test_token_manager_services passed")


if __name__ == "__main__":
    test_auth_config_defaults()
    test_auth_config_creation()
    test_ensure_config_dir()
    test_auth_manager_save_load()
    test_auth_manager_is_authenticated()
    test_token_manager_set_get()
    test_token_manager_list_masked()
    test_token_manager_env_fallback()
    test_token_manager_has_required()
    test_token_manager_services()
    print("\n🎉 All auth tests passed!")
