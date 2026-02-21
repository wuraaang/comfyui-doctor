"""Authentication and configuration management for comfyui-doctor.

Handles:
- Config directory (~/.comfyui-doctor/)
- Auth config for LLM proxy (token, proxy URL)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".comfyui-doctor"


def ensure_config_dir() -> Path:
    """Create the config directory if it doesn't exist. Returns the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


@dataclass
class AuthConfig:
    """Authentication configuration for the LLM proxy."""
    proxy_url: str = ""
    token: str = ""
    expires_at: Optional[float] = None  # Unix timestamp
    user_email: str = ""
