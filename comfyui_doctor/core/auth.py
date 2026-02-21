"""Authentication and token management for comfyui-doctor.

Handles:
- Config directory (~/.comfyui-doctor/)
- Auth config for LLM proxy (token, proxy URL)
- API tokens for HuggingFace and CivitAI (secure storage)
- Login flow for subscription-based LLM proxy
"""

import json
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()

CONFIG_DIR = Path.home() / ".comfyui-doctor"
AUTH_FILE = CONFIG_DIR / "auth.json"
TOKENS_FILE = CONFIG_DIR / "tokens.json"


def ensure_config_dir() -> Path:
    """Create the config directory if it doesn't exist. Returns the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


# ── Auth Config (LLM proxy) ─────────────────────────────────────────

@dataclass
class AuthConfig:
    """Authentication configuration for the LLM proxy."""
    proxy_url: str = ""
    token: str = ""
    expires_at: Optional[float] = None  # Unix timestamp
    user_email: str = ""


class AuthManager:
    """Manage authentication for the LLM proxy."""

    def __init__(self):
        self._config: Optional[AuthConfig] = None

    def load_config(self) -> AuthConfig:
        """Load auth config from disk."""
        if self._config:
            return self._config
        if AUTH_FILE.exists():
            try:
                data = json.loads(AUTH_FILE.read_text())
                self._config = AuthConfig(**data)
                return self._config
            except (json.JSONDecodeError, TypeError):
                pass
        self._config = AuthConfig()
        return self._config

    def save_config(self, config: AuthConfig):
        """Save auth config to disk."""
        ensure_config_dir()
        AUTH_FILE.write_text(json.dumps({
            "proxy_url": config.proxy_url,
            "token": config.token,
            "expires_at": config.expires_at,
            "user_email": config.user_email,
        }, indent=2))
        # Restrict permissions (user-only read/write)
        try:
            os.chmod(AUTH_FILE, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        self._config = config

    def is_authenticated(self) -> bool:
        """Check if we have a valid, non-expired token."""
        config = self.load_config()
        if not config.token or not config.proxy_url:
            return False
        if config.expires_at and time.time() > config.expires_at:
            return False
        return True

    def login(self, email: str, proxy_url: str = "https://api.comfyui-doctor.com") -> bool:
        """Login flow: email → verification code → JWT.

        Returns True if login succeeded.
        """
        try:
            import urllib.request
            # Step 1: Request verification code
            payload = json.dumps({"email": email}).encode()
            req = urllib.request.Request(
                f"{proxy_url}/auth/login",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if "error" in data:
                    console.print(f"  [red]Login failed: {data['error']}[/red]")
                    return False

            # Step 2: Ask for verification code
            console.print(f"  📧 Verification code sent to [cyan]{email}[/cyan]")
            code = Prompt.ask("  Enter verification code")

            # Step 3: Verify code → get JWT
            payload = json.dumps({"email": email, "code": code}).encode()
            req = urllib.request.Request(
                f"{proxy_url}/auth/verify",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if "token" in data:
                    config = AuthConfig(
                        proxy_url=proxy_url,
                        token=data["token"],
                        expires_at=data.get("expires_at"),
                        user_email=email,
                    )
                    self.save_config(config)
                    console.print("  [green]Login successful![/green]")
                    return True
                else:
                    console.print(f"  [red]Verification failed: {data.get('error', 'unknown')}[/red]")
                    return False

        except Exception as e:
            console.print(f"  [red]Login error: {e}[/red]")
            return False

    def get_quota(self) -> Optional[dict]:
        """Check remaining LLM quota from the proxy."""
        config = self.load_config()
        if not config.token or not config.proxy_url:
            return None
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{config.proxy_url}/auth/quota",
                headers={
                    "Authorization": f"Bearer {config.token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


# ── Token Manager (HF/CivitAI API tokens) ───────────────────────────

class TokenManager:
    """Manage API tokens for HuggingFace, CivitAI, etc.

    Tokens are stored in ~/.comfyui-doctor/tokens.json with
    restricted permissions (600).
    """

    SERVICES = {
        "huggingface": {
            "name": "HuggingFace",
            "url": "https://huggingface.co/settings/tokens",
            "env_var": "HF_TOKEN",
            "description": "Required for gated models (FLUX, etc.)",
        },
        "civitai": {
            "name": "CivitAI",
            "url": "https://civitai.com/user/account",
            "env_var": "CIVITAI_API_KEY",
            "description": "Required to download from CivitAI",
        },
    }

    def __init__(self):
        self._tokens: Optional[dict] = None

    def load_tokens(self) -> dict:
        """Load tokens from disk. Also checks environment variables."""
        if self._tokens is not None:
            return self._tokens

        tokens = {}
        if TOKENS_FILE.exists():
            try:
                tokens = json.loads(TOKENS_FILE.read_text())
            except json.JSONDecodeError:
                pass

        # Also check env vars as fallback
        for service, info in self.SERVICES.items():
            if service not in tokens or not tokens[service]:
                env_val = os.environ.get(info["env_var"], "")
                if env_val:
                    tokens[service] = env_val

        self._tokens = tokens
        return tokens

    def save_tokens(self, tokens: dict):
        """Save tokens to disk with restricted permissions."""
        ensure_config_dir()
        TOKENS_FILE.write_text(json.dumps(tokens, indent=2))
        try:
            os.chmod(TOKENS_FILE, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        self._tokens = tokens

    def get_token(self, service: str) -> str:
        """Get token for a service. Returns empty string if not set."""
        tokens = self.load_tokens()
        return tokens.get(service, "")

    def set_token(self, service: str, token: str):
        """Set token for a service and save."""
        tokens = self.load_tokens()
        tokens[service] = token
        self.save_tokens(tokens)

    def delete_token(self, service: str):
        """Remove token for a service."""
        tokens = self.load_tokens()
        tokens.pop(service, None)
        self.save_tokens(tokens)

    def prompt_for_token(self, service: str, reason: str = "") -> str:
        """Interactive prompt asking user for a missing token.

        Shows service info, why it's needed, and where to get it.
        Returns the token entered, or empty string if skipped.
        """
        info = self.SERVICES.get(service, {})
        name = info.get("name", service)
        url = info.get("url", "")
        desc = info.get("description", "")

        console.print(f"\n  🔑 [bold yellow]{name} API token required[/bold yellow]")
        if reason:
            console.print(f"     Reason: {reason}")
        if desc:
            console.print(f"     {desc}")
        if url:
            console.print(f"     Get your token at: [cyan]{url}[/cyan]")

        token = Prompt.ask(f"  Enter your {name} token (or press Enter to skip)", default="")
        if token:
            self.set_token(service, token)
            console.print(f"  [green]Token saved to {TOKENS_FILE}[/green]")
        return token

    def has_required_token(self, service: str) -> bool:
        """Check if we have a token for this service."""
        return bool(self.get_token(service))

    def list_tokens(self) -> dict:
        """List all stored tokens (masked for display)."""
        tokens = self.load_tokens()
        masked = {}
        for service, token in tokens.items():
            if token and len(token) > 8:
                masked[service] = f"{token[:4]}...{token[-4:]}"
            elif token:
                masked[service] = "***"
            else:
                masked[service] = "(empty)"
        return masked
