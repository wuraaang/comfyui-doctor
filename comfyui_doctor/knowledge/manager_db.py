"""ComfyUI-Manager extension-node-map fallback.

When our local node_map.py doesn't know a node type, we fall back to
ComfyUI-Manager's crowd-sourced extension-node-map.json which maps
31,000+ node class_types to their git repos.

The map is downloaded once and cached for 24h.
"""

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


EXTENSION_NODE_MAP_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/"
    "main/extension-node-map.json"
)

CUSTOM_NODE_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/"
    "main/custom-node-list.json"
)

CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "comfyui-doctor",
)
CACHE_FILE = os.path.join(CACHE_DIR, "extension-node-map.json")
CACHE_TTL = 86400  # 24 hours


@dataclass
class ManagerNodeInfo:
    """Node info from ComfyUI-Manager's database."""
    class_type: str
    repo_url: str
    package_name: str  # Derived from repo URL


def _download_map() -> dict:
    """Download the extension-node-map from GitHub."""
    try:
        req = urllib.request.Request(EXTENSION_NODE_MAP_URL)
        req.add_header("User-Agent", "comfyui-doctor/0.1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        # Cache it
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
        
        return data
    except Exception as e:
        print(f"  ⚠️  Could not download ComfyUI-Manager node map: {e}")
        return {}


def _load_map() -> dict:
    """Load the extension-node-map, using cache if fresh."""
    # Check cache
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
            if time.time() - cached.get("timestamp", 0) < CACHE_TTL:
                return cached.get("data", {})
        except Exception:
            pass
    
    return _download_map()


def _repo_to_package_name(repo_url: str) -> str:
    """Extract package name from a git repo URL.
    
    https://github.com/user/ComfyUI-KJNodes → ComfyUI-KJNodes
    """
    # Remove trailing slashes and .git
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.rsplit("/", 1)[-1]


# ── Build reverse lookup: class_type → repo URL ──────────────────────

_REVERSE_MAP: Optional[dict[str, str]] = None


def _ensure_reverse_map() -> dict[str, str]:
    """Build reverse map: class_type → repo_url (lazy, cached)."""
    global _REVERSE_MAP
    if _REVERSE_MAP is not None:
        return _REVERSE_MAP
    
    raw = _load_map()
    _REVERSE_MAP = {}
    
    for repo_url, type_list in raw.items():
        # type_list is [[class_type1, class_type2, ...], ...]
        if isinstance(type_list, list) and type_list:
            types = type_list[0] if isinstance(type_list[0], list) else type_list
            for ct in types:
                if isinstance(ct, str):
                    _REVERSE_MAP[ct] = repo_url
    
    return _REVERSE_MAP


# ── Public API ────────────────────────────────────────────────────────

def manager_lookup(class_type: str) -> Optional[ManagerNodeInfo]:
    """Look up a node class_type in ComfyUI-Manager's database.
    
    Returns ManagerNodeInfo if found, None otherwise.
    This is the fallback when our local node_map doesn't know the type.
    """
    reverse = _ensure_reverse_map()
    repo_url = reverse.get(class_type)
    
    if not repo_url:
        return None
    
    return ManagerNodeInfo(
        class_type=class_type,
        repo_url=repo_url,
        package_name=_repo_to_package_name(repo_url),
    )


def manager_lookup_multiple(class_types: set[str]) -> dict[str, Optional[ManagerNodeInfo]]:
    """Look up multiple class_types. Returns {class_type: info_or_None}."""
    return {ct: manager_lookup(ct) for ct in class_types}


def get_map_stats() -> dict:
    """Get stats about the cached map."""
    reverse = _ensure_reverse_map()
    repos = set(reverse.values())
    return {
        "total_types": len(reverse),
        "total_repos": len(repos),
        "cache_file": CACHE_FILE,
        "cache_exists": os.path.isfile(CACHE_FILE),
    }
