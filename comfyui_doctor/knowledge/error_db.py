"""Error knowledge base — maps error patterns to fixes.

Each entry: (regex_pattern, category, fix_function_or_instructions)
"""

import re
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class ErrorMatch:
    """A matched error with its fix."""
    category: str
    pattern_name: str
    description: str
    fix_commands: list[str]  # Shell commands to run
    fix_description: str  # Human-readable fix
    extracted: dict  # Regex groups extracted from error


@dataclass
class ErrorPattern:
    """An error pattern with its fix recipe."""
    name: str
    category: str  # missing_module, missing_node, cuda_oom, model_not_found, etc.
    regex: str
    description: str
    fix_template: list[str]  # Commands with {placeholders}
    fix_description: str


# ── Error Pattern Database ────────────────────────────────────────────

ERROR_PATTERNS: list[ErrorPattern] = [
    # ━━━━ MISSING PYTHON MODULES ━━━━
    ErrorPattern(
        name="missing_module_generic",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        description="Python module not installed",
        fix_template=["pip install {module}"],
        fix_description="Install missing Python module: {module}",
    ),

    # ── Specific module fixes (override generic) ──
    ErrorPattern(
        name="missing_insightface",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]?(insightface)['\"]?",
        description="InsightFace not installed (needed for face detection/swap)",
        fix_template=["pip install insightface onnxruntime-gpu"],
        fix_description="Install insightface + onnxruntime-gpu",
    ),
    ErrorPattern(
        name="missing_onnxruntime",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]onnxruntime['\"]",
        description="ONNX Runtime not installed",
        fix_template=["pip install onnxruntime-gpu"],
        fix_description="Install onnxruntime-gpu",
    ),
    ErrorPattern(
        name="missing_segment_anything",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]segment_anything['\"]",
        description="Segment Anything not installed",
        fix_template=["pip install segment-anything"],
        fix_description="Install segment-anything",
    ),
    ErrorPattern(
        name="missing_mediapipe",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]mediapipe['\"]",
        description="MediaPipe not installed",
        fix_template=["pip install mediapipe"],
        fix_description="Install mediapipe",
    ),
    ErrorPattern(
        name="missing_ultralytics",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]ultralytics['\"]",
        description="Ultralytics (YOLO) not installed",
        fix_template=["pip install ultralytics"],
        fix_description="Install ultralytics",
    ),
    ErrorPattern(
        name="missing_mmcv",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]mmcv['\"]",
        description="MMCV not installed",
        fix_template=["pip install mmcv>=2.0.0"],
        fix_description="Install mmcv",
    ),
    ErrorPattern(
        name="missing_kornia",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]kornia['\"]",
        description="Kornia not installed",
        fix_template=["pip install kornia"],
        fix_description="Install kornia",
    ),
    ErrorPattern(
        name="missing_accelerate",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]accelerate['\"]",
        description="HuggingFace Accelerate not installed",
        fix_template=["pip install accelerate"],
        fix_description="Install accelerate",
    ),
    ErrorPattern(
        name="missing_diffusers",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]diffusers['\"]",
        description="HuggingFace Diffusers not installed",
        fix_template=["pip install diffusers"],
        fix_description="Install diffusers",
    ),
    ErrorPattern(
        name="missing_transformers",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]transformers['\"]",
        description="HuggingFace Transformers not installed",
        fix_template=["pip install transformers"],
        fix_description="Install transformers",
    ),
    ErrorPattern(
        name="missing_cv2",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]cv2['\"]",
        description="OpenCV not installed",
        fix_template=["pip install opencv-python"],
        fix_description="Install opencv-python",
    ),
    ErrorPattern(
        name="missing_scipy",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]scipy['\"]",
        description="SciPy not installed",
        fix_template=["pip install scipy"],
        fix_description="Install scipy",
    ),
    ErrorPattern(
        name="missing_torchaudio",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]torchaudio['\"]",
        description="torchaudio not installed",
        fix_template=["pip install torchaudio"],
        fix_description="Install torchaudio",
    ),
    ErrorPattern(
        name="missing_torchvision",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]torchvision['\"]",
        description="torchvision not installed",
        fix_template=["pip install torchvision"],
        fix_description="Install torchvision",
    ),
    ErrorPattern(
        name="missing_timm",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]timm['\"]",
        description="PyTorch Image Models not installed",
        fix_template=["pip install timm"],
        fix_description="Install timm",
    ),
    ErrorPattern(
        name="missing_einops",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]einops['\"]",
        description="einops not installed",
        fix_template=["pip install einops"],
        fix_description="Install einops",
    ),
    ErrorPattern(
        name="missing_safetensors",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]safetensors['\"]",
        description="safetensors not installed",
        fix_template=["pip install safetensors"],
        fix_description="Install safetensors",
    ),
    ErrorPattern(
        name="missing_huggingface_hub",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]huggingface_hub['\"]",
        description="huggingface_hub not installed",
        fix_template=["pip install huggingface-hub"],
        fix_description="Install huggingface-hub",
    ),
    ErrorPattern(
        name="missing_ftfy",
        category="missing_module",
        regex=r"ModuleNotFoundError: No module named ['\"]ftfy['\"]",
        description="ftfy not installed",
        fix_template=["pip install ftfy"],
        fix_description="Install ftfy",
    ),

    # ━━━━ CUDA / GPU ERRORS ━━━━
    ErrorPattern(
        name="cuda_oom",
        category="cuda_oom",
        regex=r"CUDA out of memory\. Tried to allocate ([\d.]+ \w+)",
        description="GPU out of VRAM",
        fix_template=[],  # Handled programmatically
        fix_description="Reduce memory usage: lower resolution, enable tiling, use FP8",
    ),
    ErrorPattern(
        name="cuda_oom_allocation",
        category="cuda_oom",
        regex=r"Allocation on device \d+ would exceed allowed memory.*?Currently allocated\s*:\s*([\d.]+ \w+).*?Device limit\s*:\s*([\d.]+ \w+)",
        description="GPU allocation would exceed VRAM limit",
        fix_template=[],
        fix_description="Reduce memory usage",
    ),

    # ━━━━ MODEL NOT FOUND ━━━━
    ErrorPattern(
        name="checkpoint_not_found",
        category="model_not_found",
        regex=r"(?:FileNotFoundError|Error).*?(?:checkpoint|model).*?['\"]([^'\"]+\.(?:safetensors|ckpt|pt|pth|bin))['\"]",
        description="Model file not found",
        fix_template=[],  # Handled by model resolver
        fix_description="Download missing model: {filename}",
    ),
    ErrorPattern(
        name="model_path_not_found",
        category="model_not_found",
        regex=r"FileNotFoundError.*?(?:models|checkpoints|loras|vae|controlnet)/([^\s'\"]+)",
        description="Model file not found in models directory",
        fix_template=[],
        fix_description="Download missing model from path reference",
    ),

    # ━━━━ MISSING NODE TYPE ━━━━
    ErrorPattern(
        name="node_not_registered",
        category="missing_node",
        regex=r"['\"](\w+)['\"] is not a (?:valid|registered) node type",
        description="Custom node type not installed",
        fix_template=[],  # Handled by node resolver
        fix_description="Install custom node package providing: {node_type}",
    ),
    ErrorPattern(
        name="import_failed",
        category="missing_node",
        regex=r"IMPORT FAILED.*?custom_nodes[/\\]([^\s/\\]+)",
        description="Custom node failed to import",
        fix_template=[],  # Try reinstalling deps
        fix_description="Fix dependencies for custom node: {node_name}",
    ),

    # ━━━━ CONNECTION / TYPE ERRORS ━━━━
    ErrorPattern(
        name="type_mismatch",
        category="type_error",
        regex=r"(?:TypeError|KeyError).*?(?:node|Node)\s*(?:#?\d+|'[^']+').*?(?:input|output)\s*'([^']+)'",
        description="Input/output type mismatch between nodes",
        fix_template=[],
        fix_description="Check node connections for type compatibility",
    ),
    ErrorPattern(
        name="unexpected_kwarg",
        category="version_error",
        regex=r"TypeError: (\w+)\(\) got an unexpected keyword argument '(\w+)'",
        description="Node API changed (version mismatch)",
        fix_template=[],
        fix_description="Update custom node or check version compatibility",
    ),

    # ━━━━ GENERIC FALLBACKS ━━━━
    ErrorPattern(
        name="generic_file_not_found",
        category="file_not_found",
        regex=r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]",
        description="File not found",
        fix_template=[],
        fix_description="Missing file: {path}",
    ),
    ErrorPattern(
        name="generic_permission",
        category="permission_error",
        regex=r"PermissionError:.*?['\"]([^'\"]+)['\"]",
        description="Permission denied",
        fix_template=["chmod -R 755 {path}"],
        fix_description="Fix permissions on: {path}",
    ),
]


def match_error(error_text: str) -> list[ErrorMatch]:
    """Match error text against all known patterns.
    
    Returns list of matches, most specific first.
    Specific patterns (like missing_insightface) take priority over
    generic ones (like missing_module_generic).
    """
    matches = []

    for pattern in ERROR_PATTERNS:
        m = re.search(pattern.regex, error_text, re.IGNORECASE | re.DOTALL)
        if m:
            groups = m.groups()
            extracted = {f"group_{i}": g for i, g in enumerate(groups)}
            
            # Map common group names (safely)
            if pattern.category == "missing_module":
                extracted["module"] = groups[0] if groups else "unknown"
            elif pattern.category == "model_not_found":
                extracted["filename"] = groups[0] if groups else ""
            elif pattern.category == "missing_node":
                extracted["node_type"] = groups[0] if groups else ""

            # Build fix commands with extracted values
            fix_commands = []
            for cmd_template in pattern.fix_template:
                try:
                    fix_commands.append(cmd_template.format(**extracted))
                except KeyError:
                    fix_commands.append(cmd_template)

            fix_desc = pattern.fix_description
            try:
                fix_desc = fix_desc.format(**extracted)
            except KeyError:
                pass

            matches.append(ErrorMatch(
                category=pattern.category,
                pattern_name=pattern.name,
                description=pattern.description,
                fix_commands=fix_commands,
                fix_description=fix_desc,
                extracted=extracted,
            ))

    # Sort: specific patterns before generic (specific have longer names)
    # and non-generic categories first
    def specificity(m: ErrorMatch) -> int:
        if "generic" in m.pattern_name:
            return 0
        return len(m.pattern_name)

    matches.sort(key=specificity, reverse=True)

    # Deduplicate: if we matched both insightface-specific and module-generic, 
    # keep only the specific one
    seen_categories = set()
    deduped = []
    for m in matches:
        if m.category not in seen_categories:
            deduped.append(m)
            seen_categories.add(m.category)

    return deduped
