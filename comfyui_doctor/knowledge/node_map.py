"""Mapping of custom node class_types to their git repos.

This is the core data that lets comfyui-doctor auto-install missing nodes.
Built from ComfyUI-Manager's custom-node-list.json + manual additions.

Format: node_class_type → {repo_url, package_name, pip_deps}
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class NodePackage:
    """Info about a custom node package."""
    repo_url: str
    package_name: str  # Folder name in custom_nodes/
    pip_deps: list[str]  # Additional pip packages needed
    description: str = ""


# ── Node Type → Package mapping ──────────────────────────────────────
# This maps individual node class_types to the package that provides them.
# Not exhaustive — for unknown types we fall back to ComfyUI-Manager's API.

NODE_PACKAGES: dict[str, NodePackage] = {
    # ━━━━ rgthree-comfy ━━━━
    "Any Switch (rgthree)": NodePackage(
        "https://github.com/rgthree/rgthree-comfy",
        "rgthree-comfy", [],
        "rgthree's nodes: switches, seed, power lora loader, etc.",
    ),
    "Seed (rgthree)": NodePackage(
        "https://github.com/rgthree/rgthree-comfy",
        "rgthree-comfy", [],
    ),
    "Power Lora Loader (rgthree)": NodePackage(
        "https://github.com/rgthree/rgthree-comfy",
        "rgthree-comfy", [],
    ),
    "Fast Groups Muter (rgthree)": NodePackage(
        "https://github.com/rgthree/rgthree-comfy",
        "rgthree-comfy", [],
    ),
    "Bookmark (rgthree)": NodePackage(
        "https://github.com/rgthree/rgthree-comfy",
        "rgthree-comfy", [],
    ),

    # ━━━━ ComfyUI-Crystools ━━━━
    "Primitive float [Crystools]": NodePackage(
        "https://github.com/crystian/ComfyUI-Crystools",
        "ComfyUI-Crystools", [],
        "Primitive types, debug tools",
    ),
    "Primitive integer [Crystools]": NodePackage(
        "https://github.com/crystian/ComfyUI-Crystools",
        "ComfyUI-Crystools", [],
    ),
    "Sampler Selector [Crystools]": NodePackage(
        "https://github.com/crystian/ComfyUI-Crystools",
        "ComfyUI-Crystools", [],
    ),
    "Scheduler Selector [Crystools]": NodePackage(
        "https://github.com/crystian/ComfyUI-Crystools",
        "ComfyUI-Crystools", [],
    ),

    # ━━━━ ComfyUI-Impact-Pack ━━━━
    "FaceDetailer": NodePackage(
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Pack",
        ["segment-anything", "ultralytics"],
        "Face detection, SAM, detectors",
    ),
    "SAMLoader": NodePackage(
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Pack",
        ["segment-anything"],
    ),
    "UltralyticsDetectorProvider": NodePackage(
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Pack",
        ["ultralytics"],
    ),
    "BboxDetectorSEGS": NodePackage(
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Pack",
        ["ultralytics"],
    ),

    # ━━━━ ComfyUI_UltimateSDUpscale ━━━━
    "UltimateSDUpscale": NodePackage(
        "https://github.com/ssitu/ComfyUI_UltimateSDUpscale",
        "ComfyUI_UltimateSDUpscale", [],
        "Tiled upscaling",
    ),

    # ━━━━ ComfyUI-KJNodes ━━━━
    "ImageResizeKJ": NodePackage(
        "https://github.com/kijai/ComfyUI-KJNodes",
        "ComfyUI-KJNodes", [],
        "KJ's utility nodes",
    ),
    "GetImageSizeAndCount": NodePackage(
        "https://github.com/kijai/ComfyUI-KJNodes",
        "ComfyUI-KJNodes", [],
    ),

    # ━━━━ ComfyUI-WanVideoWrapper ━━━━
    "DownloadAndLoadWanModel": NodePackage(
        "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "ComfyUI-WanVideoWrapper",
        ["accelerate", "diffusers"],
        "Wan video generation wrapper",
    ),
    "WanVideoSampler": NodePackage(
        "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "ComfyUI-WanVideoWrapper",
        ["accelerate", "diffusers"],
    ),

    # ━━━━ ComfyUI-VideoHelperSuite ━━━━
    "VHS_LoadVideo": NodePackage(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "ComfyUI-VideoHelperSuite", [],
        "Video loading/saving helpers",
    ),
    "VHS_VideoCombine": NodePackage(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "ComfyUI-VideoHelperSuite", [],
    ),
    "VHS_LoadImages": NodePackage(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "ComfyUI-VideoHelperSuite", [],
    ),

    # ━━━━ ComfyUI-AnimateDiff ━━━━
    "ADE_AnimateDiffLoaderGen1": NodePackage(
        "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved",
        "ComfyUI-AnimateDiff-Evolved",
        [],
        "AnimateDiff animation",
    ),
    "ADE_UseEvolvedSampling": NodePackage(
        "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved",
        "ComfyUI-AnimateDiff-Evolved",
        [],
    ),

    # ━━━━ was-node-suite ━━━━
    "Image Filter Adjustments": NodePackage(
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "was-node-suite-comfyui",
        [],
        "WAS node suite: image filters, text, etc.",
    ),

    # ━━━━ ComfyUI_Comfyroll ━━━━
    "CR Latent Batch Size": NodePackage(
        "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
        "ComfyUI_Comfyroll_CustomNodes",
        [],
        "Comfyroll utility nodes",
    ),

    # ━━━━ ComfyUI-SaveImageWithMetaData ━━━━
    "SaveImageWithMetaData": NodePackage(
        "https://github.com/nkchocoai/ComfyUI-SaveImageWithMetaData",
        "ComfyUI-SaveImageWithMetaData",
        [],
        "Save images with embedded workflow metadata",
    ),

    # ━━━━ cg-image-filter ━━━━
    "Image Filter": NodePackage(
        "https://github.com/chrisgoringe/cg-image-filter",
        "cg-image-filter",
        [],
        "Image selection/filter pause node",
    ),

    # ━━━━ ComfyUI-IP-Adapter ━━━━
    "IPAdapterApply": NodePackage(
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "ComfyUI_IPAdapter_plus",
        [],
        "IP-Adapter for image-guided generation",
    ),
    "IPAdapterModelLoader": NodePackage(
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "ComfyUI_IPAdapter_plus",
        [],
    ),

    # ━━━━ ComfyUI-ControlNet-Aux ━━━━
    "CannyEdgePreprocessor": NodePackage(
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "comfyui_controlnet_aux",
        ["mediapipe", "einops"],
        "ControlNet preprocessors",
    ),
    "DepthAnythingPreprocessor": NodePackage(
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "comfyui_controlnet_aux",
        [],
    ),
    "DWPreprocessor": NodePackage(
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "comfyui_controlnet_aux",
        [],
    ),
    "OpenposePreprocessor": NodePackage(
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "comfyui_controlnet_aux",
        [],
    ),

    # ━━━━ ComfyUI-Florence2 ━━━━
    "Florence2ModelLoader": NodePackage(
        "https://github.com/kijai/ComfyUI-Florence2",
        "ComfyUI-Florence2",
        ["transformers", "einops"],
        "Florence2 vision-language model",
    ),

    # ━━━━ LongCat / Avatars ━━━━
    "LongCatAvatarLoader": NodePackage(
        "https://github.com/LongCat-AI/ComfyUI-LongCat",
        "ComfyUI-LongCat",
        [],
        "LongCat talking avatar generation",
    ),
}


def lookup_node_type(class_type: str) -> Optional[NodePackage]:
    """Find the package for a given node class_type."""
    return NODE_PACKAGES.get(class_type)


def lookup_multiple(class_types: set[str]) -> dict[str, Optional[NodePackage]]:
    """Look up packages for multiple node types.
    
    Returns: {class_type: NodePackage or None}
    """
    return {ct: lookup_node_type(ct) for ct in class_types}


def get_unique_repos(packages: dict[str, Optional[NodePackage]]) -> dict[str, NodePackage]:
    """Deduplicate packages by repo URL."""
    repos = {}
    for ct, pkg in packages.items():
        if pkg and pkg.repo_url not in repos:
            repos[pkg.repo_url] = pkg
    return repos


def get_all_pip_deps(packages: dict[str, Optional[NodePackage]]) -> set[str]:
    """Collect all pip dependencies from a set of packages."""
    deps = set()
    for pkg in packages.values():
        if pkg:
            deps.update(pkg.pip_deps)
    return deps
