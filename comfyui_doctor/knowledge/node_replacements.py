"""Node replacement map — when a custom node is truly unfindable,
suggest or apply an equivalent built-in or common alternative.

This is a last-resort: the doctor first tries to install the original node.
Only if that fails AND a replacement exists here, it can swap the node.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeReplacement:
    """A replacement for an unavailable node type."""
    original: str              # The missing node type
    replacement: str           # The replacement node type
    input_mapping: dict        # {original_input_name: replacement_input_name}
    description: str = ""
    is_builtin: bool = True    # True if replacement is a ComfyUI built-in
    notes: str = ""            # Caveats about the replacement
    output_mapping: dict = None  # {original_slot_index: replacement_slot_index}

    def __post_init__(self):
        if self.output_mapping is None:
            self.output_mapping = {}


# ── Replacement map ───────────────────────────────────────────────────
# original_type → NodeReplacement
#
# Rules:
# 1. Only replace when the original is truly unfindable (no repo, no Manager match)
# 2. Prefer built-in ComfyUI nodes as replacements
# 3. Input mapping must be accurate — wrong mapping = broken workflow
# 4. Document caveats in notes

REPLACEMENTS: dict[str, NodeReplacement] = {
    # ── Text encoding alternatives ────────────────────────────────────
    "BNK_CLIPTextEncodeAdvanced": NodeReplacement(
        "BNK_CLIPTextEncodeAdvanced",
        "CLIPTextEncode",
        {"text": "text"},
        "Advanced CLIP encoder → standard CLIP encoder",
        notes="Loses token normalization and weight interpretation options",
    ),
    "WildcardEncode": NodeReplacement(
        "WildcardEncode",
        "CLIPTextEncode",
        {"text": "text", "populated_text": "text"},
        "Impact Pack wildcard encoder → standard CLIP",
        notes="Wildcards won't be expanded — use literal text",
    ),

    # ── Loader alternatives ───────────────────────────────────────────
    "CheckpointLoaderSimple|SDXL": NodeReplacement(
        "CheckpointLoaderSimple|SDXL",
        "CheckpointLoaderSimple",
        {"ckpt_name": "ckpt_name"},
        "Some workflows use a patched SDXL loader",
    ),
    "Efficient Loader": NodeReplacement(
        "Efficient Loader",
        "CheckpointLoaderSimple",
        {"ckpt_name": "ckpt_name"},
        "Efficiency Nodes loader → standard checkpoint loader",
        is_builtin=True,
        notes="Loses LoRA/VAE/prompt bundling — those need separate nodes",
    ),
    "Eff. Loader SDXL": NodeReplacement(
        "Eff. Loader SDXL",
        "CheckpointLoaderSimple",
        {"ckpt_name": "ckpt_name"},
        "Efficiency SDXL loader → standard checkpoint loader",
        is_builtin=True,
        notes="Loses dual CLIP/LoRA bundling",
    ),

    # ── Sampler alternatives ──────────────────────────────────────────
    "KSampler (Efficient)": NodeReplacement(
        "KSampler (Efficient)",
        "KSampler",
        {"seed": "seed", "steps": "steps", "cfg": "cfg",
         "sampler_name": "sampler_name", "scheduler": "scheduler",
         "denoise": "denoise"},
        "Efficiency sampler → standard KSampler",
        notes="Loses preview and script hooks",
    ),
    "KSampler SDXL (Eff.)": NodeReplacement(
        "KSampler SDXL (Eff.)",
        "KSampler",
        {"seed": "seed", "steps": "steps", "cfg": "cfg",
         "sampler_name": "sampler_name", "scheduler": "scheduler",
         "denoise": "denoise"},
        "Efficiency SDXL sampler → standard KSampler",
    ),
    "SamplerCustom": NodeReplacement(
        "SamplerCustom",
        "KSampler",
        {"noise_seed": "seed", "steps": "steps", "cfg": "cfg"},
        "Custom sampler → KSampler",
        notes="Loses separate noise/guider/sigmas control",
    ),

    # ── Image save alternatives ───────────────────────────────────────
    "SaveImageWithMetaData": NodeReplacement(
        "SaveImageWithMetaData",
        "SaveImage",
        {"filename_prefix": "filename_prefix"},
        "Metadata save → standard SaveImage",
        notes="Loses metadata embedding (prompt, workflow info in PNG)",
    ),
    "Image Save": NodeReplacement(
        "Image Save",
        "SaveImage",
        {"filename_prefix": "filename_prefix"},
        "WAS Image Save → standard SaveImage",
    ),
    "PreviewImage": NodeReplacement(
        "PreviewImage",
        "SaveImage",
        {"filename_prefix": "filename_prefix"},
        "Preview → Save (when preview isn't available)",
    ),

    # ── Image manipulation alternatives ───────────────────────────────
    "ImageScaleBy": NodeReplacement(
        "ImageScaleBy",
        "ImageScale",
        {"upscale_method": "upscale_method"},
        "Scale by factor → scale to absolute size",
        notes="Need to set width/height manually instead of scale_by",
    ),
    "CR Image Resize": NodeReplacement(
        "CR Image Resize",
        "ImageScale",
        {"upscale_method": "upscale_method"},
        "Comfyroll resize → built-in ImageScale",
    ),

    # ── Conditioning alternatives ─────────────────────────────────────
    "ConditioningCombine": NodeReplacement(
        "ConditioningCombine",
        "ConditioningConcat",
        {"conditioning_1": "conditioning_from",
         "conditioning_2": "conditioning_to"},
        "Combine → Concat conditioning",
        notes="Slightly different behavior: concat vs average",
    ),

    # ── Utility nodes (can often be removed) ──────────────────────────
    "Note": NodeReplacement(
        "Note", "__REMOVE__", {},
        "Notes are UI-only, safe to remove",
    ),
    "PrimitiveNode": NodeReplacement(
        "PrimitiveNode", "__REMOVE__", {},
        "Primitives are UI-only, values propagated during conversion",
    ),
    "Reroute": NodeReplacement(
        "Reroute", "__PASSTHROUGH__", {},
        "Reroutes are UI-only, connections pass through",
    ),
    "SetNode": NodeReplacement(
        "SetNode", "__REMOVE__", {},
        "Set/Get are UI-only routing nodes",
    ),
    "GetNode": NodeReplacement(
        "GetNode", "__REMOVE__", {},
        "Set/Get are UI-only routing nodes",
    ),

}


def find_replacement(node_type: str) -> Optional[NodeReplacement]:
    """Find a replacement for an unavailable node type."""
    return REPLACEMENTS.get(node_type)


def get_removable_types() -> set[str]:
    """Get set of node types that can be safely removed."""
    return {k for k, v in REPLACEMENTS.items()
            if v.replacement in ("__REMOVE__", "__PASSTHROUGH__")}


def get_all_replaceable() -> dict[str, NodeReplacement]:
    """Get all replacements (excluding removable)."""
    return {k: v for k, v in REPLACEMENTS.items()
            if v.replacement not in ("__REMOVE__", "__PASSTHROUGH__")}
