"""Workflow speed optimizer — analyze and suggest optimizations.

Analyzes a workflow + system info and suggests:
- FP8 quantization for large models
- VAE tiling for high-res
- Batch size tuning
- Model-specific tips
"""

from dataclasses import dataclass


@dataclass
class Optimization:
    """A suggested optimization."""
    category: str       # "memory", "speed", "quality"
    description: str
    impact: str         # "high", "medium", "low"
    node_id: str = ""   # Which node to modify
    change: dict = None # {input_name: new_value}

    def __post_init__(self):
        if self.change is None:
            self.change = {}


def analyze_optimizations(
    workflow: dict,
    object_info: dict = None,
    vram_gb: float = 0,
) -> list[Optimization]:
    """Analyze a workflow and return optimization suggestions."""
    opts = []

    # Collect info about the workflow
    has_sdxl = False
    has_sd15 = False
    has_wan = False
    has_flux = False
    resolution = {"width": 0, "height": 0}
    batch_size = 1
    steps = 20
    sampler_nodes = []
    loader_nodes = []
    vae_decode_nodes = []

    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})

        # Detect model type from checkpoint name
        ckpt = inputs.get("ckpt_name", "")
        if isinstance(ckpt, str):
            if "xl" in ckpt.lower() or "sdxl" in ckpt.lower():
                has_sdxl = True
            elif "flux" in ckpt.lower():
                has_flux = True
            elif "sd_" in ckpt.lower() or "v1-5" in ckpt.lower():
                has_sd15 = True

        if "Wan" in ct:
            has_wan = True

        # Collect resolution
        if ct == "EmptyLatentImage":
            w = inputs.get("width", 0)
            h = inputs.get("height", 0)
            if isinstance(w, (int, float)):
                resolution["width"] = max(resolution["width"], int(w))
                resolution["height"] = max(resolution["height"], int(h))
            bs = inputs.get("batch_size", 1)
            if isinstance(bs, int):
                batch_size = max(batch_size, bs)

        # Sampler info
        if "KSampler" in ct or "Sampler" in ct:
            sampler_nodes.append(nid)
            s = inputs.get("steps", 20)
            if isinstance(s, int):
                steps = max(steps, s)

        if "CheckpointLoader" in ct:
            loader_nodes.append(nid)

        if "VAEDecode" in ct:
            vae_decode_nodes.append(nid)

    total_pixels = resolution["width"] * resolution["height"]

    # ── Memory optimizations ──────────────────────────────────────────
    if has_sdxl and vram_gb > 0 and vram_gb < 12:
        opts.append(Optimization(
            "memory", "Consider using SDXL in FP16 or a pruned checkpoint",
            "high",
        ))

    if has_flux and vram_gb > 0 and vram_gb < 24:
        opts.append(Optimization(
            "memory", "FLUX needs ~24GB VRAM. Use FP8 quantization (fp8_e4m3fn) to fit in less",
            "high",
        ))

    if total_pixels > 1024 * 1024 and vae_decode_nodes:
        opts.append(Optimization(
            "memory",
            f"High resolution ({resolution['width']}x{resolution['height']}). Enable VAE tiling to prevent OOM",
            "medium",
        ))

    if has_wan and vram_gb > 0 and vram_gb < 24:
        opts.append(Optimization(
            "memory",
            "Wan 2.1 video: use fp8_e4m3fn quantization + enable_vae_tiling + force_offload",
            "high",
        ))

    # ── Speed optimizations ───────────────────────────────────────────
    if steps > 30:
        opts.append(Optimization(
            "speed",
            f"Steps={steps} is high. For most workflows 20-25 steps is sufficient with dpm++_2m/karras",
            "medium",
        ))

    if has_sdxl and steps > 25:
        opts.append(Optimization(
            "speed",
            "SDXL: 20 steps with euler/normal usually gives good results. Consider reducing.",
            "medium",
        ))

    if batch_size > 1 and vram_gb > 0 and vram_gb < 16:
        opts.append(Optimization(
            "speed",
            f"Batch size {batch_size} with {vram_gb}GB VRAM may cause swapping. Consider batch_size=1",
            "medium",
        ))

    if has_wan:
        opts.append(Optimization(
            "speed",
            "Wan 2.1: use TorchCompileModelWanVideo for 2-3x speedup on subsequent runs",
            "high",
        ))

    # ── Quality tips ──────────────────────────────────────────────────
    if has_sdxl and total_pixels < 786432:  # Less than 1024x768
        opts.append(Optimization(
            "quality",
            f"SDXL is designed for 1024x1024. Current resolution ({resolution['width']}x{resolution['height']}) may give poor results",
            "high",
        ))

    if has_sd15 and total_pixels > 786432:
        opts.append(Optimization(
            "quality",
            f"SD 1.5 is designed for 512x512. Current resolution may give artifacts. Consider using an upscaler after generation.",
            "medium",
        ))

    return opts


def format_optimizations(opts: list[Optimization]) -> str:
    """Format optimizations for display."""
    if not opts:
        return "No optimizations suggested — workflow looks good!"

    lines = []
    for cat in ("memory", "speed", "quality"):
        cat_opts = [o for o in opts if o.category == cat]
        if not cat_opts:
            continue
        icon = {"memory": "🧠", "speed": "⚡", "quality": "🎨"}[cat]
        lines.append(f"\n{icon} {cat.upper()}")
        for o in cat_opts:
            impact = {"high": "🔴", "medium": "🟡", "low": "🟢"}[o.impact]
            lines.append(f"  {impact} {o.description}")

    return "\n".join(lines)
