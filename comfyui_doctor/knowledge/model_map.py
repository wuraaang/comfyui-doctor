"""Mapping of common model filenames to download URLs.

When a workflow references a model by filename, this map
tells comfyui-doctor where to download it from.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Download info for a model."""
    url: str
    filename: str
    size_gb: float  # Approximate size for user info
    model_folder: str  # Subfolder in models/
    description: str = ""
    hf_token_required: bool = False


# ── Model Filename → Download URL ─────────────────────────────────────
# Key = exact filename as it appears in workflow JSON
# Some models have multiple names; we list the common variants

MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ━━━━ SDXL Checkpoints ━━━━
    "sd_xl_base_1.0.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "sd_xl_base_1.0.safetensors", 6.9, "checkpoints",
        "Stable Diffusion XL Base 1.0",
    ),
    "sd_xl_refiner_1.0.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
        "sd_xl_refiner_1.0.safetensors", 6.1, "checkpoints",
        "Stable Diffusion XL Refiner 1.0",
    ),

    # ━━━━ SD 1.5 Checkpoints ━━━━
    "v1-5-pruned-emaonly.safetensors": ModelInfo(
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "v1-5-pruned-emaonly.safetensors", 4.3, "checkpoints",
        "Stable Diffusion 1.5",
    ),

    # ━━━━ Illustrious XL ━━━━
    "wai-illustrious-sdxl-v16.safetensors": ModelInfo(
        "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
        "wai-illustrious-sdxl-v16.safetensors", 6.9, "checkpoints",
        "WAI Illustrious SDXL v16",
    ),
    "illustriousXLV20_v20Stable.safetensors": ModelInfo(
        "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
        "illustriousXLV20_v20Stable.safetensors", 6.9, "checkpoints",
        "Illustrious XL v2.0 Stable (mapped to v0.1 — update URL when available)",
    ),

    # ━━━━ FLUX ━━━━
    "flux1-dev.safetensors": ModelInfo(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "flux1-dev.safetensors", 23.8, "unet",
        "FLUX.1 Dev", True,
    ),
    "flux1-schnell.safetensors": ModelInfo(
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
        "flux1-schnell.safetensors", 23.8, "unet",
        "FLUX.1 Schnell (fast)",
    ),

    # ━━━━ VAE ━━━━
    "sdxl_vae.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
        "sdxl_vae.safetensors", 0.3, "vae",
        "SDXL VAE",
    ),
    "ae.safetensors": ModelInfo(
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "ae.safetensors", 0.3, "vae",
        "FLUX VAE (Autoencoder)", True,
    ),
    "vae-ft-mse-840000-ema-pruned.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "vae-ft-mse-840000-ema-pruned.safetensors", 0.3, "vae",
        "SD 1.5 VAE FT-MSE",
    ),

    # ━━━━ CLIP ━━━━
    "clip_l.safetensors": ModelInfo(
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "clip_l.safetensors", 0.2, "clip",
        "CLIP-L text encoder",
    ),
    "t5xxl_fp16.safetensors": ModelInfo(
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
        "t5xxl_fp16.safetensors", 9.8, "clip",
        "T5-XXL text encoder (FP16)",
    ),
    "t5xxl_fp8_e4m3fn.safetensors": ModelInfo(
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors", 4.9, "clip",
        "T5-XXL text encoder (FP8, saves VRAM)",
    ),

    # ━━━━ ControlNet ━━━━
    "control-lora-depth-rank256.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors",
        "control-lora-depth-rank256.safetensors", 0.8, "controlnet",
        "SDXL Control LoRA - Depth",
    ),
    "control-lora-canny-rank256.safetensors": ModelInfo(
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors",
        "control-lora-canny-rank256.safetensors", 0.8, "controlnet",
        "SDXL Control LoRA - Canny",
    ),

    # ━━━━ Upscale ━━━━
    "4x-UltraSharp.pth": ModelInfo(
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        "4x-UltraSharp.pth", 0.07, "upscale_models",
        "4x UltraSharp upscaler",
    ),
    "4x-UltraSharpV2.safetensors": ModelInfo(
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        "4x-UltraSharpV2.safetensors", 0.07, "upscale_models",
        "4x UltraSharp V2 upscaler (mapped to V1 — same model)",
    ),
    "RealESRGAN_x4plus.pth": ModelInfo(
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth", 0.07, "upscale_models",
        "RealESRGAN 4x Plus upscaler",
    ),

    # ━━━━ Face Detection (Impact-Pack) ━━━━
    "bbox/face_yolov8m.pt": ModelInfo(
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
        "face_yolov8m.pt", 0.05, "ultralytics/bbox",
        "Face detection YOLO v8m",
    ),
    "face_yolov8m.pt": ModelInfo(
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
        "face_yolov8m.pt", 0.05, "ultralytics/bbox",
        "Face detection YOLO v8m",
    ),
    "sam_vit_b_01ec64.pth": ModelInfo(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_01ec64.pth", 0.4, "sams",
        "SAM ViT-B (Segment Anything)",
    ),

    # ━━━━ Wan 2.1 Video ━━━━
    "wan2.1_t2v_1.3B_bf16.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
        "wan2.1_t2v_1.3B_bf16.safetensors", 2.7, "diffusion_models/wan",
        "Wan 2.1 T2V 1.3B (smallest, text-to-video)",
    ),
    "wan2.1_t2v_14B_bf16.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
        "wan2.1_t2v_14B_bf16.safetensors", 28.0, "diffusion_models/wan",
        "Wan 2.1 T2V 14B (full, text-to-video)",
    ),
    "wan2.1_i2v_720p_14B_bf16.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors",
        "wan2.1_i2v_720p_14B_bf16.safetensors", 28.0, "diffusion_models/wan",
        "Wan 2.1 I2V 720p 14B (image-to-video)",
    ),
    "wan_2.1_vae.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "wan_2.1_vae.safetensors", 0.2, "vae/wan",
        "Wan 2.1 VAE",
    ),
    "umt5_xxl_fp16.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors",
        "umt5_xxl_fp16.safetensors", 11.4, "text_encoders",
        "UMT5-XXL FP16 text encoder for Wan 2.1",
    ),
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": ModelInfo(
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.7, "text_encoders",
        "UMT5-XXL FP8 text encoder for Wan 2.1 (smaller, recommended)",
    ),

    # ━━━━ IP-Adapter ━━━━
    "ip-adapter-plus_sd15.safetensors": ModelInfo(
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
        "ip-adapter-plus_sd15.safetensors", 0.1, "ipadapter",
        "IP-Adapter Plus SD1.5",
    ),
    "ip-adapter-plus_sdxl_vit-h.safetensors": ModelInfo(
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "ip-adapter-plus_sdxl_vit-h.safetensors", 0.8, "ipadapter",
        "IP-Adapter Plus SDXL",
    ),
}


def lookup_model(filename: str) -> Optional[ModelInfo]:
    """Find download info for a model by filename."""
    # Exact match
    if filename in MODEL_REGISTRY:
        return MODEL_REGISTRY[filename]
    
    # Normalize Windows backslashes (e.g., "Illustrious\model.safetensors")
    filename = filename.replace("\\", "/")
    
    # Try without path prefix (e.g., "bbox/face_yolov8m.pt" → "face_yolov8m.pt")
    basename = filename.split("/")[-1]
    if basename in MODEL_REGISTRY:
        return MODEL_REGISTRY[basename]
    
    # Try case-insensitive
    for key, info in MODEL_REGISTRY.items():
        if key.lower() == filename.lower() or key.lower() == basename.lower():
            return info
    
    return None


def estimate_download_size(models: list[dict]) -> float:
    """Estimate total download size in GB for a list of model references."""
    total = 0.0
    for ref in models:
        info = lookup_model(ref.get("filename", ""))
        if info:
            total += info.size_gb
    return total
