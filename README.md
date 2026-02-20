# 🩺 comfyui-doctor

**Make any ComfyUI workflow work in one shot.**

Give it a workflow JSON → it installs missing nodes, downloads models, fixes broken inputs, and runs it. No more "missing node type" errors.

## Features

### V1 — One-Shot Magic ✅
- **Auto-detect missing nodes** — compares workflow vs `/object_info`
- **Auto-install custom nodes** — git clone + requirements.txt (40+ curated + 30,000+ via ComfyUI-Manager fallback)
- **Auto-download models** — aria2c x16 parallel (30+ models mapped to HuggingFace URLs)
- **Auto-restart ComfyUI** — kills & relaunches after installing new nodes
- **Auto-fix broken inputs** — clamp out-of-range values, fix enum case, fill missing defaults from `/object_info`
- **Auto-retry on error** — parse errors, match 35+ patterns, apply fix, retry (max 3x)
- **Input validation** — warns about missing required inputs BEFORE queuing

### Knowledge Bases
- **35+ error patterns** — ModuleNotFoundError, CUDA OOM, missing models, validation errors...
- **40+ curated node→repo mappings** — Impact-Pack, KJNodes, ControlNet-Aux, WAS, rgthree, AnimateDiff...
- **30,000+ node types** via ComfyUI-Manager extension-node-map fallback
- **30+ model→URL mappings** — SDXL, SD1.5, LoRAs, VAEs, ControlNets, upscalers...

## Installation

```bash
pip install comfyui-doctor
# or from source:
git clone https://github.com/wuraaang/comfyui-doctor
cd comfyui-doctor && pip install -e .
```

## Usage

### One-Shot Run (the magic command)
```bash
# Give it a workflow, it just works
comfyui-doctor run workflow.json

# With custom ComfyUI URL and path
comfyui-doctor run workflow.json --url http://localhost:8188 --path /opt/ComfyUI

# Dry run — analyze without executing
comfyui-doctor run workflow.json --dry-run
```

### Other Commands
```bash
# Analyze workflow without running
comfyui-doctor analyze workflow.json

# Fix missing deps without running
comfyui-doctor fix workflow.json

# Full system diagnosis
comfyui-doctor diagnose

# Check ComfyUI status
comfyui-doctor status

# Look up where to install a node
comfyui-doctor lookup "FaceDetailer"

# Look up error fix
comfyui-doctor check-error "ModuleNotFoundError: No module named 'insightface'"

# Look up model download URL
comfyui-doctor check-model "sd_xl_base_1.0.safetensors"

# List nodes in a workflow
comfyui-doctor nodes workflow.json
```

## How It Works

```
Parse workflow.json
       ↓
Query /object_info → find missing nodes
       ↓
Look up node packages (local DB → ComfyUI-Manager fallback)
       ↓
git clone + pip install requirements.txt
       ↓
Restart ComfyUI (kill + relaunch + wait)
       ↓
Check models → aria2c x16 download
       ↓
Validate inputs → auto-fix (clamp, fill defaults, fix enums)
       ↓
Queue prompt → wait for completion
       ↓
Error? → match 35+ patterns → apply fix → retry
       ↓
✅ Success! Image saved.
```

## Test Results

Tested live on GPUhub RTX 5090 + ComfyUI 0.14.1 + PyTorch 2.10+cu128:

| # | Workflow | Custom Nodes | Result |
|---|---------|-------------|--------|
| 1 | Simple SDXL | None | ✅ one-shot |
| 2 | KJNodes resize | ComfyUI-KJNodes | ✅ auto-install + restart |
| 3 | ControlNet Canny | comfyui_controlnet_aux + deps | ✅ auto-install + restart |
| 4 | Comfyroll batch | ComfyUI_Comfyroll | ✅ auto-install + restart |
| 7 | WAS filters (broken inputs) | was-node-suite + auto-clamp | ✅ auto-fix brightness |
| 8 | Multi-node (KJNodes+WAS) | Already installed | ✅ direct |
| 9 | Built-in advanced | None | ✅ direct |
| 10 | ImageBlend | Built-in | ✅ direct |
| 11 | WAS Resize (5 missing inputs) | auto-fill defaults | ✅ auto-fix 5 inputs |

**10/11 success rate** — the one failure was a deliberately incomplete test workflow.

## Architecture

```
comfyui_doctor/
├── cli.py                    # Typer CLI (10 commands)
├── core/
│   ├── api.py                # ComfyUI HTTP client (urllib)
│   ├── doctor.py             # Auto-fix engine (the brain)
│   └── workflow.py           # JSON parser, validator, auto-fixer
└── knowledge/
    ├── error_db.py           # 35+ regex error patterns
    ├── node_map.py           # 40+ curated node→repo mappings
    ├── model_map.py          # 30+ model→URL mappings
    └── manager_db.py         # ComfyUI-Manager 30K+ fallback
```

## Requirements

- Python 3.10+
- ComfyUI running with `--listen` flag
- Minimal deps: `typer`, `rich` (that's it!)

## License

MIT

## Credits

Built by [wuraaang](https://github.com/wuraaang) with ❤️ and 🩺
