# 🩺 ComfyUI Doctor

**Make any ComfyUI workflow work in one shot.**

Give it a workflow JSON → it analyzes what's missing → installs nodes, dependencies, models → runs it → auto-fixes errors → retries. Zero manual debugging.

## 🚀 Quick Start

```bash
pip install comfyui-doctor

# The magic one-liner:
comfyui-doctor run workflow.json

# Just analyze without running:
comfyui-doctor analyze workflow.json

# Full health check:
comfyui-doctor diagnose
```

## ✨ Features

| Feature | Description |
|---|---|
| **One-shot workflow runner** | Give it any workflow → it just works |
| **Auto-detect missing nodes** | Compares workflow vs installed nodes, installs what's missing |
| **Auto-fix Python deps** | `ModuleNotFoundError`? Fixed automatically |
| **Model downloader** | Detects missing models, knows where to download them |
| **Error knowledge base** | 30+ error patterns with automatic fixes |
| **CUDA OOM handling** | Suggestions to reduce VRAM usage |
| **Health check** | Full diagnosis of your ComfyUI setup |
| **Dry run mode** | See what would happen without doing anything |

## 📖 Commands

### `comfyui-doctor run <workflow.json>`
The main command. Analyzes, fixes, runs, and auto-retries.

```bash
# Basic usage
comfyui-doctor run my-workflow.json

# Custom ComfyUI URL
comfyui-doctor run workflow.json --url http://localhost:8188

# More retries
comfyui-doctor run workflow.json --retries 5

# Dry run (analyze + show fixes, don't execute)
comfyui-doctor run workflow.json --dry-run

# Disable auto-fix
comfyui-doctor run workflow.json --no-fix
```

### `comfyui-doctor analyze <workflow.json>`
Analyze a workflow without running it. Shows missing nodes, models, dependencies.

### `comfyui-doctor fix <workflow.json>`
Apply fixes without running the workflow. Useful for setup.

### `comfyui-doctor diagnose`
Full health check: ComfyUI status, GPU info, installed nodes, models, disk space.

### `comfyui-doctor status`
Quick status: is ComfyUI running? Queue? VRAM usage?

### `comfyui-doctor check-error "ModuleNotFoundError: No module named 'insightface'"`
Look up any error in the knowledge base.

### `comfyui-doctor check-model "sd_xl_base_1.0.safetensors"`
Look up a model — shows download URL, size, folder.

### `comfyui-doctor nodes <workflow.json>`
List all node types used in a workflow.

## 🧠 How It Works

### Pre-flight Analysis
1. Parse workflow JSON (supports both API and UI formats)
2. Extract all `class_type` values (node types)
3. Query ComfyUI's `/object_info` for registered types
4. Diff → find missing nodes
5. Look up missing nodes in built-in knowledge base (100+ mappings)
6. Check model file references against disk
7. Look up missing models in built-in registry (50+ models)

### Auto-Fix Pipeline
1. Install missing custom node repos (`git clone`)
2. Install missing pip dependencies
3. Download missing models (aria2c x16 for speed)
4. Queue the workflow
5. If error → match against 30+ error patterns
6. Apply fix → retry (up to N times)

### Error Knowledge Base
The doctor knows about:
- **Missing Python modules** (insightface, onnxruntime, mediapipe, etc.)
- **CUDA out of memory** (with reduction suggestions)
- **Missing model files** (with download URLs)
- **Node type mismatches** (version compatibility)
- **Import failures** (broken custom nodes)
- **Permission errors**

## 🔧 Configuration

### ComfyUI Path Auto-Detection
The doctor looks for ComfyUI in these paths (in order):
1. `/workspace/runpod-slim/ComfyUI` (RunPod)
2. `/workspace/ComfyUI`
3. `~/ComfyUI`
4. `~/comfy/ComfyUI`
5. Current directory

Override with `--path /your/comfyui/path`.

### Adding to the Knowledge Base
The knowledge base is in `comfyui_doctor/knowledge/`:
- `error_db.py` — Error patterns and fixes
- `node_map.py` — Node type → git repo mapping
- `model_map.py` — Model filename → download URL mapping

PRs welcome to expand these databases!

## 🤝 Integration with Comfy-Pilot
When used alongside [Comfy-Pilot](https://github.com/ConstantineB6/Comfy-Pilot), 
Claude Code can see the ComfyUI frontend AND auto-fix workflows — the ultimate 
AI-powered ComfyUI experience.

## 📄 License

MIT

## 🙏 Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — The backbone
- [comfy-cli](https://github.com/Comfy-Org/comfy-cli) — Inspiration for CLI design
- [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) — Node registry data
- [Comfy-Pilot](https://github.com/ConstantineB6/Comfy-Pilot) — Frontend integration
