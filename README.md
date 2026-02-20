# 🩺 comfyui-doctor

**Make any ComfyUI workflow work in one shot.**

Drop in a workflow JSON from Reddit, CivitAI, YouTube, or anywhere — comfyui-doctor will analyze it, install missing nodes, download models, fix inputs, and run it. No manual debugging.

```bash
pip install comfyui-doctor
comfyui-doctor run workflow.json
```

## What it does

```
workflow.json → 🩺 doctor → ✅ working output
```

1. **Parses** any workflow (API format or UI format from the ComfyUI interface)
2. **Detects** missing custom nodes, models, and dependencies
3. **Installs** everything automatically (git clone + pip + aria2c downloads)
4. **Validates** all inputs (fills defaults, clamps ranges, fixes enums)
5. **Runs** the workflow on your ComfyUI instance
6. **Retries** on errors with automatic fixes (up to 3 attempts)
7. **Validates output** (detects blank/solid images)

## Features

### 🔧 One-Shot Fix
```bash
# Just works™ — installs nodes, downloads models, fixes inputs, runs it
comfyui-doctor run workflow.json
```

### 🔍 Analyze Without Running
```bash
# See what's missing without touching anything
comfyui-doctor analyze workflow.json
```

### ⚡ Speed Optimizer
```bash
# Get optimization suggestions (FP8, tiling, steps, resolution)
comfyui-doctor optimize workflow.json
```

### 🎨 Create From Template
```bash
# Generate a workflow from scratch
comfyui-doctor create txt2img-sdxl -p "a cat in space" -o my_workflow.json
comfyui-doctor create txt2video-wan -p "ocean waves at sunset" -o video.json
```

### 🔎 Node Lookup
```bash
# Find any node type across 30,000+ known types
comfyui-doctor lookup "IPAdapter"
```

### 📊 System Status
```bash
# Check ComfyUI health
comfyui-doctor status
```

## Handles Real-World Workflows

comfyui-doctor is tested against **real workflows shared by the community**, not just synthetic tests:

| Test | Source | Format | Nodes | Result |
|------|--------|--------|-------|--------|
| SDXL Advanced | cubiq/ComfyUI_Workflows | UI | 19 | ✅ One shot |
| SDXL text g-l | cubiq/ComfyUI_Workflows | UI | 12 | ✅ One shot |
| Wan 2.1 T2V | Custom video | API | 7 | ✅ One shot |
| Multi-node pack | Mixed custom nodes | API | 8 | ✅ One shot |
| ControlNet | comfyui_controlnet_aux | API | 6 | ✅ One shot |
| WAS filters | was-node-suite | API | 5 | ✅ One shot |
| Animated WEBP | SaveAnimatedWEBP | API | 5 | ✅ One shot |

**16/16 test workflows passing** (12 image + 1 animation + 1 video + 2 wild UI-format)

## UI Format Support

99% of workflows shared online are in ComfyUI's **UI format** (not the API format). comfyui-doctor handles both:

- Automatically detects UI vs API format
- Converts UI → API using `/object_info` for accurate widget mapping
- Handles `PrimitiveNode`, `Note`, `Reroute` (UI-only nodes)
- Propagates PrimitiveNode values to connected nodes

## Smart Auto-Fix

| Problem | Fix |
|---------|-----|
| Missing custom node | Auto-install via git clone (40+ curated + 30,000 from ComfyUI-Manager) |
| Missing pip dependency | `pip install` from requirements.txt |
| Missing model | Download via aria2c (16 connections) or wget |
| Missing required input | Fill with default from `/object_info` |
| Value out of range | Clamp to min/max |
| Wrong enum case | Fix case or use first valid option |
| Node unavailable | Replace with equivalent built-in node |
| Blank/solid output | Detect and report (quality validation) |

## Node Replacement

When a custom node is truly unavailable (repo deleted, archived, incompatible), the doctor can replace it with an equivalent:

- `Efficient Loader` → `CheckpointLoaderSimple`
- `KSampler (Efficient)` → `KSampler`
- `SaveImageWithMetaData` → `SaveImage`
- `BNK_CLIPTextEncodeAdvanced` → `CLIPTextEncode`
- And more...

## Installation

```bash
pip install comfyui-doctor
```

Or from source:
```bash
git clone https://github.com/wuraaang/comfyui-doctor
cd comfyui-doctor
pip install -e .
```

### Requirements

- Python 3.10+
- A running ComfyUI instance (default: `http://127.0.0.1:8188`)
- `aria2c` recommended for fast model downloads (falls back to `wget`)

## CLI Reference

```
comfyui-doctor run <workflow.json>        # 🚀 Run with auto-fix (the magic)
comfyui-doctor analyze <workflow.json>    # 🔍 Analyze without running
comfyui-doctor fix <workflow.json>        # 🔧 Apply fixes without running
comfyui-doctor optimize <workflow.json>   # ⚡ Speed/memory suggestions
comfyui-doctor create <template>          # 🎨 Create from template
comfyui-doctor diagnose                   # 🏥 Full system diagnosis
comfyui-doctor status                     # 📊 Check ComfyUI status
comfyui-doctor nodes                      # 📦 List installed nodes
comfyui-doctor lookup <query>             # 🔎 Search 30K+ node types
comfyui-doctor version                    # ℹ️  Version info
```

## How It Works

```
                    ┌─────────────────┐
                    │  workflow.json   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Parse & Detect  │ UI or API format?
                    │  Format          │ Convert if needed
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Query ComfyUI   │ /object_info
                    │  /object_info    │ (types, inputs, defaults)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Analyze         │ Missing nodes?
                    │  & Diff          │ Missing models?
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼─────┐  ┌────▼────────┐
     │ Install     │  │ Download   │  │ Validate    │
     │ Nodes       │  │ Models     │  │ & Fix       │
     │ (git+pip)   │  │ (aria2c)   │  │ Inputs      │
     └────────┬───┘  └──────┬─────┘  └────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Restart ComfyUI │ (if nodes installed)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Queue Prompt    │ Send to ComfyUI
                    │  & Wait          │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
              ┌─────│  Success?        │─────┐
              │ NO  └──────────────────┘ YES │
              │                              │
     ┌────────▼───┐                 ┌────────▼───────┐
     │ Parse Error │                │ Validate Output │
     │ & Auto-Fix  │                │ (quality check) │
     │ Retry (x3)  │                └────────────────┘
     └─────────────┘
```

## Knowledge Base

- **40+ curated** node → package mappings
- **30,000+ types** from ComfyUI-Manager fallback (cached 24h)
- **40+ model** download URLs (SDXL, SD1.5, Wan 2.1, ControlNet, IP-Adapter...)
- **35+ error patterns** with regex matching and auto-fix rules
- **20+ node replacements** for unavailable custom nodes

## License

MIT — use it, fork it, ship it.

## Credits

Built by [Bebop Studio](https://github.com/wuraaang) with ❤️ and ComfyUI frustration.
