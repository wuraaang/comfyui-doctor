# ComfyUI Doctor — Research Notes
Date: 2026-02-20 04:00 CET

## Architecture Decision

### Tools that exist
1. **comfy-cli** (Comfy-Org) — install, launch, manage nodes, download models. Python/Typer.
2. **cm-cli** (ComfyUI-Manager) — install/fix/update nodes from CLI. Python.
3. **Comfy-Pilot** (ConstantineB6) — MCP server to see/edit frontend from Claude Code. 15 tools.

### What's missing (our value-add)
- **Workflow analyzer**: parse JSON, extract required nodes + models BEFORE running
- **Auto-fixer**: detect errors → apply fixes → retry automatically
- **Error knowledge base**: map error patterns → solutions
- **One-shot workflow runner**: give it any workflow → it makes it work

### Architecture: NOT a wrapper around comfy-cli
comfy-cli uses Typer and has its own env management. We don't wrap it.
Instead we call ComfyUI's HTTP API directly + use cm-cli for node management.

**Our CLI = workflow-centric, not setup-centric.**
- comfy-cli = "manage your ComfyUI installation"
- comfyui-doctor = "make this workflow work"

### Stack
- Python 3.10+
- Typer (CLI framework, same as comfy-cli)
- Rich (pretty terminal output)
- httpx or urllib (API calls)
- No external deps beyond standard lib + typer + rich

### ComfyUI API endpoints used
- `POST /prompt` — queue a workflow
- `GET /queue` — check queue status
- `GET /history` — get results
- `GET /system_stats` — GPU/VRAM info
- `GET /object_info` — all registered node types (THE KEY to validation)
- `POST /interrupt` — cancel generation
- `GET /view` — view generated images

### Key insight: `/object_info` is everything
This endpoint returns ALL registered node types with their inputs/outputs.
By comparing workflow JSON node types vs object_info, we can:
1. Find missing nodes
2. Find wrong input types
3. Validate connections
4. Suggest fixes

## Error Categories (from research)

### Category 1: Missing Custom Nodes
**Pattern**: `"XXXNode" is not a registered node type` or node appears red
**Frequency**: #1 most common error
**Fix**: Install the custom node repo that provides it
**Challenge**: Mapping node_type → git repo (ComfyUI-Manager has this DB)

### Category 2: Missing Python Dependencies
**Pattern**: `ModuleNotFoundError: No module named 'xxx'`
**Frequency**: #2 most common
**Fix**: `pip install xxx` (sometimes specific version needed)
**Common ones**:
- insightface → `pip install insightface onnxruntime-gpu`
- segment_anything → `pip install segment-anything`
- groundingdino → complex install
- mediapipe → `pip install mediapipe`
- mmcv → `pip install mmcv` or `mim install mmcv`
- ultralytics → `pip install ultralytics`
- onnxruntime → `pip install onnxruntime-gpu` (GPU) or `onnxruntime` (CPU)
- transformers → `pip install transformers`
- accelerate → `pip install accelerate`
- diffusers → `pip install diffusers`
- opencv → `pip install opencv-python`
- scipy → `pip install scipy`
- torchaudio → `pip install torchaudio`

### Category 3: Missing Models
**Pattern**: `FileNotFoundError: model.safetensors` or `checkpoint not found`
**Frequency**: #3
**Fix**: Download the model to the right folder
**Challenge**: workflow JSON has model filename but not the download URL
**Approach**: Build a mapping of common model filenames → HuggingFace/CivitAI URLs

### Category 4: CUDA Out of Memory
**Pattern**: `CUDA out of memory. Tried to allocate XXX MiB`
**Frequency**: #4
**Fixes** (in order):
1. Free VRAM: `comfy.model_management.soft_empty_cache()`
2. Lower resolution
3. Enable tiled VAE (`VAEDecodeTiled` instead of `VAEDecode`)
4. Use FP8 quantization
5. Reduce batch size
6. Use CPU offloading

### Category 5: Version/Compatibility Errors
**Pattern**: `TypeError: xxx() got an unexpected keyword argument 'yyy'`
**Frequency**: #5
**Fix**: Update the custom node, or pin specific version
**Challenge**: Hard to automate, often requires manual investigation

### Category 6: Connection/Input Type Errors
**Pattern**: `KeyError: 'xxx'` or wrong type connected
**Frequency**: Less common (usually from manual editing)
**Fix**: Reconnect nodes properly
**Approach**: Validate connection types using object_info

## Workflow JSON Structure

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 123,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    }
  }
}
```

Key: node_id → {class_type, inputs}
Inputs: either literal values or [node_id, slot_index] references

## Plan for the CLI

### Phase 1: Workflow Analyzer (PRIORITY — this is the one-shot magic)
```
comfyui-doctor run workflow.json
  1. Parse workflow JSON
  2. Extract all class_types
  3. Query /object_info for registered types
  4. Diff → find missing nodes
  5. For each missing: lookup in ComfyUI-Manager registry → install
  6. For each installed: check pip deps → install missing
  7. Check model references → download missing
  8. Queue workflow
  9. Monitor for errors
  10. If error → parse → fix → retry (max 3 attempts)
```

### Phase 2: Error Parser + Knowledge Base
Pattern matching on error output → lookup in error-db → apply fix

### Phase 3: Model Manager
Mapping model filenames → download URLs (HuggingFace, CivitAI)

### Phase 4: Integration with Comfy-Pilot MCP
When used via Claude Code, use Comfy-Pilot to see the frontend
