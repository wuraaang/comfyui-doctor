# VPS Test Log — Phase 1+2

**Date**: 2026-02-21
**VPS**: GPUHub Singapore, RTX 5090, 31.4 Go VRAM
**ComfyUI**: v0.14.1 at `/root/autodl-tmp/ComfyUI`
**comfyui-doctor**: `/root/autodl-tmp/comfyui-doctor`

---

## Test 1: Pull + Install + Run Tests

### Steps
```bash
cd /root/autodl-tmp/comfyui-doctor
git pull origin 001-oneshot-workflow-runner
pip install -e .[llm,dev]
pytest tests/ -v
```

### Results
_(to be filled during VPS test)_

---

## Test 2: SDXL txt2img Workflow

### Steps
```bash
cd /root/autodl-tmp/comfyui-doctor
python -m comfyui_doctor.cli run research/workflow_test_sdxl.json --comfyui-path /root/autodl-tmp/ComfyUI
```

### Expected
- Detect model `sd_xl_base_1.0_0.9vae.safetensors`
- Download if missing (~6.9GB)
- Queue workflow
- Generate 1024x1024 image
- Output in ComfyUI/output/

### Results
_(to be filled during VPS test)_

---

## Test 3: Complex Workflow (custom nodes)

### Steps
Test with workflow that uses FaceDetailer, ControlNet, etc.

### Results
_(to be filled during VPS test)_

---

## Errors Encountered

_(document all errors for Phase 3 LLM escalation design)_
