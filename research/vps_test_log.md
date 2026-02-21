# VPS Test Log — Phase 1+2

**Date**: 2026-02-21
**VPS**: GPUHub Singapore, RTX 5090, 31.4 Go VRAM
**ComfyUI**: v0.14.1 at `/root/autodl-tmp/ComfyUI` (635 registered node types)
**comfyui-doctor**: `/root/autodl-tmp/comfyui-doctor` (branch: 001-oneshot-workflow-runner)

---

## Test 1: Pull + Install + Run Tests

### Steps
```bash
cd /root/autodl-tmp/comfyui-doctor
git pull origin 001-oneshot-workflow-runner
pip install anthropic>=0.25.0 pytest
python3 -m pytest tests/ -v
```

### Results: ✅ PASS
- **22/22 tests pass** (13 test_core.py + 9 test_llm.py)
- Python 3.10.12, pytest 9.0.2
- anthropic SDK installed successfully
- All LLM dataclass, client init, and Doctor integration tests pass

---

## Test 2: SDXL txt2img Workflow

### Steps
```bash
echo 'y' | python3 -m comfyui_doctor.cli run research/workflow_test_sdxl.json --path /root/autodl-tmp/ComfyUI
```

### Results: ✅ SUCCESS
- **Model detection**: `sd_xl_base_1.0.safetensors` found in model_map (6.9GB, free)
- **Download**: aria2c multi-connection download worked perfectly
- **Execution**: Queued and completed on first attempt
- **Output**: `doctor_test_sdxl_00001_.png` (1024x1024, 1.5MB) — valid image
- **Time**: ~5s generation (after model load)

### Issue Found
- Model name `sd_xl_base_1.0_0.9vae.safetensors` (common variant) is NOT in model_map
- Only `sd_xl_base_1.0.safetensors` is mapped → **LLM escalation would help here** (Phase 3)

---

## Test 3: Complex Workflow (FaceDetailer + LoRA)

### Steps
```bash
echo 'y' | python3 -m comfyui_doctor.cli run tests/test_workflow.json --path /root/autodl-tmp/ComfyUI --dry-run
```

### Results: ✅ PASS (dry-run)
- **3 missing node types detected**: FaceDetailer, UltralyticsDetectorProvider, Power Lora Loader (rgthree)
- **Package resolution**: Correctly mapped to ComfyUI-Impact-Pack (2 types) + rgthree-comfy (1 type)
- **Dedup**: 2 types → 1 repo (Impact-Pack)
- **Pip deps detected**: segment-anything, ultralytics
- **Model download**: bbox/face_yolov8m.pt (0.05GB) URL resolved
- **Already-downloaded model**: sd_xl_base_1.0.safetensors ✅ found on disk

---

## Errors Encountered / Notes for Phase 3

### E001: Model name variants not in model_map
- `sd_xl_base_1.0_0.9vae.safetensors` → marked as "unknown model"
- Common variant used by many workflows, but our curated map only has `sd_xl_base_1.0.safetensors`
- **LLM escalation opportunity**: Claude could identify this as the same base model and suggest the right download URL

### E002: Interactive prompt blocks non-interactive mode
- `Continue anyway? [y/N]:` hangs when piped through SSH
- Need a `--yes` / `-y` flag for CI/non-interactive use
- Currently workaround: `echo 'y' | comfyui-doctor run ...`

### E003: Package install (pip) uses wrong name
- `pip install '.[llm,dev]'` → package installs as "UNKNOWN" (setuptools issue)
- Workaround: install deps manually (`pip install anthropic pytest`)
- Root cause: maybe needs `setuptools>=68.0` or the `name` field isn't being picked up

### E004: `python` not available, only `python3`
- VPS (Ubuntu 22.04) doesn't have `python` alias
- Commands must use `python3` explicitly

---

## Summary

| Test | Status | Notes |
|------|--------|-------|
| 22 unit tests | ✅ PASS | All tests pass on VPS |
| SDXL simple workflow | ✅ SUCCESS | Model detected, downloaded, generated image |
| Complex workflow (dry-run) | ✅ PASS | Missing nodes + models detected correctly |
| Doctor with LLMClient | ✅ PASS | Backward compat OK, injectable OK |
| Doctor without LLMClient | ✅ PASS | Zero regression |
