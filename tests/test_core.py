"""Tests for comfyui-doctor core modules."""

import json
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comfyui_doctor.core.workflow import (
    load_workflow,
    analyze_workflow,
    validate_inputs,
    auto_fix_inputs,
    extract_node_types_from_json,
)
from comfyui_doctor.knowledge.error_db import match_error
from comfyui_doctor.knowledge.node_map import lookup_node_type, lookup_multiple, get_unique_repos
from comfyui_doctor.knowledge.model_map import lookup_model


def test_workflow_loading():
    """Test loading a workflow JSON."""
    path = os.path.join(os.path.dirname(__file__), "test_workflow.json")
    workflow, is_api = load_workflow(path)
    assert is_api, "Should detect API format"
    assert "3" in workflow, "Should have node 3"
    assert workflow["3"]["class_type"] == "KSampler"
    print("✅ test_workflow_loading passed")


def test_workflow_analysis():
    """Test workflow analysis."""
    path = os.path.join(os.path.dirname(__file__), "test_workflow.json")
    workflow, _ = load_workflow(path)
    
    # Without registered types (no ComfyUI running)
    analysis = analyze_workflow(workflow)
    assert len(analysis.nodes) == 10
    assert "KSampler" in analysis.required_node_types
    assert "FaceDetailer" in analysis.required_node_types
    assert "Power Lora Loader (rgthree)" in analysis.required_node_types
    assert len(analysis.missing_node_types) == 0  # No registered types → can't detect missing
    print(f"✅ test_workflow_analysis passed ({len(analysis.nodes)} nodes, {len(analysis.required_node_types)} types)")

    # With fake registered types (simulate ComfyUI with basic nodes only)
    registered = {"KSampler", "CheckpointLoaderSimple", "EmptyLatentImage", 
                  "CLIPTextEncode", "VAEDecode", "SaveImage"}
    analysis2 = analyze_workflow(workflow, registered)
    assert "FaceDetailer" in analysis2.missing_node_types
    assert "UltralyticsDetectorProvider" in analysis2.missing_node_types
    assert "Power Lora Loader (rgthree)" in analysis2.missing_node_types
    assert "KSampler" not in analysis2.missing_node_types
    print(f"✅ test_workflow_analysis_missing passed ({len(analysis2.missing_node_types)} missing)")


def test_model_references():
    """Test extraction of model references."""
    path = os.path.join(os.path.dirname(__file__), "test_workflow.json")
    workflow, _ = load_workflow(path)
    analysis = analyze_workflow(workflow)
    
    filenames = [ref["filename"] for ref in analysis.model_references]
    assert "sd_xl_base_1.0.safetensors" in filenames
    assert "bbox/face_yolov8m.pt" in filenames
    print(f"✅ test_model_references passed ({len(analysis.model_references)} refs)")


def test_connection_validation():
    """Test that connections to non-existent nodes are flagged."""
    workflow = {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["999", 0],  # node 999 doesn't exist
            }
        }
    }
    analysis = analyze_workflow(workflow)
    assert len(analysis.connection_errors) > 0
    assert "999" in analysis.connection_errors[0]
    print(f"✅ test_connection_validation passed ({len(analysis.connection_errors)} errors)")


def test_error_matching():
    """Test error pattern matching."""
    # Missing module
    matches = match_error("ModuleNotFoundError: No module named 'insightface'")
    assert len(matches) > 0
    assert matches[0].category == "missing_module"
    assert "insightface" in matches[0].pattern_name
    assert "pip install insightface" in matches[0].fix_commands[0]
    print(f"✅ test_error_matching_insightface passed")

    # Generic missing module
    matches = match_error("ModuleNotFoundError: No module named 'some_random_lib'")
    assert len(matches) > 0
    assert matches[0].fix_commands[0] == "pip install some_random_lib"
    print(f"✅ test_error_matching_generic_module passed")

    # CUDA OOM
    matches = match_error("CUDA out of memory. Tried to allocate 2.50 GiB")
    assert len(matches) > 0
    assert matches[0].category == "cuda_oom"
    print(f"✅ test_error_matching_cuda_oom passed")

    # No match
    matches = match_error("Everything is fine!")
    assert len(matches) == 0
    print(f"✅ test_error_matching_no_match passed")


def test_node_lookup():
    """Test node type to package lookup."""
    # Known node
    pkg = lookup_node_type("FaceDetailer")
    assert pkg is not None
    assert "Impact-Pack" in pkg.package_name
    assert "segment-anything" in pkg.pip_deps
    print(f"✅ test_node_lookup_known passed")

    # Unknown node
    pkg = lookup_node_type("SomeRandomNodeThatDoesntExist")
    assert pkg is None
    print(f"✅ test_node_lookup_unknown passed")

    # Multiple lookup with dedup
    types = {"FaceDetailer", "SAMLoader", "UltralyticsDetectorProvider"}
    packages = lookup_multiple(types)
    repos = get_unique_repos(packages)
    assert len(repos) == 1  # All from Impact-Pack
    print(f"✅ test_node_lookup_dedup passed (3 types → {len(repos)} repo)")


def test_model_lookup():
    """Test model filename lookup."""
    # Known model
    info = lookup_model("sd_xl_base_1.0.safetensors")
    assert info is not None
    assert info.size_gb == 6.9
    assert "huggingface" in info.url
    print(f"✅ test_model_lookup_known passed")

    # Case-insensitive
    info = lookup_model("SD_XL_BASE_1.0.SAFETENSORS")
    assert info is not None
    print(f"✅ test_model_lookup_case_insensitive passed")

    # With path prefix
    info = lookup_model("bbox/face_yolov8m.pt")
    assert info is not None
    print(f"✅ test_model_lookup_path_prefix passed")

    # Unknown model
    info = lookup_model("my_custom_model_v42.safetensors")
    assert info is None
    print(f"✅ test_model_lookup_unknown passed")


def test_extract_types():
    """Test quick type extraction."""
    path = os.path.join(os.path.dirname(__file__), "test_workflow.json")
    types = extract_node_types_from_json(path)
    assert "KSampler" in types
    assert "FaceDetailer" in types
    assert len(types) == 9  # 9 unique types (2 CLIPTextEncode count as 1)
    print(f"✅ test_extract_types passed ({len(types)} types)")


def test_validate_inputs():
    """Test input validation against object_info."""
    workflow = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 42}},
    }
    # Mock object_info with KSampler requiring more inputs
    object_info = {
        "KSampler": {
            "input": {
                "required": {
                    "seed": ["INT", {"default": 0}],
                    "steps": ["INT", {"default": 20}],
                    "cfg": ["FLOAT", {"default": 7.0}],
                    "model": ["MODEL"],  # Connection type — skipped
                    "positive": ["CONDITIONING"],  # Connection type — skipped
                }
            }
        }
    }
    errors = validate_inputs(workflow, object_info)
    # steps and cfg are missing scalar inputs, model and positive are connection types (skipped)
    assert len(errors) == 2  # steps, cfg missing
    assert any("steps" in e for e in errors)
    assert any("cfg" in e for e in errors)
    assert not any("model" in e for e in errors)  # Connection types skipped
    print(f"✅ test_validate_inputs passed ({len(errors)} errors)")


def test_auto_fix_inputs():
    """Test auto-fix: clamp values, fix enum case."""
    workflow = {
        "1": {"class_type": "TestNode", "inputs": {
            "brightness": 1.5,      # Over max of 1.0
            "contrast": -0.5,       # Under min of 0.0
            "mode": "NEAREST",      # Wrong case
            "normal_val": 0.5,      # Fine
        }},
    }
    object_info = {
        "TestNode": {
            "input": {
                "required": {
                    "brightness": ["FLOAT", {"min": 0.0, "max": 1.0}],
                    "contrast": ["FLOAT", {"min": 0.0, "max": 2.0}],
                    "mode": [["nearest", "bilinear", "bicubic"]],
                    "normal_val": ["FLOAT", {"min": 0.0, "max": 1.0}],
                }
            }
        }
    }
    fixed_wf, fixes = auto_fix_inputs(workflow, object_info)
    assert fixed_wf["1"]["inputs"]["brightness"] == 1.0  # Clamped to max
    assert fixed_wf["1"]["inputs"]["contrast"] == 0.0    # Clamped to min
    assert fixed_wf["1"]["inputs"]["mode"] == "nearest"   # Case fixed
    assert fixed_wf["1"]["inputs"]["normal_val"] == 0.5   # Untouched
    assert len(fixes) == 3
    print(f"✅ test_auto_fix_inputs passed ({len(fixes)} fixes)")


def test_error_matching_validation():
    """Test matching of validation errors."""
    # Test prompt_outputs_failed_validation
    error = 'prompt_outputs_failed_validation Required input is missing: file_format'
    matches = match_error(error)
    assert len(matches) >= 1
    assert any(m.pattern_name == "prompt_validation_missing_input" for m in matches)
    print(f"✅ test_error_matching_validation passed")
    
    # Test missing_node_type HTTP 400
    error2 = """missing_node_type Node 'FaceDetailer' not found"""
    matches2 = match_error(error2)
    assert len(matches2) >= 1
    assert any(m.category == "missing_node" for m in matches2)
    print(f"✅ test_error_matching_missing_node_http400 passed")


def test_civitai_wrapper_format():
    """B014: Test loading a CivitAI-wrapped workflow {"workflow": {"nodes": [...]}}."""
    import tempfile
    # CivitAI wraps UI-format workflows in {"workflow": {...}}
    civitai_data = {
        "workflow": {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "outputs": [
                    {"name": "MODEL", "links": [1]},
                    {"name": "CLIP", "links": [2, 3]},
                    {"name": "VAE", "links": [4]},
                ], "widgets_values": ["sd_xl_base_1.0.safetensors"]},
                {"id": 2, "type": "CLIPTextEncode", "inputs": [
                    {"name": "clip", "link": 2},
                ], "outputs": [
                    {"name": "CONDITIONING", "links": [5]},
                ], "widgets_values": ["a photo of a cat"]},
                {"id": 3, "type": "KSampler", "inputs": [
                    {"name": "model", "link": 1},
                    {"name": "positive", "link": 5},
                ], "outputs": [
                    {"name": "LATENT", "links": [6]},
                ], "widgets_values": [42, "fixed", 20, 7.0, "euler", "normal", 1.0]},
            ],
            "links": [
                [1, 1, 0, 3, 0, "MODEL"],
                [2, 1, 1, 2, 0, "CLIP"],
                [3, 1, 1, 3, 1, "CLIP"],
                [4, 1, 2, 3, 2, "VAE"],
                [5, 2, 0, 3, 1, "CONDITIONING"],
                [6, 3, 0, 4, 0, "LATENT"],
            ],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(civitai_data, f)
        tmp_path = f.name
    try:
        workflow, is_api = load_workflow(tmp_path)
        # Should detect as UI format (needs conversion)
        assert not is_api, "CivitAI wrapper should be detected as needing conversion"
        # Should have parsed nodes (excluding UI-only)
        assert len(workflow) >= 3, f"Expected at least 3 nodes, got {len(workflow)}"
        # KSampler should be present
        class_types = {nd.get("class_type") for nd in workflow.values()}
        assert "KSampler" in class_types
        assert "CheckpointLoaderSimple" in class_types
        print(f"✅ test_civitai_wrapper_format passed ({len(workflow)} nodes)")
    finally:
        os.unlink(tmp_path)


def test_reroute_multihop():
    """B015: Test multi-hop Reroute resolution (3+ levels)."""
    import tempfile
    # Workflow: CheckpointLoader → Reroute1 → Reroute2 → Reroute3 → KSampler
    ui_data = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "outputs": [
                {"name": "MODEL", "links": [1]},
            ], "widgets_values": ["sd_xl_base_1.0.safetensors"]},
            {"id": 10, "type": "Reroute", "inputs": [
                {"name": "input", "link": 1},
            ], "outputs": [
                {"name": "output", "links": [2]},
            ], "widgets_values": []},
            {"id": 11, "type": "Reroute", "inputs": [
                {"name": "input", "link": 2},
            ], "outputs": [
                {"name": "output", "links": [3]},
            ], "widgets_values": []},
            {"id": 12, "type": "Reroute", "inputs": [
                {"name": "input", "link": 3},
            ], "outputs": [
                {"name": "output", "links": [4]},
            ], "widgets_values": []},
            {"id": 2, "type": "KSampler", "inputs": [
                {"name": "model", "link": 4},
            ], "outputs": [
                {"name": "LATENT", "links": []},
            ], "widgets_values": [42, "fixed", 20, 7.0, "euler", "normal", 1.0]},
        ],
        "links": [
            [1, 1, 0, 10, 0, "MODEL"],
            [2, 10, 0, 11, 0, "MODEL"],
            [3, 11, 0, 12, 0, "MODEL"],
            [4, 12, 0, 2, 0, "MODEL"],
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ui_data, f)
        tmp_path = f.name
    try:
        workflow, is_api = load_workflow(tmp_path)
        # Reroute nodes should be stripped
        class_types = {nd.get("class_type") for nd in workflow.values()}
        assert "Reroute" not in class_types, "Reroute nodes should be stripped"
        # KSampler's model input should point to node 1 (CheckpointLoader), not a Reroute
        ksampler = None
        for nd in workflow.values():
            if nd.get("class_type") == "KSampler":
                ksampler = nd
                break
        assert ksampler is not None, "KSampler should exist"
        model_input = ksampler["inputs"].get("model")
        assert model_input is not None, "KSampler should have model input"
        assert isinstance(model_input, list), "model input should be a connection"
        assert model_input[0] == "1", f"model should connect to node 1, got {model_input[0]}"
        print(f"✅ test_reroute_multihop passed (3 hops resolved to node {model_input[0]})")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_workflow_loading()
    test_workflow_analysis()
    test_model_references()
    test_connection_validation()
    test_error_matching()
    test_node_lookup()
    test_model_lookup()
    test_extract_types()
    test_validate_inputs()
    test_auto_fix_inputs()
    test_error_matching_validation()
    test_civitai_wrapper_format()
    test_reroute_multihop()
    print("\n🎉 All tests passed!")
