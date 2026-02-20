"""Workflow parser and analyzer — the heart of comfyui-doctor."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WorkflowNode:
    """Parsed node from a workflow JSON."""
    node_id: str
    class_type: str
    inputs: dict = field(default_factory=dict)
    # Extracted references to other nodes: [(target_node_id, slot)]
    connections: list = field(default_factory=list)
    # Model references found in inputs
    model_refs: list = field(default_factory=list)


@dataclass
class WorkflowAnalysis:
    """Result of analyzing a workflow."""
    nodes: list[WorkflowNode]
    required_node_types: set[str]
    missing_node_types: set[str]
    model_references: list[dict]  # [{filename, input_key, node_id, node_type}]
    connection_errors: list[str]
    is_api_format: bool


# Known input keys that reference model files
MODEL_INPUT_KEYS = {
    "ckpt_name", "checkpoint_name",
    "lora_name",
    "vae_name",
    "control_net_name", "controlnet_name",
    "clip_name",
    "unet_name",
    "model_name",
    "upscale_model",
    "sam_model_name",
    "bbox_detector", "segm_detector",
    "instantid_file",
    "ipadapter_file",
    "ip_adapter_file",
    "pulid_file",
    "insightface_model",
}

# Model type mapping (input key → model subfolder)
MODEL_TYPE_MAP = {
    "ckpt_name": "checkpoints",
    "checkpoint_name": "checkpoints",
    "lora_name": "loras",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    "clip_name": "clip",
    "unet_name": "unet",
    "model_name": "checkpoints",
    "upscale_model": "upscale_models",
    "sam_model_name": "sams",
    "bbox_detector": "ultralytics/bbox",
    "segm_detector": "ultralytics/segm",
}


def load_workflow(path: str) -> tuple[dict, bool]:
    """Load workflow JSON. Returns (workflow_dict, is_api_format).
    
    Supports both API format (node_id → {class_type, inputs}) 
    and UI format (nodes array with type field).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # API format: top-level keys are node IDs mapping to {class_type, inputs}
    if isinstance(data, dict):
        # Check if it's wrapped in a "prompt" key
        if "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]
        
        # Check if it looks like API format
        sample_key = next(iter(data), None)
        if sample_key and isinstance(data.get(sample_key), dict):
            if "class_type" in data[sample_key]:
                return data, True

    # UI format: has a "nodes" array
    if isinstance(data, dict) and "nodes" in data:
        # Convert UI format to API format
        api_format = {}
        for node in data["nodes"]:
            node_id = str(node.get("id", ""))
            widgets_values = node.get("widgets_values", [])
            node_type = node.get("type", "")
            
            # UI format doesn't have clean inputs like API format
            # We store what we can
            api_format[node_id] = {
                "class_type": node_type,
                "inputs": {},  # Would need object_info to reconstruct
                "_ui_widgets": widgets_values,
                "_ui_node": node,
            }
        return api_format, False

    raise ValueError(
        "Unrecognized workflow format. Expected ComfyUI API format or UI format."
    )


def analyze_workflow(
    workflow: dict,
    registered_types: Optional[set[str]] = None,
) -> WorkflowAnalysis:
    """Analyze a workflow and find issues.
    
    Args:
        workflow: API-format workflow dict
        registered_types: Set of node types registered in ComfyUI.
                         If None, skip missing-type detection.
    """
    nodes = []
    required_types = set()
    model_refs = []
    connection_errors = []
    all_node_ids = set(workflow.keys())

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "UNKNOWN")
        inputs = node_data.get("inputs", {})
        required_types.add(class_type)

        connections = []
        node_model_refs = []

        for key, value in inputs.items():
            # Connection reference: [node_id, slot_index]
            if isinstance(value, list) and len(value) == 2:
                ref_id = str(value[0])
                connections.append((ref_id, value[1]))
                
                # Validate connection target exists
                if ref_id not in all_node_ids:
                    connection_errors.append(
                        f"Node {node_id} ({class_type}): input '{key}' "
                        f"references non-existent node {ref_id}"
                    )

            # Model file reference
            if key.lower() in MODEL_INPUT_KEYS and isinstance(value, str):
                model_type = MODEL_TYPE_MAP.get(key.lower(), "unknown")
                ref = {
                    "filename": value,
                    "input_key": key,
                    "node_id": node_id,
                    "node_type": class_type,
                    "model_folder": model_type,
                }
                node_model_refs.append(ref)
                model_refs.append(ref)

        nodes.append(WorkflowNode(
            node_id=node_id,
            class_type=class_type,
            inputs=inputs,
            connections=connections,
            model_refs=node_model_refs,
        ))

    # Find missing node types
    missing_types = set()
    if registered_types is not None:
        missing_types = required_types - registered_types

    return WorkflowAnalysis(
        nodes=nodes,
        required_node_types=required_types,
        missing_node_types=missing_types,
        model_references=model_refs,
        connection_errors=connection_errors,
        is_api_format=True,
    )


def validate_inputs(
    workflow: dict,
    object_info: dict,
) -> list[str]:
    """Validate workflow inputs against ComfyUI's /object_info.
    
    Returns list of validation error strings.
    Checks: all required inputs (non-optional) are provided.
    """
    errors = []
    
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        if class_type not in object_info:
            continue  # Missing node — handled elsewhere
        
        node_info = object_info[class_type]
        required_inputs = node_info.get("input", {}).get("required", {})
        
        for input_name, input_spec in required_inputs.items():
            if input_name not in inputs:
                # Check if this input might be provided via a connection (list ref)
                # Connections show up as [node_id, slot] in inputs
                errors.append(
                    f"Node #{node_id} ({class_type}): missing required input '{input_name}'"
                )
    
    return errors


def auto_fix_inputs(
    workflow: dict,
    object_info: dict,
) -> tuple[dict, list[str]]:
    """Auto-fix workflow inputs: clamp values to allowed ranges, fill defaults.
    
    Returns (fixed_workflow, list_of_fixes_applied).
    Modifies workflow in place.
    """
    fixes = []
    
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        if class_type not in object_info:
            continue
        
        node_info = object_info[class_type]
        all_inputs = {}
        all_inputs.update(node_info.get("input", {}).get("required", {}))
        all_inputs.update(node_info.get("input", {}).get("optional", {}))
        
        for input_name, input_spec in all_inputs.items():
            if input_name not in inputs:
                continue
            
            value = inputs[input_name]
            
            # Skip connection references
            if isinstance(value, list) and len(value) == 2:
                continue
            
            # input_spec is typically: [type_or_list, {config}] or [type_or_list]
            if not isinstance(input_spec, (list, tuple)) or len(input_spec) < 1:
                continue
            
            spec_type = input_spec[0]
            spec_config = input_spec[1] if len(input_spec) > 1 else {}
            
            if not isinstance(spec_config, dict):
                continue
            
            # Clamp numeric values
            if isinstance(value, (int, float)):
                min_val = spec_config.get("min")
                max_val = spec_config.get("max")
                
                if max_val is not None and value > max_val:
                    old_val = value
                    inputs[input_name] = max_val
                    fixes.append(
                        f"Node #{node_id} ({class_type}): clamped '{input_name}' "
                        f"from {old_val} to max {max_val}"
                    )
                elif min_val is not None and value < min_val:
                    old_val = value
                    inputs[input_name] = min_val
                    fixes.append(
                        f"Node #{node_id} ({class_type}): clamped '{input_name}' "
                        f"from {old_val} to min {min_val}"
                    )
            
            # Fix enum values (wrong string for a combo input)
            if isinstance(spec_type, list) and isinstance(value, str):
                if value not in spec_type and spec_type:
                    # Try case-insensitive match
                    lower_map = {s.lower(): s for s in spec_type}
                    if value.lower() in lower_map:
                        inputs[input_name] = lower_map[value.lower()]
                        fixes.append(
                            f"Node #{node_id} ({class_type}): fixed case of '{input_name}' "
                            f"from '{value}' to '{lower_map[value.lower()]}'"
                        )
    
    return workflow, fixes


def extract_node_types_from_json(path: str) -> set[str]:
    """Quick extraction of class_types without full analysis."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "prompt" in data:
        data = data["prompt"]
    
    types = set()
    if isinstance(data, dict):
        for node_data in data.values():
            if isinstance(node_data, dict) and "class_type" in node_data:
                types.add(node_data["class_type"])
    
    return types
