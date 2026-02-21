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
    # AnimateDiff / motion models
    "animatediff_model", "motion_model_name",
    # Multi-LoRA stacks (lora_01 through lora_10)
    "lora_01", "lora_02", "lora_03", "lora_04", "lora_05",
    "lora_06", "lora_07", "lora_08", "lora_09", "lora_10",
    # IP-Adapter variants
    "adapter_name", "ip_adapter_name",
    # Generic model path / weight name
    "model_path", "weight_name",
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
    "animatediff_model": "animatediff_models",
    "motion_model_name": "animatediff_models",
    "lora_01": "loras", "lora_02": "loras", "lora_03": "loras",
    "lora_04": "loras", "lora_05": "loras", "lora_06": "loras",
    "lora_07": "loras", "lora_08": "loras", "lora_09": "loras",
    "lora_10": "loras",
    "adapter_name": "ipadapter",
    "ip_adapter_name": "ipadapter",
    "model_path": "checkpoints",
    "weight_name": "checkpoints",
}


# ── UI-only node types to strip during conversion ──────────────────────
UI_ONLY_TYPES = {
    "Note", "PrimitiveNode", "Reroute",
    "SetNode", "GetNode",  # ComfyUI-Workflow-Component
    "Group",  # UI grouping box, no API equivalent
    "Bookmark (rgthree)", "Fast Groups Muter (rgthree)",  # rgthree UI-only
}

# ── Connection types that come from wires, not widgets ─────────────────
CONNECTION_TYPES = {
    "MODEL", "CONDITIONING", "LATENT", "IMAGE", "MASK", "VAE", "CLIP",
    "CONTROL_NET", "CLIP_VISION", "CLIP_VISION_OUTPUT", "STYLE_MODEL",
    "GLIGEN", "UPSCALE_MODEL", "SAMPLER", "SIGMAS", "NOISE", "GUIDER",
    "PHOTOMAKER", "IPADAPTER", "INSIGHTFACE", "WANVIDEOMODEL", "WANVAE",
    "WANVIDEOTEXTEMBEDS", "WANVIDIMAGE_EMBEDS", "VHS_FILENAMES",
    "AUDIO", "VHS_AUDIO",
}


def _convert_ui_to_api_basic(data: dict) -> dict:
    """Convert UI-format workflow to API format (basic, no object_info).
    
    Reconstructs connections from the 'links' table and stores
    widget_values for later reconstruction via object_info.
    """
    nodes_list = data.get("nodes", [])
    links_list = data.get("links", [])
    
    # Build link lookup: link_id → (source_node_id, source_slot, target_type?)
    link_map = {}
    for link in links_list:
        # link format: [link_id, origin_id, origin_slot, target_id, target_slot, type]
        if isinstance(link, list) and len(link) >= 5:
            link_id = link[0]
            link_map[link_id] = {
                "origin_id": str(link[1]),
                "origin_slot": link[2],
                "type": link[5] if len(link) > 5 else None,
            }
    
    # ── Build PrimitiveNode + Crystools value propagation map ─────────
    # Must be built BEFORE SetNode resolution (they may chain through Set/Get)
    primitive_values = {}  # link_id → value from PrimitiveNode/Crystools widgets
    for node in nodes_list:
        ntype = node.get("type", "")
        if ntype == "PrimitiveNode" or ("Primitive" in ntype and "Crystool" in ntype):
            widgets = node.get("widgets_values", [])
            outputs = node.get("outputs", [])
            if widgets and outputs:
                for out in outputs:
                    for link_id in out.get("links", []):
                        if link_id is not None:
                            primitive_values[link_id] = widgets[0] if widgets else None
    
    # ── Resolve SetNode/GetNode routing ──────────────────────────────
    node_by_id = {str(n.get("id", "")): n for n in nodes_list}
    
    set_sources = {}  # name → (origin_id, origin_slot) or {"_value": val}
    for node in nodes_list:
        if node.get("type") == "SetNode":
            name = (node.get("widgets_values") or [""])[0]
            for inp in node.get("inputs", []):
                link_id = inp.get("link")
                if link_id is not None:
                    if link_id in primitive_values:
                        set_sources[name] = {"_value": primitive_values[link_id]}
                    elif link_id in link_map:
                        lnk = link_map[link_id]
                        set_sources[name] = (lnk["origin_id"], lnk["origin_slot"])
                    break
    
    get_sources = {}  # get_node_id → (origin_id, origin_slot) or {"_value": val}
    for node in nodes_list:
        if node.get("type") == "GetNode":
            name = (node.get("widgets_values") or [""])[0]
            if name in set_sources:
                get_sources[str(node.get("id", ""))] = set_sources[name]
    
    api_format = {}
    for node in nodes_list:
        node_id = str(node.get("id", ""))
        node_type = node.get("type", "")
        widgets_values = node.get("widgets_values", [])
        
        # Skip UI-only nodes (SetNode/GetNode are resolved above)
        if node_type in UI_ONLY_TYPES:
            continue
        # Also skip Crystools primitives (values propagated via links)
        if "Primitive" in node_type and "Crystool" in node_type:
            continue
        
        inputs = {}
        
        # Reconstruct connections from node inputs
        node_inputs = node.get("inputs", [])  # UI inputs with links
        for inp in node_inputs:
            inp_name = inp.get("name", "")
            link_id = inp.get("link")
            if link_id is not None:
                # Check if this comes from a PrimitiveNode/Crystools
                if link_id in primitive_values:
                    inputs[inp_name] = primitive_values[link_id]
                elif link_id in link_map:
                    lnk = link_map[link_id]
                    origin_id = lnk["origin_id"]
                    origin_slot = lnk["origin_slot"]
                    
                    # Resolve GetNode → original source
                    if origin_id in get_sources:
                        src = get_sources[origin_id]
                        if isinstance(src, dict) and "_value" in src:
                            # Literal value from Crystools/Primitive via Set/Get
                            inputs[inp_name] = src["_value"]
                            continue
                        else:
                            origin_id, origin_slot = src
                    
                    # Resolve multi-hop Reroute chains
                    seen = set()
                    while True:
                        origin_node = node_by_id.get(origin_id)
                        if not origin_node or origin_node.get("type") != "Reroute":
                            break
                        if origin_id in seen:
                            break  # Prevent infinite loops
                        seen.add(origin_id)
                        # Follow the Reroute's input link
                        reroute_inputs = origin_node.get("inputs", [])
                        resolved = False
                        for ri in reroute_inputs:
                            rlink_id = ri.get("link")
                            if rlink_id is not None and rlink_id in link_map:
                                rlnk = link_map[rlink_id]
                                origin_id = rlnk["origin_id"]
                                origin_slot = rlnk["origin_slot"]
                                resolved = True
                                break
                        if not resolved:
                            break

                    inputs[inp_name] = [origin_id, origin_slot]

        api_format[node_id] = {
            "class_type": node_type,
            "inputs": inputs,
            "_ui_widgets": widgets_values,
            "_ui_node": node,
        }
    
    return api_format


def convert_ui_to_api(data: dict, object_info: dict) -> dict:
    """Convert UI-format workflow to full API format using /object_info.
    
    This is the accurate conversion that maps widget_values to the correct
    input names using the node type definitions from ComfyUI's /object_info.
    """
    api_format = _convert_ui_to_api_basic(data)
    
    for node_id, node_data in api_format.items():
        class_type = node_data["class_type"]
        widgets = node_data.get("_ui_widgets", [])
        ui_node = node_data.get("_ui_node", {})
        
        if class_type not in object_info or not widgets:
            continue
        
        type_info = object_info[class_type]
        required = type_info.get("input", {}).get("required", {})
        optional = type_info.get("input", {}).get("optional", {})
        
        # Merge required + optional, preserving order
        all_inputs = {}
        all_inputs.update(required)
        all_inputs.update(optional)
        
        # Build list of widget input names (skip connection-type inputs)
        # Connection inputs come from wires, not widget values
        connected_inputs = set(node_data["inputs"].keys())
        
        widget_names = []
        for inp_name, inp_def in all_inputs.items():
            # Skip if already connected via wire
            if inp_name in connected_inputs:
                continue
            # Skip connection types
            if isinstance(inp_def, list) and len(inp_def) >= 1:
                type_str = inp_def[0] if isinstance(inp_def[0], str) else ""
                if type_str in CONNECTION_TYPES:
                    continue
            widget_names.append(inp_name)
        
        # Map widget values to input names
        wi = 0
        for inp_name in widget_names:
            if wi >= len(widgets):
                break
            val = widgets[wi]
            # Skip "control_after_generate" phantom widgets
            if val in ("fixed", "increment", "decrement", "randomize"):
                wi += 1
                # These phantom values follow seed/noise_seed inputs
                continue
            node_data["inputs"][inp_name] = val
            wi += 1
            # Check if next value is a control_after_generate for seed-like inputs
            if inp_name in ("seed", "noise_seed") and wi < len(widgets):
                next_val = widgets[wi]
                if next_val in ("fixed", "increment", "decrement", "randomize"):
                    wi += 1  # skip it
        
        # Clean up internal keys
        node_data.pop("_ui_widgets", None)
        node_data.pop("_ui_node", None)
    
    return api_format


def load_workflow(path: str) -> tuple[dict, bool]:
    """Load workflow JSON. Returns (workflow_dict, is_api_format).
    
    Supports both API format (node_id → {class_type, inputs}) 
    and UI format (nodes array with type field).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # API format: top-level keys are node IDs mapping to {class_type, inputs}
    if isinstance(data, dict):
        # Check if it's wrapped in a "prompt" key (ComfyUI export)
        if "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]

        # Check if it's wrapped in a "workflow" key (CivitAI format)
        if "workflow" in data and isinstance(data["workflow"], dict):
            inner = data["workflow"]
            # Could be UI format (has "nodes") or API format (has "class_type" values)
            if "nodes" in inner:
                api_format = _convert_ui_to_api_basic(inner)
                return api_format, False
            sample = next(iter(inner), None)
            if sample and isinstance(inner.get(sample), dict) and "class_type" in inner[sample]:
                return inner, True

        # Check if it looks like API format
        sample_key = next(iter(data), None)
        if sample_key and isinstance(data.get(sample_key), dict):
            if "class_type" in data[sample_key]:
                return data, True

    # UI format: has a "nodes" array
    if isinstance(data, dict) and "nodes" in data:
        api_format = _convert_ui_to_api_basic(data)
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

            # Embedding detection in prompt text
            if isinstance(value, str) and "embedding:" in value:
                for emb_match in re.finditer(r"embedding:([^\s,\)]+)", value):
                    emb_name = emb_match.group(1)
                    # Add .safetensors if no extension
                    if "." not in emb_name:
                        emb_name += ".safetensors"
                    ref = {
                        "filename": emb_name,
                        "input_key": key,
                        "node_id": node_id,
                        "node_type": class_type,
                        "model_folder": "embeddings",
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
    Skips inputs that expect a node connection type (IMAGE, MODEL, etc.)
    """
    errors = []
    # Types that come from node connections, not user values
    CONNECTION_TYPES = {
        "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
        "CONTROL_NET", "UPSCALE_MODEL", "SAMPLER", "SIGMAS", "NOISE",
        "GUIDER", "SEGS", "BBOX_DETECTOR", "SAM_MODEL", "CLIP_VISION",
        "CLIP_VISION_OUTPUT", "STYLE_MODEL", "GLIGEN", "DETAILER_PIPE",
        "BASIC_PIPE", "IPADAPTER", "INSIGHTFACE",
    }
    
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
            if input_name in inputs:
                continue
            
            # Check if this is a connection type (provided by wires, not values)
            if isinstance(input_spec, (list, tuple)) and input_spec:
                spec_type = input_spec[0]
                if isinstance(spec_type, str) and spec_type.upper() in CONNECTION_TYPES:
                    continue  # Expected from a node connection
            
            errors.append(
                f"Node #{node_id} ({class_type}): missing required input '{input_name}'"
            )
    
    return errors


def auto_fix_inputs(
    workflow: dict,
    object_info: dict,
) -> tuple[dict, list[str]]:
    """Auto-fix workflow inputs: clamp values, fill defaults, fix enum case.
    
    Returns (fixed_workflow, list_of_fixes_applied).
    Modifies workflow in place.
    """
    fixes = []
    # Types that come from node connections, not user values
    CONNECTION_TYPES = {
        "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
        "CONTROL_NET", "UPSCALE_MODEL", "SAMPLER", "SIGMAS", "NOISE",
        "GUIDER", "SEGS", "BBOX_DETECTOR", "SAM_MODEL", "CLIP_VISION",
        "CLIP_VISION_OUTPUT", "STYLE_MODEL", "GLIGEN", "DETAILER_PIPE",
        "BASIC_PIPE", "IPADAPTER", "INSIGHTFACE",
    }
    
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        if class_type not in object_info:
            continue
        
        node_info = object_info[class_type]
        required_inputs = node_info.get("input", {}).get("required", {})
        optional_inputs = node_info.get("input", {}).get("optional", {})
        all_inputs = {}
        all_inputs.update(required_inputs)
        all_inputs.update(optional_inputs)
        
        for input_name, input_spec in all_inputs.items():
            if not isinstance(input_spec, (list, tuple)) or len(input_spec) < 1:
                continue
            
            spec_type = input_spec[0]
            spec_config = input_spec[1] if len(input_spec) > 1 else {}
            if not isinstance(spec_config, dict):
                spec_config = {}
            
            # === FILL MISSING REQUIRED INPUTS WITH DEFAULTS ===
            if input_name not in inputs and input_name in required_inputs:
                # Skip connection types — can't auto-fill those
                if isinstance(spec_type, str) and spec_type.upper() in CONNECTION_TYPES:
                    continue
                
                # Try to use default value
                if "default" in spec_config:
                    inputs[input_name] = spec_config["default"]
                    fixes.append(
                        f"Node #{node_id} ({class_type}): filled '{input_name}' "
                        f"with default {spec_config['default']}"
                    )
                elif isinstance(spec_type, list) and spec_type:
                    # Enum/combo — use first option
                    inputs[input_name] = spec_type[0]
                    fixes.append(
                        f"Node #{node_id} ({class_type}): filled '{input_name}' "
                        f"with first option '{spec_type[0]}'"
                    )
                continue
            
            if input_name not in inputs:
                continue
            
            value = inputs[input_name]
            
            # Skip connection references
            if isinstance(value, list) and len(value) == 2:
                continue
            
            # === CLAMP NUMERIC VALUES ===
            if isinstance(value, (int, float)):
                min_val = spec_config.get("min")
                max_val = spec_config.get("max")
                
                if max_val is not None and value > max_val:
                    old_val = value
                    inputs[input_name] = type(value)(max_val)
                    fixes.append(
                        f"Node #{node_id} ({class_type}): clamped '{input_name}' "
                        f"from {old_val} to max {max_val}"
                    )
                elif min_val is not None and value < min_val:
                    old_val = value
                    inputs[input_name] = type(value)(min_val)
                    fixes.append(
                        f"Node #{node_id} ({class_type}): clamped '{input_name}' "
                        f"from {old_val} to min {min_val}"
                    )
            
            # === FIX ENUM VALUES (wrong string for a combo input) ===
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
                    else:
                        # Use first available option
                        inputs[input_name] = spec_type[0]
                        fixes.append(
                            f"Node #{node_id} ({class_type}): '{input_name}' value "
                            f"'{value}' invalid, using '{spec_type[0]}'"
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
