"""Workflow builder — create and modify ComfyUI workflows programmatically.

V3 feature: instead of just fixing existing workflows, create new ones
from scratch or modify existing ones based on templates.
"""

import json
import copy
from typing import Optional


class WorkflowBuilder:
    """Build ComfyUI API-format workflows programmatically."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._next_id = 1

    def add_node(
        self,
        class_type: str,
        inputs: Optional[dict] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Add a node to the workflow. Returns the node ID."""
        if node_id is None:
            node_id = str(self._next_id)
            self._next_id += 1
        
        self._nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs or {},
        }
        return node_id

    def connect(
        self,
        from_node: str,
        from_slot: int,
        to_node: str,
        to_input: str,
    ) -> None:
        """Connect output slot of one node to input of another."""
        if to_node not in self._nodes:
            raise ValueError(f"Target node {to_node} not found")
        self._nodes[to_node]["inputs"][to_input] = [from_node, from_slot]

    def set_input(self, node_id: str, key: str, value) -> None:
        """Set an input value on a node."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")
        self._nodes[node_id]["inputs"][key] = value

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all connections to it."""
        if node_id in self._nodes:
            del self._nodes[node_id]
        # Clean up references
        for nid, node in self._nodes.items():
            for key, val in list(node["inputs"].items()):
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == node_id:
                    del node["inputs"][key]

    def replace_node(
        self,
        old_id: str,
        new_class_type: str,
        input_mapping: Optional[dict] = None,
    ) -> str:
        """Replace a node with a different type, preserving connections.
        
        input_mapping: {old_input_name: new_input_name} for remapping inputs.
        """
        if old_id not in self._nodes:
            raise ValueError(f"Node {old_id} not found")
        
        old_node = self._nodes[old_id]
        new_inputs = {}
        
        mapping = input_mapping or {}
        for key, val in old_node["inputs"].items():
            new_key = mapping.get(key, key)
            new_inputs[new_key] = val
        
        self._nodes[old_id] = {
            "class_type": new_class_type,
            "inputs": new_inputs,
        }
        return old_id

    def build(self) -> dict:
        """Return the workflow as a ComfyUI API-format dict."""
        return copy.deepcopy(self._nodes)

    def save(self, path: str) -> None:
        """Save workflow to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.build(), f, indent=2)

    @classmethod
    def from_workflow(cls, workflow: dict) -> "WorkflowBuilder":
        """Create a builder from an existing workflow dict."""
        builder = cls()
        builder._nodes = copy.deepcopy(workflow)
        max_id = 0
        for nid in workflow:
            try:
                max_id = max(max_id, int(nid))
            except ValueError:
                pass
        builder._next_id = max_id + 1
        return builder

    @classmethod
    def from_file(cls, path: str) -> "WorkflowBuilder":
        """Create a builder from a workflow JSON file."""
        with open(path) as f:
            data = json.load(f)
        if "prompt" in data:
            data = data["prompt"]
        return cls.from_workflow(data)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        types = [n["class_type"] for n in self._nodes.values()]
        return f"WorkflowBuilder({len(types)} nodes: {', '.join(types[:5])}{'...' if len(types) > 5 else ''})"


# ── Template workflows ────────────────────────────────────────────────

def build_txt2img_sdxl(
    prompt: str,
    negative: str = "ugly, blurry, low quality",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 42,
    checkpoint: str = "sd_xl_base_1.0.safetensors",
    sampler: str = "euler",
    scheduler: str = "normal",
) -> dict:
    """Build a simple SDXL txt2img workflow."""
    b = WorkflowBuilder()
    ckpt = b.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
    pos = b.add_node("CLIPTextEncode", {"text": prompt})
    neg = b.add_node("CLIPTextEncode", {"text": negative})
    lat = b.add_node("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1})
    samp = b.add_node("KSampler", {
        "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
    })
    decode = b.add_node("VAEDecode")
    save = b.add_node("SaveImage", {"filename_prefix": "comfyui_doctor"})
    
    b.connect(ckpt, 1, pos, "clip")
    b.connect(ckpt, 1, neg, "clip")
    b.connect(ckpt, 0, samp, "model")
    b.connect(pos, 0, samp, "positive")
    b.connect(neg, 0, samp, "negative")
    b.connect(lat, 0, samp, "latent_image")
    b.connect(samp, 0, decode, "samples")
    b.connect(ckpt, 2, decode, "vae")
    b.connect(decode, 0, save, "images")
    
    return b.build()


def build_txt2video_wan(
    prompt: str,
    negative: str = "ugly, blurry",
    width: int = 832,
    height: int = 480,
    num_frames: int = 33,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = 42,
    model: str = "wan2.1_t2v_1.3B_bf16.safetensors",
) -> dict:
    """Build a Wan 2.1 text-to-video workflow."""
    b = WorkflowBuilder()
    loader = b.add_node("WanVideoModelLoader", {
        "model": model,
        "base_precision": "bf16",
        "quantization": "disabled",
        "load_device": "main_device",
    })
    text = b.add_node("WanVideoTextEncode", {
        "positive_prompt": prompt,
        "negative_prompt": negative,
    })
    sampler = b.add_node("WanVideoSampler", {
        "steps": steps, "cfg": cfg, "seed": seed,
        "shift": 5.0, "force_offload": True,
        "scheduler": "unipc",
        "riflex_freq_index": 0,
    })
    decode = b.add_node("WanVideoDecode", {
        "enable_vae_tiling": True,
        "tile_x": 272, "tile_y": 272,
        "tile_stride_x": 144, "tile_stride_y": 128,
    })
    save = b.add_node("VHS_VideoCombine", {
        "frame_rate": 16, "loop_count": 0,
        "filename_prefix": "wan_doctor",
        "format": "video/h264-mp4",
        "pingpong": False, "save_output": True,
    })
    
    b.connect(loader, 0, sampler, "model")
    b.connect(loader, 1, decode, "vae")
    b.connect(text, 0, sampler, "image_embeds")
    b.connect(sampler, 0, decode, "samples")
    b.connect(decode, 0, save, "images")
    
    return b.build()
