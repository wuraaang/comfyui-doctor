"""The Doctor — core auto-fix engine for ComfyUI workflows.

This is the brain of comfyui-doctor. It:
1. Analyzes a workflow BEFORE running it
2. Detects missing nodes, models, dependencies
3. Installs what's needed
4. Runs the workflow
5. If it fails, parses the error and fixes it
6. Retries (up to max_retries)
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from .api import ComfyAPI
from .workflow import (
    load_workflow,
    convert_ui_to_api,
    analyze_workflow,
    validate_inputs,
    auto_fix_inputs,
    WorkflowAnalysis,
)
from ..knowledge.error_db import match_error, ErrorMatch
from ..knowledge.node_map import (
    lookup_multiple,
    get_unique_repos,
    get_all_pip_deps,
    NodePackage,
)
from ..knowledge.model_map import lookup_model, ModelInfo
from ..knowledge.node_replacements import find_replacement, get_removable_types

console = Console()


@dataclass
class FixAction:
    """A fix action to be applied."""
    description: str
    commands: list[str]
    category: str  # install_node, install_pip, download_model, config_change
    auto: bool = True  # Can be applied automatically


@dataclass
class DoctorReport:
    """Report from analyzing/running a workflow."""
    workflow_path: str
    total_nodes: int = 0
    required_types: int = 0
    missing_nodes: list[str] = field(default_factory=list)
    missing_models: list[dict] = field(default_factory=list)
    known_models: list[dict] = field(default_factory=list)  # Models we know where to download
    unknown_models: list[dict] = field(default_factory=list)  # Models we can't find
    connection_errors: list[str] = field(default_factory=list)
    fixes_needed: list[FixAction] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    success: bool = False
    attempts: int = 0


class Doctor:
    """The ComfyUI Doctor — makes workflows work."""

    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        comfyui_path: Optional[str] = None,
        max_retries: int = 3,
        auto_fix: bool = True,
        dry_run: bool = False,
    ):
        self.api = ComfyAPI(url=comfyui_url)
        self.comfyui_path = comfyui_path or self._detect_comfyui_path()
        self.max_retries = max_retries
        self.auto_fix = auto_fix
        self.dry_run = dry_run

    def _detect_comfyui_path(self) -> str:
        """Try to auto-detect ComfyUI installation path."""
        # Common paths
        candidates = [
            "/workspace/runpod-slim/ComfyUI",  # RunPod
            "/workspace/ComfyUI",
            os.path.expanduser("~/ComfyUI"),
            os.path.expanduser("~/comfy/ComfyUI"),
            "/opt/ComfyUI",
        ]
        for p in candidates:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "main.py")):
                return p
        return "."

    @property
    def custom_nodes_path(self) -> str:
        return os.path.join(self.comfyui_path, "custom_nodes")

    @property
    def models_path(self) -> str:
        return os.path.join(self.comfyui_path, "models")

    # ── ComfyUI restart ───────────────────────────────────────────────

    def restart_comfyui(self) -> bool:
        """Restart ComfyUI to load newly installed nodes.
        
        Kills the current process and relaunches it.
        Returns True if ComfyUI comes back up.
        """
        console.print("\n🔄 [bold yellow]Restarting ComfyUI to load new nodes...[/bold yellow]")
        
        if self.dry_run:
            console.print("  [dim]DRY RUN: would restart ComfyUI[/dim]")
            return True
        
        # Find and kill the current ComfyUI process
        try:
            result = subprocess.run(
                "pgrep -f 'main.py.*--listen'",
                shell=True, capture_output=True, text=True,
            )
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    subprocess.run(f"kill {pid.strip()}", shell=True, timeout=5)
                    console.print(f"  🔪 Killed PID {pid.strip()}")
        except Exception as e:
            console.print(f"  ⚠️  Could not kill ComfyUI: {e}")
        
        time.sleep(3)
        
        # Relaunch
        try:
            subprocess.Popen(
                f"cd {self.comfyui_path} && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /tmp/comfyui.log 2>&1 &",
                shell=True,
            )
            console.print("  🚀 ComfyUI relaunching...")
        except Exception as e:
            console.print(f"  ❌ Failed to relaunch: {e}")
            return False
        
        # Wait for it to come back up
        console.print("  ⏳ Waiting for ComfyUI to start...")
        for i in range(60):  # Wait up to 60 seconds
            time.sleep(2)
            if self.api.ping():
                # Extra wait for all nodes to register
                time.sleep(5)
                types = self.api.registered_node_types()
                console.print(f"  ✅ ComfyUI is back! {len(types)} node types registered")
                return True
            if i % 5 == 0 and i > 0:
                console.print(f"  ⏳ Still waiting... ({i*2}s)")
        
        console.print("  ❌ ComfyUI did not come back after 120s")
        return False

    # ── Pre-flight analysis ───────────────────────────────────────────

    def analyze(self, workflow_path: str) -> DoctorReport:
        """Analyze a workflow and report everything that's needed."""
        report = DoctorReport(workflow_path=workflow_path)

        # 1. Load workflow
        console.print(f"\n🔍 [bold]Analyzing workflow:[/bold] {workflow_path}")
        try:
            workflow, is_api = load_workflow(workflow_path)
        except Exception as e:
            report.errors_encountered.append(f"Failed to load workflow: {e}")
            console.print(f"  ❌ Failed to load: {e}")
            return report

        # 2. Get registered node types from ComfyUI
        registered = set()
        object_info_data = {}
        if self.api.ping():
            console.print("  🟢 ComfyUI is running")
            registered = self.api.registered_node_types()
            console.print(f"  📦 {len(registered)} node types registered")
            if not is_api:
                # Convert UI → API using object_info for accurate widget mapping
                console.print("  🔄 UI format detected — converting to API format")
                object_info_data = self.api.object_info() or {}
                workflow = convert_ui_to_api(
                    json.loads(Path(workflow_path).read_text(encoding="utf-8")),
                    object_info_data,
                )
                is_api = True  # Now it's API format
                console.print(f"  ✅ Converted {len(workflow)} nodes to API format")
        else:
            console.print("  🔴 ComfyUI is not reachable — skipping node validation")

        # 3. Analyze
        analysis = analyze_workflow(workflow, registered if registered else None)
        report.total_nodes = len(analysis.nodes)
        report.required_types = len(analysis.required_node_types)
        report.connection_errors = analysis.connection_errors

        console.print(f"  📊 {report.total_nodes} nodes, {report.required_types} unique types")

        # 4. Missing nodes
        if analysis.missing_node_types:
            report.missing_nodes = sorted(analysis.missing_node_types)
            console.print(f"  ❌ [red]{len(report.missing_nodes)} missing node types:[/red]")
            
            # Look up packages
            packages = lookup_multiple(analysis.missing_node_types)
            repos = get_unique_repos(packages)
            pip_deps = get_all_pip_deps(packages)
            unknown_types = [ct for ct, pkg in packages.items() if pkg is None]

            for ct, pkg in sorted(packages.items()):
                if pkg:
                    console.print(f"     🔧 {ct} → [cyan]{pkg.package_name}[/cyan]")
                else:
                    console.print(f"     ❓ {ct} → [yellow]unknown package[/yellow]")

            # Generate fix actions
            for url, pkg in repos.items():
                # Clone the repo + install its requirements.txt if it exists
                clone_cmd = f"cd {self.custom_nodes_path} && git clone {pkg.repo_url}"
                req_file = f"{self.custom_nodes_path}/{pkg.package_name}/requirements.txt"
                install_cmd = (
                    f"{clone_cmd} ; "
                    f"test -f {req_file} && pip install -r {req_file} || true"
                )
                report.fixes_needed.append(FixAction(
                    description=f"Install {pkg.package_name}",
                    commands=[install_cmd],
                    category="install_node",
                ))
            
            for dep in pip_deps:
                report.fixes_needed.append(FixAction(
                    description=f"Install pip: {dep}",
                    commands=[f"pip install {dep}"],
                    category="install_pip",
                ))
            
            if unknown_types:
                # Try node replacements for truly unknown types
                replaceable = []
                still_unknown = []
                for ut in unknown_types:
                    repl = find_replacement(ut)
                    if repl:
                        replaceable.append((ut, repl))
                    else:
                        still_unknown.append(ut)
                
                for orig, repl in replaceable:
                    if repl.replacement in ("__REMOVE__", "__PASSTHROUGH__"):
                        console.print(f"     🗑️  {orig} → [dim]removed (UI-only)[/dim]")
                        # Remove from workflow
                        nodes_to_remove = [nid for nid, nd in workflow.items()
                                          if nd.get("class_type") == orig]
                        for nid in nodes_to_remove:
                            del workflow[nid]
                        report.fixes_applied += 1
                    else:
                        console.print(f"     🔄 {orig} → [green]{repl.replacement}[/green] ({repl.description})")
                        if repl.notes:
                            console.print(f"        ⚠️  {repl.notes}")
                        # Apply replacement in workflow
                        for nid, nd in workflow.items():
                            if nd.get("class_type") == orig:
                                nd["class_type"] = repl.replacement
                                # Remap inputs
                                old_inputs = nd.get("inputs", {})
                                new_inputs = {}
                                for k, v in old_inputs.items():
                                    new_key = repl.input_mapping.get(k, k)
                                    new_inputs[new_key] = v
                                nd["inputs"] = new_inputs
                        report.fixes_applied += 1
                
                if still_unknown:
                    report.fixes_needed.append(FixAction(
                        description=f"Lookup unknown nodes via ComfyUI-Manager: {', '.join(still_unknown)}",
                        commands=[],
                        category="lookup_node",
                        auto=False,
                    ))
        else:
            if registered:
                console.print("  ✅ All node types are installed")

        # 5. Model references
        if analysis.model_references:
            console.print(f"  📁 {len(analysis.model_references)} model references found:")
            
            for ref in analysis.model_references:
                filename = ref["filename"]
                folder = ref["model_folder"]
                
                # Check if model exists on disk
                model_path = os.path.join(self.models_path, folder, filename)
                exists = os.path.isfile(model_path)
                
                if exists:
                    console.print(f"     ✅ {filename}")
                else:
                    info = lookup_model(filename)
                    if info:
                        report.known_models.append({**ref, "download_info": info})
                        console.print(
                            f"     📥 {filename} → [cyan]{info.size_gb}GB[/cyan] "
                            f"({'🔑 token required' if info.hf_token_required else '🆓 free'})"
                        )
                        report.fixes_needed.append(FixAction(
                            description=f"Download {filename} ({info.size_gb}GB)",
                            commands=[self._download_command(info)],
                            category="download_model",
                        ))
                    else:
                        report.unknown_models.append(ref)
                        console.print(f"     ❓ {filename} → [yellow]unknown model, search manually[/yellow]")
                        report.fixes_needed.append(FixAction(
                            description=f"Find and download: {filename}",
                            commands=[],
                            category="download_model",
                            auto=False,
                        ))

        # 6. Connection errors
        if analysis.connection_errors:
            console.print(f"  ⚠️  {len(analysis.connection_errors)} connection warnings:")
            for err in analysis.connection_errors:
                console.print(f"     {err}")

        return report

    # ── Fix application ───────────────────────────────────────────────

    def apply_fixes(self, report: DoctorReport) -> DoctorReport:
        """Apply all auto-fixable fixes from the report."""
        auto_fixes = [f for f in report.fixes_needed if f.auto and f.commands]
        
        if not auto_fixes:
            console.print("\n✅ No automatic fixes needed")
            return report

        console.print(f"\n🔧 [bold]Applying {len(auto_fixes)} fixes:[/bold]")

        installed_nodes = False

        for fix in auto_fixes:
            console.print(f"\n  → {fix.description}")
            
            if self.dry_run:
                for cmd in fix.commands:
                    console.print(f"    [dim]DRY RUN: {cmd}[/dim]")
                report.fixes_applied.append(f"[DRY] {fix.description}")
                continue

            for cmd in fix.commands:
                console.print(f"    $ {cmd}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=600,
                    )
                    if result.returncode == 0:
                        console.print(f"    ✅ Done")
                        report.fixes_applied.append(fix.description)
                        if fix.category == "install_node":
                            installed_nodes = True
                    elif "already exists" in result.stderr:
                        # Node already cloned — still run the deps part of the command
                        # The command format is: "git clone ... ; test -f req && pip install -r req"
                        # Re-run just the pip part
                        console.print(f"    ✅ Already cloned, installing deps...")
                        import re as _re
                        # Extract package name from the command to find requirements.txt
                        pkg_match = _re.search(r'custom_nodes/([^\s/]+)/requirements\.txt', cmd)
                        if pkg_match:
                            req_path = f"{self.custom_nodes_path}/{pkg_match.group(1)}/requirements.txt"
                            dep_result = subprocess.run(
                                f"test -f {req_path} && pip install -r {req_path} || true",
                                shell=True, capture_output=True, text=True, timeout=300,
                            )
                            if dep_result.returncode == 0:
                                console.print(f"    ✅ Deps installed")
                            else:
                                console.print(f"    ⚠️  Some deps failed")
                        report.fixes_applied.append(f"{fix.description} (deps updated)")
                        if fix.category == "install_node":
                            installed_nodes = True
                    else:
                        stderr = result.stderr.strip()[-200:]
                        console.print(f"    ❌ Failed: {stderr}")
                        report.errors_encountered.append(
                            f"Fix failed ({fix.description}): {stderr}"
                        )
                except subprocess.TimeoutExpired:
                    console.print(f"    ❌ Timeout (10min)")
                    report.errors_encountered.append(
                        f"Fix timed out: {fix.description}"
                    )
                except Exception as e:
                    console.print(f"    ❌ Error: {e}")
                    report.errors_encountered.append(str(e))

        # Restart ComfyUI if we installed new nodes (they need a restart to load)
        if installed_nodes and not self.dry_run:
            self.restart_comfyui()

        return report

    # ── Run workflow with auto-fix ────────────────────────────────────

    def run(self, workflow_path: str) -> DoctorReport:
        """The main entry point — analyze, fix, run, and auto-fix errors.
        
        This is the "one-shot" magic:
        1. Analyze workflow → find missing stuff
        2. Install missing nodes, deps, models
        3. Queue the workflow
        4. If error → parse → fix → retry
        5. Repeat up to max_retries
        """
        # Phase 1: Pre-flight analysis
        report = self.analyze(workflow_path)

        if report.errors_encountered:
            console.print("\n❌ [red]Cannot proceed due to errors above[/red]")
            return report

        # Phase 2: Apply pre-flight fixes
        if report.fixes_needed and self.auto_fix:
            report = self.apply_fixes(report)

            # Check if there are manual fixes needed
            manual = [f for f in report.fixes_needed if not f.auto]
            if manual:
                console.print(f"\n⚠️  [yellow]{len(manual)} manual fixes needed:[/yellow]")
                for f in manual:
                    console.print(f"  → {f.description}")
                
                if not self._confirm("Continue anyway?"):
                    return report

        # Phase 3: Run the workflow
        if not self.api.ping():
            console.print("\n🔴 [red]ComfyUI is not running. Start it first.[/red]")
            console.print(f"   Hint: cd {self.comfyui_path} && python main.py --listen")
            return report

        workflow, _ = load_workflow(workflow_path)

        # Pre-flight: validate and auto-fix inputs against /object_info
        object_info = self.api.object_info()
        if "error" not in object_info:
            # Auto-fix: clamp out-of-range values, fix enum case
            workflow, input_fixes = auto_fix_inputs(workflow, object_info)
            if input_fixes:
                console.print(f"\n🔧 [bold]Auto-fixed {len(input_fixes)} input values:[/bold]")
                for fix in input_fixes:
                    console.print(f"  → {fix}")
                    report.fixes_applied.append(fix)
            
            # Validate remaining issues
            validation_errors = validate_inputs(workflow, object_info)
            if validation_errors:
                console.print(f"\n⚠️  [yellow]{len(validation_errors)} input validation warnings:[/yellow]")
                for err in validation_errors[:10]:
                    console.print(f"  → {err}")
                if len(validation_errors) > 10:
                    console.print(f"  ... and {len(validation_errors) - 10} more")
                console.print("  [dim]These may cause HTTP 400 when queuing.[/dim]")

        for attempt in range(1, self.max_retries + 1):
            report.attempts = attempt
            console.print(f"\n🚀 [bold]Attempt {attempt}/{self.max_retries}[/bold]")

            if self.dry_run:
                console.print("  [dim]DRY RUN: would queue workflow[/dim]")
                report.success = True
                return report

            # Queue it
            result = self.api.queue_prompt(workflow)
            if "error" in result:
                error_msg = result["error"]
                details = result.get("details", {})
                # ComfyUI returns details about missing nodes in the error body
                detail_str = json.dumps(details) if details else ""
                full_error = f"{error_msg} {detail_str}"
                console.print(f"  ❌ Queue error: {error_msg}")
                if details:
                    console.print(f"     Details: {json.dumps(details, indent=2)[:300]}")
                
                # Try to fix the queue error
                if not self._try_fix_error(full_error, report):
                    break
                continue

            prompt_id = result.get("prompt_id", "")
            if not prompt_id:
                console.print("  ❌ No prompt_id returned")
                break

            console.print(f"  📨 Queued: {prompt_id[:8]}...")

            # Wait for completion
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating...", total=None)
                result = self.api.wait_for_prompt(prompt_id, timeout=600)

            if result["status"] == "success":
                console.print("  ✅ [green bold]Success![/green bold]")
                outputs = result.get("outputs", {})
                has_quality_issue = False
                if outputs:
                    console.print("  📁 Outputs:")
                    for node_id, out in outputs.items():
                        if "images" in out:
                            for img in out["images"]:
                                fname = img.get("filename", "?")
                                subfolder = img.get("subfolder", "")
                                # Validate image quality
                                quality = self._check_image_quality(fname, subfolder)
                                status_icon = quality.get("icon", "🖼️")
                                console.print(
                                    f"     {status_icon}  {fname} "
                                    f"({quality.get('info', '')})"
                                )
                                if quality.get("blank"):
                                    has_quality_issue = True
                                    console.print(
                                        f"     ⚠️  [yellow]Image appears blank/solid! "
                                        f"(mean={quality.get('mean', 0):.0f}, std={quality.get('std', 0):.0f})[/yellow]"
                                    )
                        if "gifs" in out:
                            for gif in out["gifs"]:
                                console.print(
                                    f"     🎬 {gif.get('filename', '?')}"
                                )
                if has_quality_issue:
                    console.print("  ⚠️  [yellow]Some outputs may be blank — check the images![/yellow]")
                report.success = True
                return report

            elif result["status"] == "error":
                messages = result.get("messages", [])
                error_text = json.dumps(messages, indent=2) if messages else "Unknown error"
                console.print(f"  ❌ Execution error")
                report.errors_encountered.append(error_text)

                # Try to auto-fix
                if attempt < self.max_retries and self.auto_fix:
                    if self._try_fix_error(error_text, report):
                        console.print("  🔄 Retrying...")
                        continue
                    else:
                        console.print("  ❌ Could not auto-fix this error")
                        break
                else:
                    break

            elif result["status"] == "timeout":
                console.print("  ⏱️  Timed out waiting for completion")
                break

        if not report.success:
            console.print("\n❌ [red bold]Workflow failed after all attempts[/red bold]")

        return report

    # ── Auto-fix helpers ──────────────────────────────────────────────

    def _try_fix_error(self, error_text: str, report: DoctorReport) -> bool:
        """Try to fix an error using the knowledge base. Returns True if fix applied."""
        # Special case: ComfyUI returns "missing_node_type" when nodes aren't loaded
        # This means we installed nodes but didn't restart
        if "missing_node_type" in error_text or "not found" in error_text.lower():
            import re
            m = re.search(r"Node '([^']+)' not found", error_text)
            if m:
                node_type = m.group(1)
                console.print(f"  🔍 Missing node at runtime: [cyan]{node_type}[/cyan]")
                console.print(f"     ComfyUI needs a restart to load newly installed nodes")
                if self.restart_comfyui():
                    return True
                return False

        matches = match_error(error_text)
        
        if not matches:
            console.print("  🤷 Error not recognized in knowledge base")
            console.print(f"  [dim]{error_text[:300]}[/dim]")
            return False

        fixed_any = False
        need_restart = False
        for match in matches:
            console.print(f"  🔍 Matched: [cyan]{match.pattern_name}[/cyan]")
            console.print(f"     {match.fix_description}")

            if match.fix_commands and not self.dry_run:
                for cmd in match.fix_commands:
                    console.print(f"     $ {cmd}")
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, timeout=300,
                        )
                        if result.returncode == 0:
                            console.print(f"     ✅ Fixed")
                            report.fixes_applied.append(match.fix_description)
                            fixed_any = True
                            if match.category in ("missing_module", "missing_node"):
                                need_restart = True
                        else:
                            console.print(f"     ❌ Fix failed: {result.stderr[:100]}")
                    except Exception as e:
                        console.print(f"     ❌ Error: {e}")

            elif match.category == "cuda_oom":
                console.print("     💡 Suggestions:")
                console.print("        - Lower resolution in the workflow")
                console.print("        - Replace VAEDecode with VAEDecodeTiled")
                console.print("        - Enable FP8 quantization")
                console.print("        - Reduce batch size to 1")

        if need_restart and fixed_any:
            self.restart_comfyui()

        return fixed_any

    def _download_command(self, info: ModelInfo) -> str:
        """Generate download command for a model.
        
        Resolves HuggingFace/CivitAI redirections before aria2c
        (aria2c multi-connection chokes on redirected URLs).
        """
        dest = os.path.join(self.models_path, info.model_folder, info.filename)
        dest_dir = os.path.dirname(dest)
        # Resolve redirections first, then use aria2c for fast downloads
        return (
            f'mkdir -p "{dest_dir}" && '
            f'RESOLVED_URL=$(curl -sI -L -o /dev/null -w "%{{url_effective}}" "{info.url}") && '
            f'(command -v aria2c >/dev/null 2>&1 && '
            f'aria2c -x 16 -s 16 --max-connection-per-server=16 '
            f'--min-split-size=5M --file-allocation=none '
            f'-d "{dest_dir}" -o "{info.filename}" "$RESOLVED_URL" || '
            f'wget -q --show-progress -O "{dest}" "{info.url}")'
        )

    def _check_image_quality(self, filename: str, subfolder: str = "") -> dict:
        """Check if an output image is blank/solid color.
        
        Returns: {icon, info, blank, mean, std, width, height}
        """
        try:
            # Build path to the image
            output_dir = os.path.join(self.comfyui_path, "output")
            if subfolder:
                img_path = os.path.join(output_dir, subfolder, filename)
            else:
                img_path = os.path.join(output_dir, filename)
            
            if not os.path.isfile(img_path):
                return {"icon": "🖼️", "info": "file not found", "blank": False}
            
            # Quick check using PIL
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(img_path)
                w, h = img.size
                arr = np.array(img, dtype=float)
                mean_val = arr.mean()
                std_val = arr.std()
                
                is_blank = std_val < 10  # Almost no variation = blank/solid
                is_dark = mean_val < 15 and std_val < 15  # All black
                
                if is_blank:
                    if mean_val > 240:
                        return {"icon": "⚪", "info": f"{w}x{h} WHITE", "blank": True, "mean": mean_val, "std": std_val, "width": w, "height": h}
                    elif is_dark:
                        return {"icon": "⚫", "info": f"{w}x{h} BLACK", "blank": True, "mean": mean_val, "std": std_val, "width": w, "height": h}
                    else:
                        return {"icon": "🟫", "info": f"{w}x{h} SOLID", "blank": True, "mean": mean_val, "std": std_val, "width": w, "height": h}
                
                size_mb = os.path.getsize(img_path) / (1024 * 1024)
                return {"icon": "🖼️", "info": f"{w}x{h} {size_mb:.1f}MB", "blank": False, "mean": mean_val, "std": std_val, "width": w, "height": h}
                
            except ImportError:
                # No PIL — check file size as heuristic
                size = os.path.getsize(img_path)
                if size < 1000:  # < 1KB is suspicious for an image
                    return {"icon": "⚠️", "info": f"{size}B (suspiciously small)", "blank": True}
                size_mb = size / (1024 * 1024)
                return {"icon": "🖼️", "info": f"{size_mb:.1f}MB", "blank": False}
                
        except Exception as e:
            return {"icon": "🖼️", "info": str(e)[:50], "blank": False}

    def _confirm(self, message: str) -> bool:
        """Ask user for confirmation."""
        try:
            answer = console.input(f"\n  {message} [y/N]: ")
            return answer.strip().lower() in ("y", "yes", "oui", "o")
        except (EOFError, KeyboardInterrupt):
            return False

    # ── Diagnosis ─────────────────────────────────────────────────────

    def diagnose(self) -> dict:
        """Full health check of the ComfyUI setup."""
        console.print("\n🏥 [bold]ComfyUI Doctor — System Diagnosis[/bold]\n")
        
        results = {}

        # ComfyUI connection
        if self.api.ping():
            stats = self.api.system_stats()
            system = stats.get("system", {})
            devices = stats.get("devices", [])
            
            console.print(f"  🟢 ComfyUI running at {self.api.url}")
            console.print(f"     OS: {system.get('os', '?')}")
            console.print(f"     Python: {system.get('python_version', '?')}")
            
            for i, dev in enumerate(devices):
                name = dev.get("name", "GPU")
                vram_total = dev.get("vram_total", 0) / (1024**3)
                vram_free = dev.get("vram_free", 0) / (1024**3)
                console.print(f"     GPU {i}: {name} — {vram_free:.1f}/{vram_total:.1f} GB free")
            
            results["comfyui"] = "running"
            results["system"] = system
            results["devices"] = devices
        else:
            console.print(f"  🔴 ComfyUI not reachable at {self.api.url}")
            results["comfyui"] = "down"

        # ComfyUI path
        console.print(f"\n  📂 ComfyUI path: {self.comfyui_path}")
        if os.path.isdir(self.comfyui_path):
            console.print(f"     ✅ Directory exists")
        else:
            console.print(f"     ❌ Directory not found")

        # Custom nodes
        cn_path = self.custom_nodes_path
        if os.path.isdir(cn_path):
            nodes = [d for d in os.listdir(cn_path) if os.path.isdir(os.path.join(cn_path, d))]
            console.print(f"\n  📦 Custom nodes: {len(nodes)} installed")
            results["custom_nodes_count"] = len(nodes)
        else:
            console.print(f"\n  ❌ Custom nodes directory not found: {cn_path}")

        # Models
        models_path = self.models_path
        if os.path.isdir(models_path):
            total_size = 0
            model_count = 0
            for root, dirs, files in os.walk(models_path):
                for f in files:
                    if f.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
                        model_count += 1
                        total_size += os.path.getsize(os.path.join(root, f))
            console.print(f"\n  💾 Models: {model_count} files ({total_size / (1024**3):.1f} GB)")
            results["models_count"] = model_count
            results["models_size_gb"] = total_size / (1024**3)

        # Disk space
        try:
            stat = os.statvfs(self.comfyui_path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            console.print(f"\n  💿 Disk: {free_gb:.1f}/{total_gb:.1f} GB free")
            results["disk_free_gb"] = free_gb
        except Exception:
            pass

        # Registered node types
        if results.get("comfyui") == "running":
            types = self.api.registered_node_types()
            console.print(f"\n  🧩 Registered node types: {len(types)}")
            results["node_types_count"] = len(types)

        return results
