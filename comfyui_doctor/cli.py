"""ComfyUI Doctor CLI — Make any workflow work in one shot."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .core.api import ComfyAPI
from .core.doctor import Doctor, DoctorReport
from .core.workflow import load_workflow, analyze_workflow, extract_node_types_from_json
from .knowledge.error_db import match_error
from .knowledge.model_map import lookup_model
from .knowledge.node_map import lookup_node_type
from .knowledge.manager_db import manager_lookup, get_map_stats

app = typer.Typer(
    name="comfyui-doctor",
    help="🩺 Make any ComfyUI workflow work in one shot.",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()

# ── Global options ────────────────────────────────────────────────────

def get_doctor(
    url: str = "http://127.0.0.1:8188",
    path: Optional[str] = None,
    retries: int = 3,
    auto_fix: bool = True,
    dry_run: bool = False,
) -> Doctor:
    return Doctor(
        comfyui_url=url,
        comfyui_path=path,
        max_retries=retries,
        auto_fix=auto_fix,
        dry_run=dry_run,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def run(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="ComfyUI install path"),
    retries: int = typer.Option(3, "--retries", "-r", help="Max auto-fix retries"),
    no_fix: bool = typer.Option(False, "--no-fix", help="Disable auto-fix"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Analyze only, don't execute"),
):
    """🚀 Run a workflow with auto-fix. The one-shot magic."""
    console.print(Panel(
        f"[bold]🩺 ComfyUI Doctor v{__version__}[/bold]\n"
        f"Workflow: {workflow}",
        title="comfyui-doctor run",
        border_style="cyan",
    ))

    if not os.path.isfile(workflow):
        console.print(f"❌ File not found: {workflow}")
        raise typer.Exit(1)

    doctor = get_doctor(url, path, retries, auto_fix=not no_fix, dry_run=dry_run)
    report = doctor.run(workflow)

    # Summary
    _print_summary(report)
    raise typer.Exit(0 if report.success else 1)


@app.command()
def analyze(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="ComfyUI install path"),
):
    """🔍 Analyze a workflow without running it. Shows what's missing."""
    console.print(Panel(
        f"[bold]🩺 ComfyUI Doctor — Analyzer[/bold]",
        border_style="cyan",
    ))

    if not os.path.isfile(workflow):
        console.print(f"❌ File not found: {workflow}")
        raise typer.Exit(1)

    doctor = get_doctor(url, path, dry_run=True)
    report = doctor.analyze(workflow)
    _print_summary(report)


@app.command()
def diagnose(
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="ComfyUI install path"),
):
    """🏥 Full health check of your ComfyUI setup."""
    doctor = get_doctor(url, path)
    doctor.diagnose()


@app.command()
def fix(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="ComfyUI install path"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show fixes without applying"),
):
    """🔧 Analyze and apply fixes without running the workflow."""
    console.print(Panel(
        f"[bold]🩺 ComfyUI Doctor — Fix Mode[/bold]",
        border_style="cyan",
    ))

    if not os.path.isfile(workflow):
        console.print(f"❌ File not found: {workflow}")
        raise typer.Exit(1)

    doctor = get_doctor(url, path, dry_run=dry_run)
    report = doctor.analyze(workflow)

    if report.fixes_needed:
        report = doctor.apply_fixes(report)
        _print_summary(report)
    else:
        console.print("\n✅ No fixes needed — workflow looks good!")


@app.command(name="check-error")
def check_error(
    error: str = typer.Argument(..., help="Error message to diagnose"),
):
    """🔍 Look up an error in the knowledge base."""
    matches = match_error(error)

    if not matches:
        console.print("🤷 Error not found in knowledge base")
        console.print("   Try providing the full error traceback for better matching")
        raise typer.Exit(1)

    console.print(f"\n🔍 [bold]Found {len(matches)} match(es):[/bold]\n")
    for m in matches:
        console.print(f"  [cyan]{m.pattern_name}[/cyan]")
        console.print(f"  Category: {m.category}")
        console.print(f"  Description: {m.description}")
        console.print(f"  Fix: {m.fix_description}")
        if m.fix_commands:
            for cmd in m.fix_commands:
                console.print(f"    $ {cmd}")
        console.print()


@app.command(name="check-model")
def check_model(
    filename: str = typer.Argument(..., help="Model filename to look up"),
):
    """📦 Look up a model in the database."""
    info = lookup_model(filename)

    if not info:
        console.print(f"❓ Model '{filename}' not found in database")
        console.print("   Try searching on HuggingFace or CivitAI manually")
        raise typer.Exit(1)

    console.print(f"\n📦 [bold]{info.filename}[/bold]")
    console.print(f"   Description: {info.description}")
    console.print(f"   Size: {info.size_gb} GB")
    console.print(f"   Folder: models/{info.model_folder}/")
    console.print(f"   URL: {info.url}")
    if info.hf_token_required:
        console.print(f"   ⚠️  Requires HuggingFace token")


@app.command()
def nodes(
    workflow: str = typer.Argument(..., help="Path to workflow JSON file"),
):
    """📋 List all node types used in a workflow."""
    if not os.path.isfile(workflow):
        console.print(f"❌ File not found: {workflow}")
        raise typer.Exit(1)

    types = extract_node_types_from_json(workflow)
    
    table = Table(title=f"Node types in {workflow}")
    table.add_column("Class Type", style="cyan")
    table.add_column("Count", justify="right")
    
    # Count occurrences
    with open(workflow) as f:
        data = json.load(f)
    if "prompt" in data:
        data = data["prompt"]
    
    type_counts = {}
    for node_data in data.values():
        if isinstance(node_data, dict) and "class_type" in node_data:
            ct = node_data["class_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1

    for ct in sorted(type_counts.keys()):
        table.add_row(ct, str(type_counts[ct]))

    console.print(table)
    console.print(f"\n[bold]{len(type_counts)}[/bold] unique types, [bold]{sum(type_counts.values())}[/bold] total nodes")


@app.command()
def status(
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
):
    """📊 Show ComfyUI status (queue, GPU, etc.)."""
    api = ComfyAPI(url=url)

    if not api.ping():
        console.print(f"🔴 ComfyUI not reachable at {url}")
        raise typer.Exit(1)

    # System stats
    stats = api.system_stats()
    system = stats.get("system", {})
    devices = stats.get("devices", [])

    console.print(f"\n🟢 [bold]ComfyUI[/bold] at {url}")
    console.print(f"   OS: {system.get('os', '?')} | Python {system.get('python_version', '?')}")
    
    for i, dev in enumerate(devices):
        name = dev.get("name", "GPU")
        vram_total = dev.get("vram_total", 0) / (1024**3)
        vram_free = dev.get("vram_free", 0) / (1024**3)
        pct = ((vram_total - vram_free) / vram_total * 100) if vram_total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        console.print(f"   GPU {i}: {name}")
        console.print(f"     VRAM: [{bar}] {vram_total - vram_free:.1f}/{vram_total:.1f} GB ({pct:.0f}%)")

    # Queue
    queue = api.get_queue()
    running = len(queue.get("queue_running", []))
    pending = len(queue.get("queue_pending", []))
    console.print(f"\n   Queue: {running} running, {pending} pending")

    # Node types
    types = api.registered_node_types()
    console.print(f"   Nodes: {len(types)} types registered")


@app.command()
def lookup(
    node_type: str = typer.Argument(..., help="Node class_type to look up"),
):
    """🔎 Look up where to install a node type."""
    # Try local map first
    pkg = lookup_node_type(node_type)
    
    if pkg:
        source = "[via ComfyUI-Manager]" if "[via ComfyUI-Manager]" in (pkg.description or "") else "[local db]"
        console.print(f"\n🔎 [bold]{node_type}[/bold]  {source}")
        console.print(f"   Package: {pkg.package_name}")
        console.print(f"   Repo: {pkg.repo_url}")
        if pkg.pip_deps:
            console.print(f"   Pip deps: {', '.join(pkg.pip_deps)}")
        if pkg.description:
            console.print(f"   Info: {pkg.description}")
        console.print(f"\n   Install: cd custom_nodes && git clone {pkg.repo_url}")
    else:
        console.print(f"\n❓ [bold]{node_type}[/bold] not found in any database")
        console.print("   Try searching on GitHub or ComfyUI-Manager")
    
    # Show stats
    stats = get_map_stats()
    console.print(f"\n   [dim]Database: {stats['total_types']} types from {stats['total_repos']} repos[/dim]")


@app.command()
def version():
    """Show version."""
    console.print(f"comfyui-doctor v{__version__}")


# ── Helpers ───────────────────────────────────────────────────────────

def _print_summary(report: DoctorReport):
    """Print a summary table of the doctor report."""
    console.print("\n" + "═" * 60)
    console.print("[bold]📋 Summary[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Workflow", report.workflow_path)
    table.add_row("Nodes", f"{report.total_nodes} ({report.required_types} unique types)")
    
    if report.missing_nodes:
        table.add_row("Missing nodes", f"[red]{len(report.missing_nodes)}[/red]")
    
    if report.known_models:
        table.add_row("Models to download", f"[yellow]{len(report.known_models)}[/yellow]")
    
    if report.unknown_models:
        table.add_row("Unknown models", f"[red]{len(report.unknown_models)}[/red]")
    
    if report.fixes_applied:
        table.add_row("Fixes applied", f"[green]{len(report.fixes_applied)}[/green]")

    if report.errors_encountered:
        table.add_row("Errors", f"[red]{len(report.errors_encountered)}[/red]")

    table.add_row("Attempts", str(report.attempts))
    
    status = "[green bold]✅ SUCCESS[/green bold]" if report.success else "[red bold]❌ FAILED[/red bold]"
    table.add_row("Result", status)

    console.print(table)
    console.print("═" * 60)


if __name__ == "__main__":
    app()
