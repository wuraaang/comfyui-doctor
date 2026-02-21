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
    no_llm: bool = False,
    verbose: bool = False,
) -> Doctor:
    """Create a Doctor instance, optionally with LLM client."""
    llm_client = None
    if not no_llm:
        llm_client = _try_create_llm_client()

    return Doctor(
        comfyui_url=url,
        comfyui_path=path,
        max_retries=retries,
        auto_fix=auto_fix,
        dry_run=dry_run,
        llm_client=llm_client,
        verbose=verbose,
    )


def _try_create_llm_client():
    """Try to create an LLM client from auth config."""
    try:
        from .core.auth import AuthManager
        from .core.llm import LLMClient

        auth = AuthManager()
        if auth.is_authenticated():
            config = auth.load_config()
            client = LLMClient(
                proxy_url=config.proxy_url,
                auth_token=config.token,
            )
            return client
    except Exception:
        pass
    return None


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
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM escalation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show LLM reasoning"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept prompts"),
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

    doctor = get_doctor(url, path, retries, auto_fix=not no_fix, dry_run=dry_run,
                        no_llm=no_llm, verbose=verbose)

    # Show LLM status
    if doctor.llm_client:
        console.print("  🤖 [dim]LLM escalation enabled[/dim]")
    if doctor.mcp_client:
        console.print("  🔗 [dim]MCP connected[/dim]")

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
    """📊 Show ComfyUI status (queue, GPU, LLM, MCP)."""
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

    # LLM status
    llm_client = _try_create_llm_client()
    if llm_client:
        available = llm_client.is_available()
        status_str = "[green]connected[/green]" if available else "[yellow]configured but unreachable[/yellow]"
        console.print(f"\n   🤖 LLM: {status_str}")
        # Quota
        try:
            from .core.auth import AuthManager
            auth = AuthManager()
            quota = auth.get_quota()
            if quota:
                remaining = quota.get("remaining", "?")
                total = quota.get("total", "?")
                console.print(f"      Quota: {remaining}/{total} requests remaining")
        except Exception:
            pass
    else:
        console.print(f"\n   🤖 LLM: [dim]not configured (run `comfyui-doctor login`)[/dim]")

    # MCP status
    try:
        from .core.mcp_client import MCPConnection
        mcp = MCPConnection(url=comfyui_url)
        if mcp.connect():
            if mcp.has_comfy_pilot():
                tools = mcp.list_tools()
                console.print(f"   🔗 Comfy-Pilot: [green]connected[/green] ({len(tools)} tools)")
            else:
                console.print(f"   🔗 ComfyUI API: [green]connected[/green] (no Comfy-Pilot)")
        else:
            console.print(f"   🔗 MCP: [dim]not available[/dim]")
    except Exception:
        console.print(f"   🔗 MCP: [dim]not available[/dim]")

    # Tokens
    try:
        from .core.auth import TokenManager
        tm = TokenManager()
        masked = tm.list_tokens()
        if masked:
            console.print(f"\n   🔑 Tokens:")
            for service, token_str in masked.items():
                console.print(f"      {service}: {token_str}")
        else:
            console.print(f"\n   🔑 Tokens: [dim]none configured[/dim]")
    except Exception:
        pass


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
def login(
    email: str = typer.Argument(..., help="Your email address"),
    proxy_url: str = typer.Option(
        "https://api.comfyui-doctor.com", "--proxy", help="LLM proxy URL"
    ),
):
    """🔑 Login to enable LLM-powered auto-fix."""
    from .core.auth import AuthManager

    console.print(Panel(
        "[bold]🔑 ComfyUI Doctor — Login[/bold]",
        border_style="cyan",
    ))

    auth = AuthManager()
    if auth.login(email, proxy_url):
        console.print("\n✅ You're logged in! LLM escalation is now enabled.")
        console.print("   Run `comfyui-doctor status` to verify.")
    else:
        console.print("\n❌ Login failed.")
        raise typer.Exit(1)


@app.command()
def tokens(
    action: str = typer.Argument("list", help="Action: list, set, delete"),
    service: Optional[str] = typer.Argument(None, help="Service: huggingface, civitai"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Token value (for set)"),
):
    """🔑 Manage API tokens (HuggingFace, CivitAI)."""
    from .core.auth import TokenManager

    tm = TokenManager()

    if action == "list":
        masked = tm.list_tokens()
        if not masked:
            console.print("No tokens configured.")
            console.print("\nAvailable services:")
            for svc, info in tm.SERVICES.items():
                console.print(f"  {svc}: {info['description']}")
            console.print(f"\nSet a token: comfyui-doctor tokens set <service> --token <value>")
            return

        table = Table(title="API Tokens")
        table.add_column("Service", style="cyan")
        table.add_column("Token")
        table.add_column("Get token at")
        for svc, val in masked.items():
            info = tm.SERVICES.get(svc, {})
            table.add_row(svc, val, info.get("url", ""))
        console.print(table)

    elif action == "set":
        if not service:
            console.print("Usage: comfyui-doctor tokens set <service> --token <value>")
            console.print(f"Services: {', '.join(tm.SERVICES.keys())}")
            raise typer.Exit(1)
        if token:
            tm.set_token(service, token)
            console.print(f"✅ Token set for {service}")
        else:
            tm.prompt_for_token(service)

    elif action == "delete":
        if not service:
            console.print("Usage: comfyui-doctor tokens delete <service>")
            raise typer.Exit(1)
        tm.delete_token(service)
        console.print(f"✅ Token deleted for {service}")

    else:
        console.print(f"Unknown action: {action}")
        console.print("Available: list, set, delete")
        raise typer.Exit(1)


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

    if report.escalated_to_llm:
        table.add_row("LLM escalation", f"[cyan]{len(report.llm_suggestions)} suggestions[/cyan]")

    table.add_row("Attempts", str(report.attempts))

    status = "[green bold]✅ SUCCESS[/green bold]" if report.success else "[red bold]❌ FAILED[/red bold]"
    table.add_row("Result", status)

    console.print(table)
    console.print("═" * 60)


@app.command()
def optimize(
    workflow_path: str = typer.Argument(help="Path to workflow JSON file"),
    url: str = typer.Option("http://127.0.0.1:8188", "--url", "-u", help="ComfyUI URL"),
    vram: float = typer.Option(0, "--vram", help="GPU VRAM in GB (auto-detected if 0)"),
):
    """⚡ Analyze a workflow for speed/memory/quality optimizations."""
    from .core.optimizer import analyze_optimizations, format_optimizations
    from .core.workflow import load_workflow

    console.print(Panel(
        f"[bold]⚡ ComfyUI Doctor Optimizer[/bold]\nWorkflow: {workflow_path}",
        border_style="cyan",
    ))

    workflow, is_api = load_workflow(workflow_path)

    # Auto-detect VRAM
    if vram == 0:
        api = ComfyAPI(url)
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(f"{url}/system_stats", timeout=5) as r:
                stats = _json.loads(r.read())
                devices = stats.get("system", {}).get("devices", [])
                if devices:
                    vram = devices[0].get("vram_total", 0) / 1024 / 1024 / 1024
                    console.print(f"  GPU: {devices[0].get('name', '?')} ({vram:.0f}GB VRAM)")
        except Exception:
            console.print("  ⚠️  Could not detect GPU — pass --vram manually")

    opts = analyze_optimizations(workflow, vram_gb=vram)
    console.print(format_optimizations(opts))

    if not opts:
        raise typer.Exit(0)
    raise typer.Exit(0)


@app.command()
def create(
    template: str = typer.Argument(help="Template name: txt2img-sdxl, txt2video-wan"),
    output: str = typer.Option("workflow.json", "--output", "-o", help="Output file path"),
    prompt: str = typer.Option("a beautiful sunset over mountains", "--prompt", "-p"),
    negative: str = typer.Option("ugly, blurry", "--negative", "-n"),
    width: int = typer.Option(1024, "--width", "-W"),
    height: int = typer.Option(1024, "--height", "-H"),
    steps: int = typer.Option(20, "--steps"),
    seed: int = typer.Option(42, "--seed"),
):
    """🎨 Create a workflow from a template."""
    from .core.builder import build_txt2img_sdxl, build_txt2video_wan

    templates = {
        "txt2img-sdxl": lambda: build_txt2img_sdxl(
            prompt=prompt, negative=negative, width=width, height=height,
            steps=steps, seed=seed,
        ),
        "txt2video-wan": lambda: build_txt2video_wan(
            prompt=prompt, negative=negative, width=width, height=height,
            steps=steps, seed=seed,
        ),
    }

    if template not in templates:
        console.print(f"[red]Unknown template '{template}'[/red]")
        console.print(f"Available: {', '.join(templates.keys())}")
        raise typer.Exit(1)

    workflow = templates[template]()

    import json
    with open(output, "w") as f:
        json.dump(workflow, f, indent=2)

    console.print(f"✅ Created {template} workflow → {output}")
    console.print(f"   Run it: comfyui-doctor run {output}")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
