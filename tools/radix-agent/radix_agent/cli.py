"""
Radix Agent CLI — Infrastructure discovery and control.

Usage:
    radix-agent discover     Discover AWS resources (read-only)
    radix-agent reconcile    Compare inventory against existing Terraform
    radix-agent generate     Generate Terraform for unmanaged resources
    radix-agent apply        Run terraform plan/import (requires --confirm)
    radix-agent dashboard-gaps  Report dashboard feature gaps
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import AgentConfig
from .state import AgentState

console = Console()
app = typer.Typer(
    name="radix-agent",
    help="Infrastructure discovery and control agent for radix-core.",
    no_args_is_help=True,
)


def _get_session(region: str, profile: Optional[str] = None):
    """Create a boto3 session."""
    import boto3

    kwargs = {"region_name": region}
    if profile:
        kwargs["profile_name"] = profile
    return boto3.Session(**kwargs)


@app.command()
def discover(
    services: List[str] = typer.Option(
        ["all"], "--services", "-s", help="Services to discover (comma-sep or 'all')"
    ),
    region: str = typer.Option("us-west-2", "--region", "-r", help="AWS region"),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="AWS profile name"
    ),
    output_dir: str = typer.Option(
        "tools/radix-agent/output", "--output", "-o", help="Output directory"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from previous run"),
):
    """Discover AWS resources (read-only). Outputs inventory JSON."""
    from .discover import run_discovery

    config = AgentConfig(aws_region=region, aws_profile=profile, output_dir=output_dir)
    state_path = config.state_file

    if resume:
        state = AgentState.load(state_path)
        console.print(f"[yellow]Resuming run {state.run_id}[/yellow]")
    else:
        state = AgentState()
        state.phase = "discover"

    session = _get_session(region, profile)

    # Get account ID
    try:
        sts = session.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        console.print(f"[green]AWS Account:[/green] {account_id}")
        console.print(f"[green]Region:[/green] {region}")
    except Exception as e:
        console.print(f"[red]Failed to get AWS identity:[/red] {e}")
        raise typer.Exit(code=1)

    # Parse services
    svc_list = []
    for s in services:
        svc_list.extend(s.split(","))

    console.print(
        Panel(
            f"Services: {', '.join(svc_list)}\nOutput: {output_dir}",
            title="Discovery Configuration",
            border_style="blue",
        )
    )

    inventory = run_discovery(session, region, svc_list, state, output_dir, account_id)

    # Save inventory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    inventory_file = out_path / "inventory.json"
    inventory_file.write_text(inventory.model_dump_json(indent=2))

    state.inventory_path = str(inventory_file)
    state.phase = "discover_complete"
    state.save(state_path)

    # Summary table
    table = Table(title="Discovery Summary", box=box.ROUNDED)
    table.add_column("Service", style="bold")
    table.add_column("Resources", justify="right")

    for svc in inventory.services:
        table.add_row(svc.service, str(svc.resource_count))

    table.add_row("[bold]Total[/bold]", f"[bold]{inventory.total_resources}[/bold]")
    console.print(table)
    console.print(f"\n[green]Inventory saved to {inventory_file}[/green]")


@app.command()
def reconcile(
    inventory: str = typer.Argument(
        "tools/radix-agent/output/inventory.json", help="Path to inventory.json"
    ),
    tf_dir: str = typer.Option("infra/aws", "--tf-dir", help="Terraform directory"),
    output_dir: str = typer.Option(
        "tools/radix-agent/output", "--output", "-o", help="Output directory"
    ),
):
    """Compare inventory against existing Terraform files."""
    from .terraform.reconciler import TerraformReconciler

    inv_path = Path(inventory)
    if not inv_path.exists():
        console.print(f"[red]Inventory file not found:[/red] {inventory}")
        console.print("Run 'radix-agent discover' first.")
        raise typer.Exit(code=1)

    from .models import Inventory

    inv = Inventory.model_validate_json(inv_path.read_text())

    reconciler = TerraformReconciler(tf_dir)
    all_resources = []
    for svc in inv.services:
        all_resources.extend(svc.resources)

    result = reconciler.reconcile(all_resources)

    # Save report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "reconcile-report.json"
    report_file.write_text(result.model_dump_json(indent=2))

    # Display
    table = Table(title="Reconciliation Report", box=box.ROUNDED)
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("[green]Already in .tf[/green]", str(len(result.managed)))
    table.add_row("[yellow]Unmanaged (need import)[/yellow]", str(len(result.unmanaged)))
    table.add_row("[dim]Framework-only (.tf, no AWS)[/dim]", str(len(result.framework_only)))

    console.print(table)

    if result.unmanaged:
        console.print("\n[yellow]Unmanaged resources by type:[/yellow]")
        by_type: dict = {}
        for r in result.unmanaged:
            by_type.setdefault(r.resource_type, []).append(r)
        for rtype, resources in sorted(by_type.items()):
            console.print(f"  {rtype}: {len(resources)}")

    console.print(f"\n[green]Report saved to {report_file}[/green]")


@app.command()
def generate(
    inventory: str = typer.Argument(
        "tools/radix-agent/output/inventory.json", help="Path to inventory.json"
    ),
    tf_dir: str = typer.Option("infra/aws", "--tf-dir", help="Target Terraform directory"),
    output_dir: str = typer.Option(
        "tools/radix-agent/output", "--output", "-o", help="Output directory"
    ),
):
    """Generate Terraform files for unmanaged resources."""
    from .models import Inventory
    from .terraform.generator import TerraformGenerator
    from .terraform.importer import ImportScriptGenerator
    from .terraform.reconciler import TerraformReconciler

    inv_path = Path(inventory)
    if not inv_path.exists():
        console.print(f"[red]Inventory file not found:[/red] {inventory}")
        raise typer.Exit(code=1)

    inv = Inventory.model_validate_json(inv_path.read_text())

    # Reconcile first
    reconciler = TerraformReconciler(tf_dir)
    all_resources = []
    for svc in inv.services:
        all_resources.extend(svc.resources)
    result = reconciler.reconcile(all_resources)

    if not result.unmanaged:
        console.print("[green]All resources are already managed in Terraform.[/green]")
        return

    # Generate .tf files
    generator = TerraformGenerator(tf_dir)
    generated = generator.generate(result.unmanaged)

    # Generate import script
    importer = ImportScriptGenerator()
    import_script = importer.generate(result.unmanaged, tf_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    import_file = out_path / "import.sh"
    import_file.write_text(import_script)
    import_file.chmod(0o755)

    # Summary
    table = Table(title="Generated Files", box=box.ROUNDED)
    table.add_column("File", style="bold")
    table.add_column("Resources", justify="right")

    for filepath, count in generated.items():
        table.add_row(filepath, str(count))

    console.print(table)
    console.print(f"\n[green]Import script saved to {import_file}[/green]")
    console.print(
        "[dim]Run 'cd infra/aws && terraform init && bash ../../tools/radix-agent/output/import.sh'[/dim]"
    )


@app.command()
def apply(
    confirm: bool = typer.Option(
        False, "--confirm", help="Actually run terraform plan/import"
    ),
    tf_dir: str = typer.Option("infra/aws", "--tf-dir", help="Terraform directory"),
):
    """Run terraform plan (and optionally import). Requires --confirm."""
    import subprocess

    if not confirm:
        console.print("[yellow]Dry run mode.[/yellow] Pass --confirm to execute.")
        console.print(f"Would run: cd {tf_dir} && terraform init && terraform plan")
        return

    console.print("[bold]Running terraform init...[/bold]")
    result = subprocess.run(
        ["terraform", "init"], cwd=tf_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        console.print(f"[red]terraform init failed:[/red]\n{result.stderr}")
        raise typer.Exit(code=1)

    console.print("[bold]Running terraform plan...[/bold]")
    result = subprocess.run(
        ["terraform", "plan", "-out=tfplan"], cwd=tf_dir, capture_output=True, text=True
    )
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"[red]terraform plan failed:[/red]\n{result.stderr}")
        raise typer.Exit(code=1)

    console.print("[green]Plan complete. Review the output above.[/green]")


@app.command(name="dashboard-gaps")
def dashboard_gaps():
    """Analyze and report dashboard feature gaps vs production."""
    from .dashboard.gap_analyzer import analyze_gaps

    gaps = analyze_gaps()

    table = Table(title="Dashboard Feature Gaps", box=box.ROUNDED)
    table.add_column("Priority", style="bold")
    table.add_column("Feature")
    table.add_column("Status")
    table.add_column("Complexity")

    for gap in gaps:
        color = {"P0": "green", "P1": "yellow", "P2": "cyan", "P3": "red"}.get(
            gap["priority"], "white"
        )
        table.add_row(
            f"[{color}]{gap['priority']}[/{color}]",
            gap["feature"],
            gap["status"],
            gap["complexity"],
        )

    console.print(table)
    console.print(f"\n[bold]Total gaps: {len(gaps)}[/bold]")


if __name__ == "__main__":
    app()
