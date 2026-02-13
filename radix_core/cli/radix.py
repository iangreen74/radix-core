"""
Radix Core CLI — GPU orchestration from the command line.

Usage:
    radix-core status        Show system status and safety configuration
    radix-core submit        Submit a job for execution
    radix-core plan          Create an execution plan for a set of jobs
    radix-core info          Show detailed system information
    radix-core validate      Validate configuration and safety guards
"""

import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..config import get_config
from ..types import Job, ResourceRequirements
from ..dryrun import DryRunGuard
from ..cost_simulator import CostSimulator
from ..errors import RadixError, SafetyViolationError

console = Console()
cli = typer.Typer(
    name="radix-core",
    help="GPU Orchestration Platform — scheduling, placement, execution, and governance.",
    no_args_is_help=True,
)


@cli.command()
def status():
    """Show system status, safety configuration, and runtime summary."""
    config = get_config()

    # Safety panel
    safety_table = Table(show_header=False, box=box.SIMPLE)
    safety_table.add_column("Setting", style="bold")
    safety_table.add_column("Value")

    safety_table.add_row("Dry-run mode", _bool_badge(config.safety.dry_run))
    safety_table.add_row("No-deploy mode", _bool_badge(config.safety.no_deploy_mode))
    safety_table.add_row("Cost cap (USD)", f"${config.safety.cost_cap_usd:.2f}")
    safety_table.add_row("Max job cost (USD)", f"${config.safety.max_job_cost_usd:.2f}")

    console.print(Panel(safety_table, title="Safety Configuration", border_style="green"))

    # Execution panel
    exec_table = Table(show_header=False, box=box.SIMPLE)
    exec_table.add_column("Setting", style="bold")
    exec_table.add_column("Value")

    exec_table.add_row("Default executor", config.execution.default_executor)
    exec_table.add_row("Max parallelism", str(config.execution.max_parallelism))
    exec_table.add_row("GPU enabled", _bool_badge(config.execution.enable_gpu))
    exec_table.add_row("Ray local mode", _bool_badge(config.execution.ray_local_mode))
    exec_table.add_row("Ray CPUs", str(config.execution.ray_num_cpus))
    exec_table.add_row("Ray GPUs", str(config.execution.ray_num_gpus))

    console.print(Panel(exec_table, title="Execution Configuration", border_style="blue"))

    # Batching panel
    batch_table = Table(show_header=False, box=box.SIMPLE)
    batch_table.add_column("Setting", style="bold")
    batch_table.add_column("Value")

    batch_table.add_row("Default batch size", str(config.batching.default_batch_size))
    batch_table.add_row("Microbatch size", str(config.batching.microbatch_size))
    batch_table.add_row("Max batch wait (s)", str(config.batching.max_batch_wait))
    batch_table.add_row("Dynamic batching", _bool_badge(config.batching.enable_dynamic_batching))

    console.print(Panel(batch_table, title="Batching Configuration", border_style="cyan"))


@cli.command()
def submit(
    command: str = typer.Argument(..., help="Command to execute"),
    name: str = typer.Option("", "--name", "-n", help="Job name"),
    cpus: float = typer.Option(1.0, "--cpus", help="CPU cores required"),
    memory: int = typer.Option(512, "--memory", "-m", help="Memory in MB"),
    gpus: int = typer.Option(0, "--gpus", help="GPUs required"),
    priority: int = typer.Option(0, "--priority", "-p", help="Job priority (higher = more urgent)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run mode (always on)"),
):
    """Submit a job for scheduling and execution."""
    config = get_config()

    if not dry_run:
        console.print("[red]Error:[/red] dry-run mode cannot be disabled for safety.")
        raise typer.Exit(code=1)

    try:
        requirements = ResourceRequirements(
            cpu_cores=cpus,
            memory_mb=memory,
            gpu_count=gpus,
        )

        job = Job(
            name=name or "cli-job",
            command=command,
            requirements=requirements,
            priority=priority,
        )

        # Estimate cost
        simulator = CostSimulator()
        estimate = simulator.estimate_job_cost(job)

        # Display job info
        table = Table(title="Job Submitted", box=box.ROUNDED)
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Job ID", job.job_id)
        table.add_row("Name", job.name)
        table.add_row("Command", command)
        table.add_row("Status", job.status.name)
        table.add_row("Priority", str(priority))
        table.add_row("CPU cores", str(cpus))
        table.add_row("Memory (MB)", str(memory))
        table.add_row("GPU count", str(gpus))
        table.add_row("Est. duration (s)", f"{job.estimated_duration():.1f}")
        table.add_row("Est. cost (USD)", f"${estimate.estimated_cost_usd:.2f}")
        table.add_row("Dry-run", _bool_badge(config.safety.dry_run))

        console.print(table)

        # Output JSON for piping
        console.print(f"\n[dim]Job ID: {job.job_id}[/dim]")

    except SafetyViolationError as e:
        console.print(f"[red]Safety violation:[/red] {e}")
        raise typer.Exit(code=1)
    except RadixError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(code=1)


@cli.command()
def plan(
    count: int = typer.Option(3, "--count", "-c", help="Number of sample jobs to plan"),
    planner_type: str = typer.Option("greedy", "--planner", help="Planner type: greedy or optimal"),
    output_json: bool = typer.Option(False, "--json", help="Output plan as JSON"),
):
    """Create an execution plan for a set of sample jobs."""
    from ..scheduler.planner import get_planner

    try:
        planner = get_planner(planner_type)

        # Create sample jobs
        jobs = []
        for i in range(count):
            job = Job(
                name=f"plan-job-{i}",
                command=f"echo 'Job {i}'",
                priority=count - i,
                requirements=ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=512,
                ),
            )
            jobs.append(job)

        execution_plan = planner.create_execution_plan(jobs)

        if output_json:
            plan_data = {
                "plan_id": execution_plan.plan_id,
                "planner_type": planner_type,
                "job_count": len(execution_plan.scheduled_jobs),
                "dependencies_resolved": execution_plan.dependencies_resolved,
                "resource_requirements": execution_plan.resource_requirements,
                "metadata": execution_plan.plan_metadata,
            }
            console.print(json.dumps(plan_data, indent=2))
        else:
            # Display plan
            console.print(Panel(
                f"Plan ID: {execution_plan.plan_id}\n"
                f"Planner: {planner_type}\n"
                f"Jobs: {len(execution_plan.scheduled_jobs)}\n"
                f"Dependencies resolved: {execution_plan.dependencies_resolved}",
                title="Execution Plan",
                border_style="yellow",
            ))

            table = Table(title="Scheduled Jobs", box=box.ROUNDED)
            table.add_column("#", style="dim")
            table.add_column("Job ID", style="bold")
            table.add_column("Name")
            table.add_column("Priority")
            table.add_column("CPUs")
            table.add_column("Memory (MB)")
            table.add_column("Est. Duration")

            for i, job in enumerate(execution_plan.scheduled_jobs):
                table.add_row(
                    str(i + 1),
                    job.job_id[:12] + "...",
                    job.name,
                    str(job.priority),
                    str(job.requirements.cpu_cores),
                    str(job.requirements.memory_mb),
                    f"{job.estimated_duration():.1f}s",
                )

            console.print(table)

            if execution_plan.resource_requirements:
                res_table = Table(title="Total Resources", box=box.SIMPLE)
                res_table.add_column("Resource", style="bold")
                res_table.add_column("Required")
                for key, val in execution_plan.resource_requirements.items():
                    res_table.add_row(key, str(val))
                console.print(res_table)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@cli.command()
def info():
    """Show detailed system information including available components."""
    # Version info
    console.print(Panel(
        "radix-core v0.1.0\nGPU Orchestration Platform",
        title="Radix Core",
        border_style="magenta",
    ))

    # Components table
    table = Table(title="Available Components", box=box.ROUNDED)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Scheduler components
    table.add_row("FIFO Policy", "[green]available[/green]", "First-in-first-out scheduling")
    table.add_row("Priority Policy", "[green]available[/green]", "Priority-based scheduling")
    table.add_row("Fair Share Policy", "[green]available[/green]", "Fair resource allocation")
    table.add_row("SJF Policy", "[green]available[/green]", "Shortest-job-first scheduling")
    table.add_row("Greedy Planner", "[green]available[/green]", "Greedy execution planning")
    table.add_row("Optimal Planner", "[green]available[/green]", "Optimized execution planning")
    table.add_row("Local Placement", "[green]available[/green]", "Single-node placement")
    table.add_row("Load-Balanced", "[green]available[/green]", "Multi-node load balancing")
    table.add_row("Job Graph (DAG)", "[green]available[/green]", "Dependency-aware scheduling")

    # Executor components
    table.add_row("ThreadPool", "[green]available[/green]", "Thread-based execution")

    try:
        import ray
        table.add_row("Ray Local", "[green]available[/green]", f"Ray {ray.__version__}")
    except ImportError:
        table.add_row("Ray Local", "[yellow]not installed[/yellow]", "pip install ray[default]")

    try:
        import transformers
        table.add_row("HuggingFace", "[green]available[/green]", f"transformers {transformers.__version__}")
    except ImportError:
        table.add_row("HuggingFace", "[yellow]not installed[/yellow]", "pip install transformers")

    try:
        from vllm import __version__ as vllm_ver
        table.add_row("vLLM", "[green]available[/green]", f"vLLM {vllm_ver}")
    except ImportError:
        table.add_row("vLLM", "[yellow]not installed[/yellow]", "pip install vllm")

    # Batching
    table.add_row("Dynamic Batcher", "[green]available[/green]", "SLA-aware batching")
    table.add_row("Microbatcher", "[green]available[/green]", "Memory-efficient microbatching")

    console.print(table)

    # Safety verification
    try:
        DryRunGuard.verify_safety()
        console.print("\n[green]All safety checks passed.[/green]")
    except SafetyViolationError as e:
        console.print(f"\n[red]Safety check failed:[/red] {e}")


@cli.command()
def validate():
    """Validate configuration and safety guards."""
    config = get_config()

    console.print("[bold]Running validation checks...[/bold]\n")
    errors = []

    # 1. Config validation
    config_errors = config.validate()
    if config_errors:
        for err in config_errors:
            errors.append(f"Config: {err}")
            console.print(f"  [red]FAIL[/red] {err}")
    else:
        console.print("  [green]PASS[/green] Configuration is valid")

    # 2. Safety validation
    try:
        DryRunGuard.verify_safety()
        console.print("  [green]PASS[/green] Safety guards are active")
    except SafetyViolationError as e:
        errors.append(f"Safety: {e}")
        console.print(f"  [red]FAIL[/red] {e}")

    # 3. Import checks
    import_checks = [
        ("radix_core.types", "Core types"),
        ("radix_core.scheduler", "Scheduler"),
        ("radix_core.executors.threadpool", "ThreadPool executor"),
        ("radix_core.batching", "Batching"),
        ("radix_core.cost_simulator", "Cost simulator"),
    ]

    for module_path, label in import_checks:
        try:
            __import__(module_path)
            console.print(f"  [green]PASS[/green] {label} imports OK")
        except ImportError as e:
            errors.append(f"Import {module_path}: {e}")
            console.print(f"  [red]FAIL[/red] {label}: {e}")

    # Summary
    console.print()
    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s).[/red]")
        raise typer.Exit(code=1)
    else:
        console.print("[green]All validation checks passed.[/green]")


def _bool_badge(value: bool) -> str:
    """Return a colored badge for a boolean value."""
    if value:
        return "[green]enabled[/green]"
    return "[red]disabled[/red]"


if __name__ == "__main__":
    cli()
