"""CLI for running scheduler simulations."""

import typer
import yaml
import json
import os
import random
import numpy as np
import time
import statistics
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ..schedulers import SCHEDULERS
from ..schedulers.base import Job, GPU, JobStatus
from ..core.costs import CostModel
from ..metrics.report import MetricsCalculator, create_summary_markdown

# Import predictive components
try:
    from ..predictive import SimplePredictor, pack_job_features
    
    # Test if we can import required dependencies
    import pandas as pd
    import numpy as np
    import joblib
    import sklearn
    
    PREDICTIVE_AVAILABLE = True
except ImportError as e:
    PREDICTIVE_AVAILABLE = False
    _import_error = str(e)

app = typer.Typer(help="Radix-Bench Scheduler Simulation CLI")
console = Console()


def set_deterministic_seeds(seed: int = 1337):
    """Set seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHON_SEED'] = str(seed)
    os.environ['NUMPY_SEED'] = str(seed)


def create_synthetic_workload(config: dict) -> List[Job]:
    """Create synthetic job workload based on configuration."""
    jobs = []
    workload_config = config.get('workload', {})

    num_jobs = workload_config.get('num_jobs', 100)
    arrival_rate = workload_config.get('arrival_rate', 1.0)  # jobs per second

    # Job size distribution
    memory_sizes = workload_config.get('memory_sizes', [8, 16, 32, 80])
    memory_weights = workload_config.get('memory_weights', [0.4, 0.3, 0.2, 0.1])

    # Runtime distribution
    runtime_mean = workload_config.get('runtime_mean', 300.0)  # seconds
    runtime_std = workload_config.get('runtime_std', 100.0)

    current_time = 0.0

    for i in range(num_jobs):
        # Generate arrival time (Poisson process)
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival

        # Sample job characteristics
        memory_gb = np.random.choice(memory_sizes, p=memory_weights)
        runtime_estimate = max(10.0, np.random.normal(runtime_mean, runtime_std))

        # Create job
        job = Job(
            job_id=f"job_{i:04d}",
            submit_time=current_time,
            runtime_estimate=runtime_estimate,
            memory_gb=memory_gb,
            priority=0,
            user=f"user_{i % 5}"  # 5 users
        )
        jobs.append(job)

    return jobs


def create_gpu_cluster(config: dict) -> List[GPU]:
    """Create GPU cluster based on configuration."""
    cluster_config = config.get('cluster', {})

    gpus = []
    gpu_types = cluster_config.get('gpu_types', [
        {'memory_gb': 80, 'count': 4, 'compute_capability': '8.0'}
    ])

    gpu_id = 0
    for gpu_type in gpu_types:
        for _ in range(gpu_type['count']):
            gpu = GPU(
                gpu_id=f"gpu_{gpu_id:02d}",
                memory_gb=gpu_type['memory_gb'],
                compute_capability=gpu_type.get('compute_capability', '8.0'),
                is_available=True
            )
            gpus.append(gpu)
            gpu_id += 1

    return gpus


def run_simulation(scheduler_name: str, jobs: List[Job], cluster: List[GPU],
                  config: dict, predictor=None, latency_ctx: Optional[dict] = None) -> dict:
    """Run simulation for a single scheduler."""
    # Initialize scheduler
    if scheduler_name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    scheduler_class = SCHEDULERS[scheduler_name]
    scheduler = scheduler_class()

    # Attach predictor to scheduler if available and supported
    if predictor is not None and hasattr(scheduler, 'set_predictor'):
        scheduler.set_predictor(predictor)
        console.print(f"[green]Predictive mode enabled for {scheduler_name}[/green]")
    elif predictor is not None:
        console.print(f"[yellow]Predictor provided but {scheduler_name} doesn't support predictive mode[/yellow]")

    # Initialize cost model
    cost_model = CostModel(config.get('cost_model', {}))

    # Submit all jobs
    for job in jobs:
        scheduler.submit(job)

    # Simulation parameters
    sim_config = config.get('simulation', {})
    time_step = sim_config.get('time_step', 1.0)  # seconds
    max_time = sim_config.get('max_time', 3600.0)  # 1 hour

    current_time = 0.0
    completed_jobs = []

    # Reset GPU availability
    for gpu in cluster:
        gpu.is_available = True
        gpu.current_job = None

    # Simulation loop
    while current_time < max_time and len(completed_jobs) < len(jobs):
        # Check for completed jobs
        for gpu in cluster:
            if not gpu.is_available and gpu.current_job:
                # Find the job running on this GPU
                running_job = None
                for job in jobs:
                    if job.assigned_gpu == gpu.gpu_id and job.status == JobStatus.RUNNING:
                        running_job = job
                        break

                if running_job and running_job.start_time is not None:
                    # Check if job should complete
                    elapsed_time = current_time - running_job.start_time
                    if elapsed_time >= running_job.runtime_estimate:
                        # Complete the job
                        running_job.status = JobStatus.COMPLETED
                        running_job.end_time = current_time

                        # Record cost
                        runtime_hours = elapsed_time / 3600.0
                        cost_model.record_job_cost(
                            running_job.job_id, gpu.gpu_id, gpu.memory_gb,
                            runtime_hours, gpu.compute_capability
                        )

                        # Free GPU
                        gpu.is_available = True
                        gpu.current_job = None

                        completed_jobs.append(running_job)

                        # Notify scheduler of completion
                        if hasattr(scheduler, 'job_completed'):
                            scheduler.job_completed(running_job)

        # Check for preemption opportunities (opt-in via RADIX_PREEMPT_GATE=1)
        preempt_gate_enabled = os.environ.get("RADIX_PREEMPT_GATE", "0") not in ("0", "false", "False")
        if preempt_gate_enabled and hasattr(scheduler, 'preempt'):
            preempt_actions = scheduler.preempt(cluster, current_time)
            
            # Execute preemption actions
            for action in preempt_actions:
                if len(action) >= 4:
                    action_type, victim_job, replacement_job, gpu = action[:4]
                    
                    if action_type == "preempt":
                        # Find and preempt the victim job
                        for job in jobs:
                            if (getattr(job, "job_id", "") == getattr(victim_job, "job_id", "") and
                                job.status == JobStatus.RUNNING):
                                # Mark job as pending again
                                job.status = JobStatus.PENDING
                                job.start_time = None
                                job.assigned_gpu = None
                                
                                # Free the GPU
                                gpu.is_available = True
                                gpu.current_job = None
                                
                                if verbose:
                                    console.print(f"[yellow]Preempted job {job.job_id} from GPU {gpu.gpu_id}[/yellow]")
                                break

        # Schedule new jobs (optionally measure decision latency)
        _t0 = None
        if latency_ctx and latency_ctx.get("enabled"):
            try:
                _t0 = time.monotonic_ns()
            except Exception:
                _t0 = None
        assignments = scheduler.schedule(cluster, current_time)
        if _t0 is not None:
            try:
                dt = time.monotonic_ns() - _t0
                latency_ctx.setdefault("samples", []).append(int(dt))
            except Exception:
                pass

        # Apply assignments
        for assignment in assignments:
            assignment.job.status = JobStatus.RUNNING
            assignment.job.start_time = current_time
            assignment.job.assigned_gpu = assignment.gpu.gpu_id
            
            assignment.gpu.is_available = False
            assignment.gpu.current_job = assignment.job.job_id

        current_time += time_step

    # Calculate metrics
    metrics_calc = MetricsCalculator(cost_model)
    metrics = metrics_calc.calculate_basic_metrics(scheduler_name, jobs, current_time)

    result = {
        'scheduler': scheduler_name,
        'metrics': metrics,
        'completed_jobs': len(completed_jobs),
        'total_jobs': len(jobs),
        'simulation_time': current_time,
        'cost_metrics': cost_model.get_cost_metrics()
    }
    if latency_ctx and latency_ctx.get("enabled"):
        result["latency_samples_ns"] = list(latency_ctx.get("samples", []))
    return result


@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to configuration YAML file"),
    schedulers: Optional[List[str]] = typer.Option(None, "--scheduler", "-s", help="Schedulers to run (default: all)"),
    output_dir: str = typer.Option("results", "--output", "-o", "--out", help="Output directory"),
    seed: int = typer.Option(1337, "--seed", help="Random seed for reproducibility"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    predictive_artifacts: Optional[str] = typer.Option(None, "--predictive-artifacts", help="Path to ONNX predictive model artifacts directory"),
    measure_latency: bool = typer.Option(False, "--measure-latency", help="Measure scheduler decision latency (ns/op)"),
    tau: Optional[float] = typer.Option(None, "--tau", help="Softmax temperature for Radix policy"),
    aging: Optional[float] = typer.Option(None, "--aging", help="Aging coefficient for Radix policy")
):
    """Run scheduler simulation with specified configuration."""

    # Set deterministic execution
    set_deterministic_seeds(seed)

    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Determine schedulers to run
    if schedulers is None:
        schedulers = list(SCHEDULERS.keys())

    # Validate schedulers
    for scheduler_name in schedulers:
        if scheduler_name not in SCHEDULERS:
            console.print(f"[red]Error: Unknown scheduler '{scheduler_name}'[/red]")
            console.print(f"Available schedulers: {list(SCHEDULERS.keys())}")
            raise typer.Exit(1)

    console.print(f"[green]Running simulation with {len(schedulers)} schedulers[/green]")
    console.print(f"Configuration: {config_file}")
    console.print(f"Seed: {seed}")

    # Create workload and cluster
    jobs = create_synthetic_workload(config)
    cluster = create_gpu_cluster(config)

    console.print(f"Workload: {len(jobs)} jobs")
    console.print(f"Cluster: {len(cluster)} GPUs")

    # Initialize predictor if artifacts provided
    predictor = None
    if predictive_artifacts and PREDICTIVE_AVAILABLE:
        try:
            predictor = SimplePredictor(Path(predictive_artifacts))
            backend = getattr(predictor, 'backend', 'unknown')
            console.print(f"[green]Loaded predictor from {predictive_artifacts}[/green]")
            console.print(f"[blue]Backend: {backend}[/blue]")
        except Exception as e:
            console.print(f"[red]Failed to load predictor: {e}[/red]")
            console.print("[yellow]Continuing without predictive mode[/yellow]")
    elif predictive_artifacts and not PREDICTIVE_AVAILABLE:
        console.print("[red]Predictive artifacts provided but dependencies not available[/red]")
        if '_import_error' in globals():
            console.print(f"[yellow]Import error: {_import_error}[/yellow]")
        console.print("[yellow]Install: pip install onnxruntime scikit-learn joblib[/yellow]")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run simulations
    results = {}

    for scheduler_name in track(schedulers, description="Running schedulers..."):
        if verbose:
            console.print(f"Running {scheduler_name}...")

        # Create fresh copies of jobs and cluster for each scheduler
        job_copies = [Job(
            job_id=job.job_id,
            submit_time=job.submit_time,
            runtime_estimate=job.runtime_estimate,
            memory_gb=job.memory_gb,
            priority=job.priority,
            user=job.user
        ) for job in jobs]

        cluster_copies = [GPU(
            gpu_id=gpu.gpu_id,
            memory_gb=gpu.memory_gb,
            compute_capability=gpu.compute_capability,
            is_available=True
        ) for gpu in cluster]

        # Latency context per scheduler
        latency_ctx = {"enabled": bool(measure_latency), "samples": []} if measure_latency else None

        # Map CLI policy parameters for Radix variants
        scheduler_kwargs = {}
        if scheduler_name in ("radix", "radix_softmax"):
            if tau is not None:
                scheduler_kwargs["softmax_tau"] = float(tau)
            if aging is not None:
                # Map aging to alpha in current implementation
                scheduler_kwargs["alpha"] = float(aging)

        # If kwargs provided and scheduler supports them, rebuild scheduler
        if scheduler_kwargs:
            try:
                scheduler_class = SCHEDULERS[scheduler_name]
                scheduler = scheduler_class(**scheduler_kwargs)  # type: ignore[arg-type]
                # Re-submit jobs to the new scheduler instance
                for job in job_copies:
                    scheduler.submit(job)
                # Replace default scheduling by calling through a tiny shim
                # Create a cluster copy is already done
                # Run simulation with injected scheduler via monkeypatch pattern
                # Simpler approach: temporarily override SCHEDULERS entry
                saved = SCHEDULERS[scheduler_name]
                SCHEDULERS[scheduler_name] = lambda: scheduler  # type: ignore[assignment]
                try:
                    result = run_simulation(scheduler_name, job_copies, cluster_copies, config, predictor, latency_ctx=latency_ctx)
                finally:
                    SCHEDULERS[scheduler_name] = saved
            except Exception:
                # Fallback to standard path
                result = run_simulation(scheduler_name, job_copies, cluster_copies, config, predictor, latency_ctx=latency_ctx)
        else:
            result = run_simulation(scheduler_name, job_copies, cluster_copies, config, predictor, latency_ctx=latency_ctx)
        results[scheduler_name] = result

        # Per-scheduler output directory (support single-scheduler runs writing directly to output_dir)
        scheduler_out_dir = output_dir if len(schedulers) == 1 else os.path.join(output_dir, scheduler_name)
        os.makedirs(scheduler_out_dir, exist_ok=True)

        # Write a concise summary.json for downstream tooling
        try:
            summary = {
                "scheduler": scheduler_name,
                "throughput_mean": float(getattr(result.get('metrics'), 'throughput', 0.0)),
                "completed_jobs": int(result.get('completed_jobs', 0)),
                "total_jobs": int(result.get('total_jobs', 0)),
                "simulation_time": float(result.get('simulation_time', 0.0))
            }
            with open(os.path.join(scheduler_out_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]WARN: failed to write summary.json for {scheduler_name}: {e}[/yellow]")

        # Persist latency summary if enabled
        if measure_latency and result.get("latency_samples_ns"):
            try:
                lat = sorted(result["latency_samples_ns"]) or []
                def pct(p: float) -> int:
                    if not lat:
                        return 0
                    i = max(0, min(len(lat)-1, int(round(p*(len(lat)-1)))))
                    return int(lat[i])
                summary = {
                    "count": len(lat),
                    "mean_ns": int(statistics.fmean(lat)) if lat else 0,
                    "p50_ns": pct(0.50),
                    "p95_ns": pct(0.95),
                    "p99_ns": pct(0.99),
                    "max_ns": int(lat[-1]) if lat else 0,
                }
                with open(os.path.join(scheduler_out_dir, "latency.json"), 'w') as f:
                    json.dump(summary, f, indent=2)
                console.print(f"[blue][latency][/blue] {scheduler_name}: decisions={summary['count']} mean_ns={summary['mean_ns']} p95_ns={summary['p95_ns']} max_ns={summary['max_ns']}")
            except Exception as e:
                console.print(f"[yellow]WARN: failed to write latency.json for {scheduler_name}: {e}[/yellow]")

    # Generate report
    metrics_calc = MetricsCalculator()
    for scheduler_name, result in results.items():
        metrics_calc.add_scheduler_result(result['metrics'])

    # Export results
    report = metrics_calc.generate_statistical_report()

    # Save JSON report
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save markdown summary
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, 'w') as f:
        f.write(create_summary_markdown(report))

    # Print summary
    console.print("\n[green]Simulation completed![/green]")
    metrics_calc.print_summary()

    console.print(f"\nResults saved to: {output_dir}/")
    console.print(f"  - JSON report: {json_path}")
    console.print(f"  - Markdown summary: {md_path}")


@app.command()
def list_schedulers():
    """List available schedulers."""
    table = Table(title="Available Schedulers")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="magenta")
    table.add_column("Description", style="green")

    descriptions = {
        "fifo": "First-In-First-Out baseline",
        "srpt": "Shortest Remaining Processing Time",
        "drf": "Dominant Resource Fairness",
        "easy": "EASY Backfilling",
        "heft": "Heterogeneous Earliest Finish Time",
        "bfd": "Best Fit Decreasing",
        "gavel": "Utility-aware scheduling",
        "radix": "Information-theoretic scheduler (Radix v0)"
    }

    for name, scheduler_class in SCHEDULERS.items():
        description = descriptions.get(name, "GPU scheduler implementation")
        table.add_row(name, scheduler_class.__name__, description)

    console.print(table)


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
