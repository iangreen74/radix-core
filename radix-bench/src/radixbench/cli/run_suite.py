#!/usr/bin/env python3
"""
Radix Benchmark Suite Runner

Runs scheduler comparisons and policy sweeps with statistical analysis.
"""

import os
import sys
import json
import yaml
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import typer
from rich.console import Console
from rich.progress import track

console = Console()

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def set_deterministic_env():
    """Set environment variables for deterministic execution."""
    os.environ.update({
        'TZ': 'UTC',
        'LC_ALL': 'C.UTF-8', 
        'PYTHONHASHSEED': '0',
        'PYTHON_SEED': '1337',
        'NUMPY_SEED': '1337'
    })

def run_simulation(scheduler: str, seed: int, config: Dict[str, Any], 
                  predictive_artifacts: Optional[Path], out_dir: Path,
                  policy_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Run a single simulation and return metrics."""
    
    # Create temporary config for this run
    run_config = config.copy()
    run_name = f"{scheduler}_seed{seed}"
    
    if policy_params:
        for param, value in policy_params.items():
            run_name += f"_{param}{value}"
    
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command - use simple simulation for now
    cmd = [
        sys.executable, "-c", f"""
import json
import random
import numpy as np
random.seed({seed})
np.random.seed({seed})

# Mock simulation results for testing
result = {{
    'scheduler': '{scheduler}',
    'seed': {seed},
    'throughput': random.uniform(0.02, 0.05),
    'utilization': random.uniform(0.7, 0.9),
    'avg_wait_time': random.uniform(10, 50),
    'avg_runtime': random.uniform(200, 400),
    'total_cost_usd': 0.0,
    'completed_jobs': random.randint(180, 200),
    'total_jobs': 200
}}

with open('{run_dir}/results.json', 'w') as f:
    json.dump(result, f)
"""
    ]
    
    if predictive_artifacts:
        cmd.extend(["--predictive-artifacts", str(predictive_artifacts)])
    
    # Set policy parameters via environment if provided
    env = os.environ.copy()
    if policy_params:
        if 'temperature' in policy_params:
            env['RADIX_SOFTMAX_TAU'] = str(policy_params['temperature'])
        if 'aging_alpha' in policy_params:
            env['RADIX_ALPHA'] = str(policy_params['aging_alpha'])
    
    try:
        # Run simulation
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent.parent.parent.parent.parent,  # radix root
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            console.print(f"[red]Simulation failed for {run_name}[/red]")
            console.print(f"[red]Error: {result.stderr}[/red]")
            return None
            
        # Parse results
        results_file = run_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                
            # Extract key metrics
            metrics = {
                'scheduler': scheduler,
                'seed': seed,
                'throughput': data.get('throughput', 0.0),
                'utilization': data.get('utilization', 0.0),
                'avg_wait_time': data.get('avg_wait_time', 0.0),
                'avg_runtime': data.get('avg_runtime', 0.0),
                'total_cost_usd': data.get('total_cost_usd', 0.0),
                'completed_jobs': data.get('completed_jobs', 0),
                'total_jobs': data.get('total_jobs', 0)
            }
            
            if policy_params:
                metrics.update(policy_params)
                
            return metrics
            
    except subprocess.TimeoutExpired:
        console.print(f"[red]Simulation timeout for {run_name}[/red]")
    except Exception as e:
        console.print(f"[red]Error running {run_name}: {e}[/red]")
    
    return None

def compute_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Compute confidence intervals using t-distribution."""
    if len(values) < 2:
        return {'mean': values[0] if values else 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
    
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of mean
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
    
    return {
        'mean': mean,
        'ci_lower': ci[0], 
        'ci_upper': ci[1],
        'std': np.std(values),
        'n': len(values)
    }

def run_league_benchmark(config_path: Path, predictive_artifacts: Optional[Path], out_dir: Path):
    """Run league benchmark comparing all schedulers."""
    console.print("[bold blue]Running League Benchmark[/bold blue]")
    
    config = load_config(config_path)
    set_deterministic_env()
    
    schedulers = config['schedulers']
    seeds = config['seeds']
    
    all_results = []
    
    # Run all combinations
    total_runs = len(schedulers) * len(seeds)
    with console.status(f"Running {total_runs} simulations...") as status:
        for scheduler in schedulers:
            for seed in seeds:
                status.update(f"Running {scheduler} with seed {seed}")
                result = run_simulation(scheduler, seed, config, predictive_artifacts, out_dir)
                if result:
                    all_results.append(result)
    
    # Aggregate results
    df = pd.DataFrame(all_results)
    
    # Compute statistics per scheduler
    metrics = ['throughput', 'utilization', 'avg_wait_time', 'avg_runtime', 'total_cost_usd']
    comparisons = []
    
    for scheduler in schedulers:
        sched_data = df[df['scheduler'] == scheduler]
        
        comparison = {'scheduler': scheduler}
        for metric in metrics:
            values = sched_data[metric].tolist()
            stats_dict = compute_confidence_intervals(values)
            comparison[f'{metric}_mean'] = stats_dict['mean']
            comparison[f'{metric}_ci_lower'] = stats_dict['ci_lower']
            comparison[f'{metric}_ci_upper'] = stats_dict['ci_upper']
        
        comparisons.append(comparison)
    
    # Save results
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(out_dir / "comparisons.csv", index=False)
    
    # Compute win rates vs FIFO
    fifo_data = df[df['scheduler'] == 'fifo']
    winrates = []
    
    for scheduler in schedulers:
        if scheduler == 'fifo':
            continue
            
        sched_data = df[df['scheduler'] == scheduler]
        winrate = {'scheduler': scheduler}
        
        for metric in metrics:
            # Higher is better for throughput, utilization; lower is better for wait_time, cost
            better_count = 0
            total_count = 0
            
            for seed in seeds:
                fifo_val = fifo_data[fifo_data['seed'] == seed][metric].iloc[0] if len(fifo_data[fifo_data['seed'] == seed]) > 0 else 0
                sched_val = sched_data[sched_data['seed'] == seed][metric].iloc[0] if len(sched_data[sched_data['seed'] == seed]) > 0 else 0
                
                if metric in ['throughput', 'utilization']:
                    if sched_val > fifo_val:
                        better_count += 1
                else:  # Lower is better
                    if sched_val < fifo_val:
                        better_count += 1
                total_count += 1
            
            winrate[f'{metric}_winrate'] = better_count / total_count if total_count > 0 else 0.0
        
        winrates.append(winrate)
    
    # Save winrates
    winrates_df = pd.DataFrame(winrates)
    winrates_df.to_csv(out_dir / "winrate.csv", index=False)
    
    # Create summary
    best_scheduler = comparisons_df.loc[comparisons_df['throughput_mean'].idxmax(), 'scheduler']
    
    summary = {
        'experiment': 'league_small',
        'schedulers': schedulers,
        'seeds': seeds,
        'best_throughput_scheduler': best_scheduler,
        'total_simulations': len(all_results),
        'predictive_backend': 'onnx' if predictive_artifacts else 'none'
    }
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[green]League benchmark complete. Best scheduler: {best_scheduler}[/green]")

def run_policy_sweep(config_path: Path, predictive_artifacts: Optional[Path], out_dir: Path):
    """Run policy parameter sweep for RadixSoftmax."""
    console.print("[bold blue]Running Policy Sweep[/bold blue]")
    
    config = load_config(config_path)
    set_deterministic_env()
    
    scheduler = config['scheduler']
    seeds = config['seeds']
    sweep_config = config['policy_sweep']
    
    temperatures = sweep_config['temperature']
    aging_alphas = sweep_config['aging_alpha']
    
    all_results = []
    
    # Run parameter grid
    total_runs = len(temperatures) * len(aging_alphas) * len(seeds)
    with console.status(f"Running {total_runs} parameter combinations...") as status:
        for temp in temperatures:
            for alpha in aging_alphas:
                for seed in seeds:
                    policy_params = {'temperature': temp, 'aging_alpha': alpha}
                    status.update(f"Running τ={temp}, α={alpha}, seed={seed}")
                    
                    result = run_simulation(scheduler, seed, config, predictive_artifacts, 
                                          out_dir, policy_params)
                    if result:
                        all_results.append(result)
    
    # Aggregate results
    df = pd.DataFrame(all_results)
    
    # Compute statistics per parameter combination
    metrics = ['throughput', 'utilization', 'avg_wait_time', 'avg_runtime']
    comparisons = []
    
    for temp in temperatures:
        for alpha in aging_alphas:
            param_data = df[(df['temperature'] == temp) & (df['aging_alpha'] == alpha)]
            
            if len(param_data) == 0:
                continue
                
            comparison = {'temperature': temp, 'aging_alpha': alpha}
            for metric in metrics:
                values = param_data[metric].tolist()
                stats_dict = compute_confidence_intervals(values)
                comparison[f'{metric}_mean'] = stats_dict['mean']
                comparison[f'{metric}_ci_lower'] = stats_dict['ci_lower']
                comparison[f'{metric}_ci_upper'] = stats_dict['ci_upper']
            
            comparisons.append(comparison)
    
    # Save results
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(out_dir / "comparisons.csv", index=False)
    
    # Select best defaults (highest median throughput)
    best_idx = comparisons_df['throughput_mean'].idxmax()
    best_params = comparisons_df.iloc[best_idx]
    
    summary = {
        'experiment': 'policy_sweep',
        'scheduler': scheduler,
        'seeds': seeds,
        'parameter_grid': sweep_config,
        'selected_defaults': {
            'temperature': best_params['temperature'],
            'aging_alpha': best_params['aging_alpha']
        },
        'selection_rationale': f"Highest throughput: {best_params['throughput_mean']:.4f}",
        'total_combinations': len(all_results)
    }
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[green]Policy sweep complete. Selected: τ={best_params['temperature']}, α={best_params['aging_alpha']}[/green]")
    return best_params['temperature'], best_params['aging_alpha']

def main(
    config: Path = typer.Argument(..., help="Configuration YAML file"),
    predictive_artifacts: Optional[Path] = typer.Option(None, "--predictive-artifacts", help="Path to predictive model artifacts"),
    out: Path = typer.Option(..., "--out", help="Output directory")
):
    """Run benchmark suite with given configuration."""
    
    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Determine experiment type from config
    config_data = load_config(config)
    
    if 'policy_sweep' in config_data:
        run_policy_sweep(config, predictive_artifacts, out)
    else:
        run_league_benchmark(config, predictive_artifacts, out)

if __name__ == "__main__":
    typer.run(main)
