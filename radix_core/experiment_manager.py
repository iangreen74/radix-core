"""
Experiment Tracking and Management System for Radix

Provides comprehensive experiment lifecycle management, parameter tracking,
result storage, and reproducibility features for GPU orchestration research.
"""

import json
import uuid
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import yaml
import random
import numpy as np
import threading

from .config import get_config
from .logging import get_logger, CorrelationContext, trace_operation
from .metrics_enhanced import get_metrics_collector


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    expected_duration_minutes: Optional[int] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Individual experiment run instance."""
    run_id: str
    experiment_id: str
    config: ExperimentConfig
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    correlation_id: Optional[str] = None
    git_commit: Optional[str] = None
    python_version: str = ""
    host_info: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    reproducibility_hash: Optional[str] = None


@dataclass
class ExperimentSeries:
    """Series of related experiments (e.g., hyperparameter sweep)."""
    series_id: str
    name: str
    description: str
    base_config: ExperimentConfig
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)
    runs: List[str] = field(default_factory=list)  # Run IDs
    created_time: datetime = field(default_factory=datetime.utcnow)
    status: str = "created"  # 'created', 'running', 'completed', 'failed'
    completion_criteria: Dict[str, Any] = field(default_factory=dict)


class ExperimentManager:
    """Comprehensive experiment management system."""

    def __init__(self, storage_path: Path = None):
        self.config = get_config()
        self.logger = get_logger("radix.experiments")

        # Storage setup
        if storage_path is None:
            storage_path = Path(self.config.research.results_dir) / "experiments"
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Internal storage
        self.experiments_db_path = self.storage_path / "experiments.json"
        self.runs_db_path = self.storage_path / "runs.json"
        self.series_db_path = self.storage_path / "series.json"

        # In-memory caches
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._runs: Dict[str, ExperimentRun] = {}
        self._series: Dict[str, ExperimentSeries] = {}
        self._active_runs: Dict[str, ExperimentRun] = {}

        # Threading
        self._lock = threading.RLock()

        # Metrics integration
        self.metrics_collector = get_metrics_collector()

        # Load existing data
        self._load_from_storage()

        self.logger.info("Experiment manager initialized",
                        storage_path=str(self.storage_path),
                        loaded_experiments=len(self._experiments),
                        loaded_runs=len(self._runs))

    @trace_operation("create_experiment")
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        experiment_id = self._generate_id(f"exp_{config.name}")

        with self._lock:
            self._experiments[experiment_id] = config

        self._save_to_storage()

        self.logger.info("Experiment created",
                        experiment_id=experiment_id,
                        name=config.name,
                        description=config.description,
                        parameters=config.parameters)

        return experiment_id

    @trace_operation("start_run")
    def start_run(self, experiment_id: str, parameters: Dict[str, Any] = None,
                  tags: List[str] = None) -> str:
        """Start a new experiment run."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment_config = self._experiments[experiment_id]
        run_id = self._generate_id(f"run_{experiment_config.name}")

        # Merge parameters
        merged_parameters = experiment_config.parameters.copy()
        if parameters:
            merged_parameters.update(parameters)

        # Create run instance
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            config=experiment_config,
            status="pending",
            correlation_id=CorrelationContext.get_correlation_id(),
            parameters=merged_parameters,
            reproducibility_hash=self._compute_reproducibility_hash(merged_parameters)
        )

        # Add environment information
        run.host_info = self._collect_host_info()
        run.python_version = self._get_python_version()
        run.git_commit = self._get_git_commit()

        with self._lock:
            self._runs[run_id] = run
            self._active_runs[run_id] = run

        # Start metrics tracking
        self.metrics_collector.start_job_metrics(
            job_id=run_id,
            job_type="experiment_run",
            executor_type="experiment_manager"
        )

        # Set reproducibility
        self._set_reproducibility(merged_parameters)

        # Update status and start time
        run.status = "running"
        run.start_time = datetime.utcnow()

        self._save_to_storage()

        self.logger.info("Experiment run started",
                        run_id=run_id,
                        experiment_id=experiment_id,
                        parameters=merged_parameters,
                        reproducibility_hash=run.reproducibility_hash)

        return run_id

    @trace_operation("finish_run")
    def finish_run(self, run_id: str, success: bool = True, results: Dict[str, Any] = None,
                   error_message: str = None):
        """Finish an experiment run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        run = self._runs[run_id]
        run.end_time = datetime.utcnow()
        run.duration_seconds = (run.end_time - run.start_time).total_seconds()
        run.status = "completed" if success else "failed"

        if results:
            run.results.update(results)

        if error_message:
            run.error_message = error_message

        # Collect final metrics
        self._collect_run_metrics(run)

        # Finish metrics tracking
        self.metrics_collector.finish_job_metrics(
            job_id=run_id,
            success=success,
            duration_seconds=run.duration_seconds,
            peak_memory_mb=run.metrics.get("peak_memory_mb", 0.0),
            cpu_time_seconds=run.metrics.get("cpu_time_seconds", 0.0)
        )

        with self._lock:
            if run_id in self._active_runs:
                del self._active_runs[run_id]

        self._save_to_storage()

        self.logger.info("Experiment run finished",
                        run_id=run_id,
                        success=success,
                        duration_seconds=run.duration_seconds,
                        results_count=len(run.results))

    @contextmanager
    def experiment_run(self, experiment_id: str, parameters: Dict[str, Any] = None,
                      tags: List[str] = None):
        """Context manager for experiment runs."""
        run_id = self.start_run(experiment_id, parameters, tags)

        try:
            yield run_id
            self.finish_run(run_id, success=True)
        except Exception as e:
            self.finish_run(run_id, success=False, error_message=str(e))
            raise

    def log_metric(self, run_id: str, name: str, value: float, step: int = 0,
                  timestamp: datetime = None):
        """Log a metric for an experiment run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        if timestamp is None:
            timestamp = datetime.utcnow()

        metric_entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": timestamp.isoformat()
        }

        run = self._runs[run_id]
        if "metrics" not in run.metrics:
            run.metrics["metrics"] = []

        run.metrics["metrics"].append(metric_entry)

        # Also record in global metrics system
        self.metrics_collector.record_metric(
            f"experiment.{name}",
            value,
            {"run_id": run_id, "experiment_id": run.experiment_id}
        )

        self.logger.debug("Metric logged",
                         run_id=run_id,
                         metric_name=name,
                         value=value,
                         step=step)

    def log_result(self, run_id: str, key: str, value: Any):
        """Log a result for an experiment run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        run = self._runs[run_id]
        run.results[key] = value

        self.logger.debug("Result logged",
                         run_id=run_id,
                         result_key=key,
                         result_value=value)

    def save_artifact(self, run_id: str, artifact_path: Path, artifact_name: str = None) -> str:
        """Save an artifact for an experiment run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        if artifact_name is None:
            artifact_name = artifact_path.name

        # Create run artifacts directory
        run_artifacts_dir = self.storage_path / "artifacts" / run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        destination = run_artifacts_dir / artifact_name
        if artifact_path.is_file():
            shutil.copy2(artifact_path, destination)
        elif artifact_path.is_dir():
            shutil.copytree(artifact_path, destination, dirs_exist_ok=True)
        else:
            raise ValueError(f"Artifact path {artifact_path} does not exist")

        # Update run record
        run = self._runs[run_id]
        artifact_rel_path = str(destination.relative_to(self.storage_path))
        run.artifacts.append(artifact_rel_path)

        self.logger.info("Artifact saved",
                        run_id=run_id,
                        artifact_name=artifact_name,
                        artifact_path=str(destination))

        return artifact_rel_path

    def create_experiment_series(self, name: str, description: str,
                               base_config: ExperimentConfig,
                               parameter_grid: Dict[str, List[Any]]) -> str:
        """Create an experiment series for parameter sweeps."""
        series_id = self._generate_id(f"series_{name}")

        series = ExperimentSeries(
            series_id=series_id,
            name=name,
            description=description,
            base_config=base_config,
            parameter_grid=parameter_grid
        )

        with self._lock:
            self._series[series_id] = series

        self._save_to_storage()

        self.logger.info("Experiment series created",
                        series_id=series_id,
                        name=name,
                        parameter_combinations=self._count_parameter_combinations(parameter_grid))

        return series_id

    def run_experiment_series(self, series_id: str, max_parallel: int = 1) -> List[str]:
        """Run all experiments in a series."""
        if series_id not in self._series:
            raise ValueError(f"Series {series_id} not found")

        series = self._series[series_id]
        series.status = "running"

        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(series.parameter_grid)

        # Create experiment if it doesn't exist
        experiment_id = self.create_experiment(series.base_config)

        run_ids = []

        try:
            for params in parameter_combinations:
                run_id = self.start_run(experiment_id, params)
                run_ids.append(run_id)
                series.runs.append(run_id)

                # For now, run sequentially (parallel execution could be added)
                try:
                    # This would be replaced with actual experiment execution
                    self.finish_run(run_id, success=True)
                except Exception as e:
                    self.finish_run(run_id, success=False, error_message=str(e))

            series.status = "completed"

        except Exception as e:
            series.status = "failed"
            self.logger.error("Experiment series failed",
                            series_id=series_id,
                            error=str(e))
            raise

        finally:
            self._save_to_storage()

        self.logger.info("Experiment series completed",
                        series_id=series_id,
                        total_runs=len(run_ids),
                        successful_runs=len([r for r in run_ids if self._runs[r].status == "completed"]))

        return run_ids

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        return self._experiments.get(experiment_id)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run."""
        return self._runs.get(run_id)

    def get_series(self, series_id: str) -> Optional[ExperimentSeries]:
        """Get experiment series."""
        return self._series.get(series_id)

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        return list(self._experiments.keys())

    def list_runs(self, experiment_id: str = None, status: str = None) -> List[str]:
        """List experiment runs with optional filtering."""
        runs = list(self._runs.keys())

        if experiment_id:
            runs = [r for r in runs if self._runs[r].experiment_id == experiment_id]

        if status:
            runs = [r for r in runs if self._runs[r].status == status]

        return runs

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs."""
        if not run_ids:
            return {}

        comparison = {
            "run_ids": run_ids,
            "runs": [self._runs[rid] for rid in run_ids if rid in self._runs],
            "parameter_differences": {},
            "result_comparison": {},
            "metric_comparison": {}
        }

        # Find parameter differences
        if comparison["runs"]:
            base_params = comparison["runs"][0].parameters
            for key in base_params:
                values = [run.parameters.get(key) for run in comparison["runs"]]
                if len(set(str(v) for v in values)) > 1:
                    comparison["parameter_differences"][key] = values

        # Compare results
        all_result_keys = set()
        for run in comparison["runs"]:
            all_result_keys.update(run.results.keys())

        for key in all_result_keys:
            values = [run.results.get(key) for run in comparison["runs"]]
            comparison["result_comparison"][key] = values

        return comparison

    def export_experiment_data(self, experiment_id: str = None, format: str = "json") -> Path:
        """Export experiment data to file."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        if experiment_id:
            filename = f"experiment_{experiment_id}_{timestamp}.{format}"
            data = {
                "experiment": asdict(self._experiments[experiment_id]) if experiment_id in self._experiments else None,
                "runs": [asdict(run) for run in self._runs.values() if run.experiment_id == experiment_id]
            }
        else:
            filename = f"all_experiments_{timestamp}.{format}"
            data = {
                "experiments": {eid: asdict(exp) for eid, exp in self._experiments.items()},
                "runs": {rid: asdict(run) for rid, run in self._runs.items()},
                "series": {sid: asdict(series) for sid, series in self._series.items()}
            }

        export_path = self.storage_path / filename

        if format == "json":
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "yaml":
            with open(export_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info("Experiment data exported",
                        export_path=str(export_path),
                        format=format)

        return export_path

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

    def _compute_reproducibility_hash(self, parameters: Dict[str, Any]) -> str:
        """Compute hash for reproducibility."""
        # Include parameters, git commit, and other relevant info
        hash_data = {
            "parameters": parameters,
            "git_commit": self._get_git_commit(),
            "python_version": self._get_python_version(),
            "random_seed": parameters.get("random_seed", self.config.research.random_seed)
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]

    def _set_reproducibility(self, parameters: Dict[str, Any]):
        """Set random seeds for reproducibility."""
        seed = parameters.get("random_seed", self.config.research.random_seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # Additional framework seeds could be set here

    def _collect_host_info(self) -> Dict[str, Any]:
        """Collect host system information."""
        try:
            import platform
            import psutil

            return {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception:
            return {"error": "Failed to collect host info"}

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.storage_path.parent
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _collect_run_metrics(self, run: ExperimentRun):
        """Collect final metrics for a run."""
        try:
            # Get system metrics
            import psutil
            process = psutil.Process()

            run.metrics.update({
                "final_memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0
            })
        except Exception as e:
            self.logger.warning("Failed to collect run metrics", error=str(e))

    def _count_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in parameter_grid.values():
            count *= len(values)
        return count

    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools

        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))

        return combinations

    def _load_from_storage(self):
        """Load data from storage files."""
        try:
            if self.experiments_db_path.exists():
                with open(self.experiments_db_path) as f:
                    experiments_data = json.load(f)
                    for eid, exp_data in experiments_data.items():
                        self._experiments[eid] = ExperimentConfig(**exp_data)

            if self.runs_db_path.exists():
                with open(self.runs_db_path) as f:
                    runs_data = json.load(f)
                    for rid, run_data in runs_data.items():
                        # Convert datetime strings back to datetime objects
                        if run_data.get("start_time"):
                            run_data["start_time"] = datetime.fromisoformat(run_data["start_time"])
                        if run_data.get("end_time"):
                            run_data["end_time"] = datetime.fromisoformat(run_data["end_time"])

                        self._runs[rid] = ExperimentRun(**run_data)

            if self.series_db_path.exists():
                with open(self.series_db_path) as f:
                    series_data = json.load(f)
                    for sid, series_dict in series_data.items():
                        if series_dict.get("created_time"):
                            series_dict["created_time"] = datetime.fromisoformat(series_dict["created_time"])
                        self._series[sid] = ExperimentSeries(**series_dict)

        except Exception as e:
            self.logger.warning("Failed to load from storage", error=str(e))

    def _save_to_storage(self):
        """Save data to storage files."""
        try:
            # Save experiments
            experiments_data = {eid: asdict(exp) for eid, exp in self._experiments.items()}
            with open(self.experiments_db_path, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)

            # Save runs
            runs_data = {rid: asdict(run) for rid, run in self._runs.items()}
            with open(self.runs_db_path, 'w') as f:
                json.dump(runs_data, f, indent=2, default=str)

            # Save series
            series_data = {sid: asdict(series) for sid, series in self._series.items()}
            with open(self.series_db_path, 'w') as f:
                json.dump(series_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error("Failed to save to storage", error=str(e))


# Global experiment manager instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager instance."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


# Convenience functions
def create_experiment(name: str, description: str, parameters: Dict[str, Any] = None,
                     tags: List[str] = None) -> str:
    """Convenience function to create an experiment."""
    config = ExperimentConfig(
        name=name,
        description=description,
        parameters=parameters or {},
        tags=tags or []
    )
    return get_experiment_manager().create_experiment(config)


def experiment_run(experiment_id: str, parameters: Dict[str, Any] = None):
    """Convenience function for experiment run context manager."""
    return get_experiment_manager().experiment_run(experiment_id, parameters)


def log_metric(run_id: str, name: str, value: float, step: int = 0):
    """Convenience function to log a metric."""
    get_experiment_manager().log_metric(run_id, name, value, step)


def log_result(run_id: str, key: str, value: Any):
    """Convenience function to log a result."""
    get_experiment_manager().log_result(run_id, key, value)


def save_artifact(run_id: str, artifact_path: Path, artifact_name: str = None) -> str:
    """Convenience function to save an artifact."""
    return get_experiment_manager().save_artifact(run_id, artifact_path, artifact_name)
