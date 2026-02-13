"""Information-theoretic GPU scheduling model with Bayesian learning."""

import math
import sqlite3
import json
import threading
from typing import Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import time


@dataclass
class JobFeatures:
    """Features extracted from a job request."""
    job_type: str
    gpu_mem_gb: float = 0.0
    model_params_m: float = 0.0
    batch_size: int = 1
    tenant: str = "default"


@dataclass
class Stats:
    """Statistics for a (job_type, gpu_type) pair."""
    mean_runtime: float
    variance: float
    count: int
    last_updated: float


@dataclass
class ScoringResult:
    """Result of scoring a job for GPU assignment."""
    chosen_gpu: str
    priority_score: float
    gpu_selector: Dict[str, Any]
    avoid_co_locate_with: List[str]
    terms: Dict[str, float]


class SchedulerModel:
    """Information-theoretic GPU scheduler model."""

    def __init__(self, config, storage_path: str = "/tmp/scheduler.db"):
        self.config = config
        self.storage_path = storage_path
        self.lock = threading.RLock()

        # In-memory stats: (job_type, gpu_type) -> Stats
        self.stats: Dict[Tuple[str, str], Stats] = {}

        # Interference matrix: job_type -> {co_job_type -> slowdown_factor}
        self.interference: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Exploration tracking
        self.exploration_decisions = 0
        self.total_decisions = 0

        # Initialize storage
        self._init_storage()
        self._load_from_storage()

    def _init_storage(self):
        """Initialize SQLite storage."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    job_type TEXT,
                    gpu_type TEXT,
                    mean_runtime REAL,
                    variance REAL,
                    count INTEGER,
                    last_updated REAL,
                    PRIMARY KEY (job_type, gpu_type)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS interference (
                    job_type TEXT,
                    co_job_type TEXT,
                    slowdown_factor REAL,
                    PRIMARY KEY (job_type, co_job_type)
                )
            """)

    def _load_from_storage(self):
        """Load stats from storage into memory."""
        with self.lock:
            try:
                with sqlite3.connect(self.storage_path) as conn:
                    # Load stats
                    cursor = conn.execute("SELECT * FROM stats")
                    for row in cursor:
                        job_type, gpu_type, mean_runtime, variance, count, last_updated = row
                        key = (job_type, gpu_type)
                        self.stats[key] = Stats(mean_runtime, variance, count, last_updated)

                    # Load interference
                    cursor = conn.execute("SELECT * FROM interference")
                    for row in cursor:
                        job_type, co_job_type, slowdown_factor = row
                        self.interference[job_type][co_job_type] = slowdown_factor
            except Exception as e:
                print(f"Warning: Could not load from storage: {e}")

    def checkpoint(self):
        """Save current state to storage."""
        with self.lock:
            try:
                with sqlite3.connect(self.storage_path) as conn:
                    # Clear and save stats
                    conn.execute("DELETE FROM stats")
                    for (job_type, gpu_type), stats in self.stats.items():
                        conn.execute("""
                            INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?)
                        """, (job_type, gpu_type, stats.mean_runtime, stats.variance,
                             stats.count, stats.last_updated))

                    # Clear and save interference
                    conn.execute("DELETE FROM interference")
                    for job_type, co_jobs in self.interference.items():
                        for co_job_type, slowdown in co_jobs.items():
                            if slowdown > 0:  # Only save non-zero interference
                                conn.execute("""
                                    INSERT INTO interference VALUES (?, ?, ?)
                                """, (job_type, co_job_type, slowdown))

                    conn.commit()
            except Exception as e:
                print(f"Warning: Could not checkpoint to storage: {e}")

    def get_stats(self, job_type: str, gpu_type: str) -> Tuple[float, float]:
        """Get mean and variance for a (job_type, gpu_type) pair."""
        with self.lock:
            key = (job_type, gpu_type)
            if key in self.stats:
                stats = self.stats[key]
                return stats.mean_runtime, stats.variance
            else:
                # Default uninformative prior: high mean, high variance
                return 10.0, 25.0

    def information_gain(self, variance: float) -> float:
        """Calculate information gain: IG = 0.5 * log(1 + σ²/τ²)."""
        return 0.5 * math.log(1 + variance / self.config.tau_squared)

    def interference_penalty(self, job_type: str, gpu_type: str,
                           colocated_types: List[str]) -> float:
        """Calculate interference penalty for job collocation."""
        if not self.config.enable_interference or not colocated_types:
            return 0.0

        penalty = 0.0
        for co_job in colocated_types:
            if co_job in self.interference[job_type]:
                penalty += self.interference[job_type][co_job]

        return penalty / len(colocated_types) if colocated_types else 0.0

    def score_job(self, features: JobFeatures, candidates: List[str],
                  colocated_types: List[str] = None) -> ScoringResult:
        """Score a job for GPU assignment using information-theoretic objective."""
        if colocated_types is None:
            colocated_types = []

        records = []
        for gpu_type in candidates:
            mu, sigma2 = self.get_stats(features.job_type, gpu_type)
            ig = self.information_gain(sigma2)
            penalty = self.interference_penalty(features.job_type, gpu_type, colocated_types)

            # cost(j,g) = μ + λ*√σ² - β*IG + γ*penalty
            cost = (mu +
                   self.config.lambda_uncertainty * math.sqrt(sigma2) -
                   self.config.beta_exploration * ig +
                   self.config.gamma_interference * penalty)

            records.append((gpu_type, mu, sigma2, ig, penalty, cost))

        # Choose minimum cost
        chosen_record = min(records, key=lambda r: r[-1])
        chosen_gpu, mu, sigma2, ig, penalty, cost = chosen_record

        # Check if this is an exploration decision
        is_exploration = sigma2 > 1.0  # High uncertainty threshold
        with self.lock:
            self.total_decisions += 1
            if is_exploration:
                self.exploration_decisions += 1

                # Enforce exploration cap
                exploration_ratio = self.exploration_decisions / self.total_decisions
                if exploration_ratio > self.config.exploration_cap:
                    # Fall back to exploitation (choose lowest mean)
                    exploit_record = min(records, key=lambda r: r[1])  # min by mu
                    chosen_gpu, mu, sigma2, ig, penalty, cost = exploit_record

        # Normalize cost to priority score [0, 100]
        costs = [r[-1] for r in records]
        if len(costs) > 1:
            max_cost, min_cost = max(costs), min(costs)
            if max_cost > min_cost:
                priority_score = 100 * (max_cost - cost) / (max_cost - min_cost)
            else:
                priority_score = 50.0  # All costs equal
        else:
            priority_score = 50.0

        # Generate GPU selector
        gpu_selector = {
            "nodeSelector": {"gpu.nvidia.com/class": chosen_gpu},
            "tolerations": [{
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule"
            }]
        }

        # Generate anti-affinity recommendations
        avoid_co_locate_with = []
        if self.config.enable_interference:
            for co_job in colocated_types:
                if self.interference[features.job_type].get(co_job, 0) > 0.1:  # 10% slowdown threshold
                    avoid_co_locate_with.append(co_job)

        return ScoringResult(
            chosen_gpu=chosen_gpu,
            priority_score=min(100.0, max(0.0, priority_score)),
            gpu_selector=gpu_selector,
            avoid_co_locate_with=avoid_co_locate_with,
            terms={
                "mu": mu,
                "sigma": math.sqrt(sigma2),
                "ig": ig,
                "penalty": penalty,
                "cost": cost
            }
        )

    def observe(self, job_type: str, gpu_type: str, runtime: float,
                colocated_types: List[str] = None):
        """Update model with observed runtime."""
        if colocated_types is None:
            colocated_types = []

        with self.lock:
            key = (job_type, gpu_type)
            current_time = time.time()

            if key in self.stats:
                # Bayesian update using exponential moving average
                stats = self.stats[key]
                alpha = 0.1  # Learning rate

                # Update mean: μ_new = (1-α)μ_old + α*observation
                new_mean = (1 - alpha) * stats.mean_runtime + alpha * runtime

                # Update variance: σ²_new = (1-α)σ²_old + α*(obs-μ_old)²
                error = runtime - stats.mean_runtime
                new_variance = (1 - alpha) * stats.variance + alpha * (error ** 2)

                self.stats[key] = Stats(
                    mean_runtime=new_mean,
                    variance=max(0.01, new_variance),  # Minimum variance
                    count=stats.count + 1,
                    last_updated=current_time
                )
            else:
                # Initialize with first observation
                self.stats[key] = Stats(
                    mean_runtime=runtime,
                    variance=1.0,  # Initial uncertainty
                    count=1,
                    last_updated=current_time
                )

            # Update interference if enabled and colocated jobs present
            if self.config.enable_interference and colocated_types:
                # Simple interference learning: track slowdown relative to solo runs
                solo_mean, _ = self.get_stats(job_type, gpu_type)
                if solo_mean > 0:
                    slowdown = max(0.0, (runtime - solo_mean) / solo_mean)

                    for co_job in colocated_types:
                        current_interference = self.interference[job_type][co_job]
                        # EMA update for interference
                        self.interference[job_type][co_job] = (
                            0.9 * current_interference + 0.1 * slowdown
                        )

            # Retention policy
            self._apply_retention_policy()

    def _apply_retention_policy(self):
        """Apply retention policy to limit memory usage."""
        if len(self.stats) <= self.config.retention_observations:
            return

        # Remove oldest entries
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1].last_updated
        )

        to_remove = len(self.stats) - self.config.retention_observations
        for i in range(to_remove):
            del self.stats[sorted_stats[i][0]]

    def get_metrics(self) -> Dict[str, float]:
        """Get model metrics for monitoring."""
        with self.lock:
            exploration_ratio = (
                self.exploration_decisions / max(1, self.total_decisions)
            )

            avg_uncertainty = 0.0
            if self.stats:
                avg_uncertainty = sum(
                    math.sqrt(s.variance) for s in self.stats.values()
                ) / len(self.stats)

            return {
                "scheduler_decisions_total": self.total_decisions,
                "exploration_ratio": exploration_ratio,
                "avg_uncertainty": avg_uncertainty,
                "stats_count": len(self.stats),
                "interference_pairs": sum(
                    len(co_jobs) for co_jobs in self.interference.values()
                )
            }
