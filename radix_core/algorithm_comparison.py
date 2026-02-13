"""
GPU Orchestration Algorithm Comparison Framework

This module provides comprehensive benchmarking and comparison capabilities
for different GPU orchestration algorithms, including:

1. Classical Approaches:
   - Round Robin scheduling
   - Shortest Job First (SJF)
   - Bin Packing algorithms
   - Kubernetes-style scheduling
   - Industry standard orchestrators

2. Information Theory Approaches:
   - Entropy-based scheduling
   - Mutual information optimization
   - Channel capacity maximization

3. Quantum Information Theory Approaches:
   - Quantum entanglement orchestration
   - Superposition-based exploration
   - Quantum error correction

The framework measures performance across multiple dimensions:
- Throughput (jobs/second)
- Resource utilization efficiency
- Latency distribution
- Fairness metrics
- Cost efficiency
- Fault tolerance
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

from .types import Job, ResourceRequirements, ExecutionResult
from .quantum_orchestration import QuantumEntanglementOrchestrator
from .information_theory import InformationTheoryOrchestrator
from .logging import get_logger
from .utils.timers import time_operation

logger = get_logger(__name__)


class AlgorithmType(Enum):
    """Types of orchestration algorithms."""
    CLASSICAL_ROUND_ROBIN = "classical_round_robin"
    CLASSICAL_SJF = "classical_shortest_job_first"
    CLASSICAL_BIN_PACKING = "classical_bin_packing"
    CLASSICAL_KUBERNETES = "classical_kubernetes_style"
    INFORMATION_THEORY = "information_theory"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    HYBRID_QUANTUM_CLASSICAL = "hybrid_quantum_classical"


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for algorithm performance."""

    # Throughput metrics
    jobs_per_second: float = 0.0
    total_execution_time: float = 0.0
    scheduling_overhead: float = 0.0

    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    resource_efficiency: float = 0.0

    # Latency metrics
    mean_latency: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Fairness and optimization
    fairness_index: float = 0.0  # Jain's fairness index
    load_balance_score: float = 0.0
    optimization_score: float = 0.0

    # Cost and efficiency
    cost_efficiency: float = 0.0
    energy_efficiency: float = 0.0

    # Fault tolerance
    failure_recovery_time: float = 0.0
    error_correction_rate: float = 0.0

    # Algorithm-specific metrics
    algorithm_specific: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'throughput': {
                'jobs_per_second': self.jobs_per_second,
                'total_execution_time': self.total_execution_time,
                'scheduling_overhead': self.scheduling_overhead
            },
            'resource_utilization': {
                'cpu_utilization': self.cpu_utilization,
                'memory_utilization': self.memory_utilization,
                'gpu_utilization': self.gpu_utilization,
                'resource_efficiency': self.resource_efficiency
            },
            'latency': {
                'mean_latency': self.mean_latency,
                'median_latency': self.median_latency,
                'p95_latency': self.p95_latency,
                'p99_latency': self.p99_latency
            },
            'optimization': {
                'fairness_index': self.fairness_index,
                'load_balance_score': self.load_balance_score,
                'optimization_score': self.optimization_score
            },
            'efficiency': {
                'cost_efficiency': self.cost_efficiency,
                'energy_efficiency': self.energy_efficiency
            },
            'fault_tolerance': {
                'failure_recovery_time': self.failure_recovery_time,
                'error_correction_rate': self.error_correction_rate
            },
            'algorithm_specific': self.algorithm_specific
        }


@dataclass
class ComparisonResult:
    """Result of algorithm comparison."""

    algorithm_type: AlgorithmType
    metrics: BenchmarkMetrics
    relative_performance: Dict[str, float]  # Relative to baseline
    statistical_significance: Dict[str, float]  # p-values
    recommendation: str
    confidence_score: float


class ClassicalOrchestrators:
    """Implementation of classical GPU orchestration algorithms."""

    @staticmethod
    def round_robin_schedule(jobs: List[Job], resources: List[str]) -> List[Tuple[Job, str]]:
        """Simple round-robin scheduling."""
        allocations = []
        resource_index = 0

        for job in jobs:
            resource = resources[resource_index % len(resources)]
            allocations.append((job, resource))
            resource_index += 1

        return allocations

    @staticmethod
    def shortest_job_first(jobs: List[Job], resources: List[str]) -> List[Tuple[Job, str]]:
        """Shortest Job First scheduling."""
        # Sort jobs by estimated execution time (using CPU requirements as proxy)
        sorted_jobs = sorted(jobs, key=lambda j: j.requirements.cpu_cores)

        allocations = []
        resource_loads = {r: 0 for r in resources}

        for job in sorted_jobs:
            # Assign to least loaded resource
            best_resource = min(resource_loads.keys(), key=lambda r: resource_loads[r])
            allocations.append((job, best_resource))
            resource_loads[best_resource] += job.requirements.cpu_cores

        return allocations

    @staticmethod
    def bin_packing_schedule(jobs: List[Job], resources: List[str]) -> List[Tuple[Job, str]]:
        """Best-fit bin packing algorithm."""
        allocations = []
        resource_capacities = {r: 100.0 for r in resources}  # Simulated capacity

        # Sort jobs by resource requirements (largest first)
        sorted_jobs = sorted(jobs, key=lambda j: j.requirements.cpu_cores + j.requirements.memory_mb/1000, reverse=True)

        for job in sorted_jobs:
            job_size = job.requirements.cpu_cores + job.requirements.memory_mb/1000

            # Find best-fit resource
            best_resource = None
            best_fit = float('inf')

            for resource in resources:
                if resource_capacities[resource] >= job_size:
                    remaining = resource_capacities[resource] - job_size
                    if remaining < best_fit:
                        best_fit = remaining
                        best_resource = resource

            if best_resource:
                allocations.append((job, best_resource))
                resource_capacities[best_resource] -= job_size
            else:
                # If no fit, assign to resource with most capacity
                best_resource = max(resource_capacities.keys(), key=lambda r: resource_capacities[r])
                allocations.append((job, best_resource))
                resource_capacities[best_resource] = max(0, resource_capacities[best_resource] - job_size)

        return allocations

    @staticmethod
    def kubernetes_style_schedule(jobs: List[Job], resources: List[str]) -> List[Tuple[Job, str]]:
        """Kubernetes-style priority and resource-aware scheduling."""
        allocations = []

        # Sort jobs by priority (higher first), then by resource requirements
        sorted_jobs = sorted(jobs, key=lambda j: (-j.priority, -j.requirements.cpu_cores))

        resource_scores = {}
        for resource in resources:
            resource_scores[resource] = {
                'load': 0.0,
                'suitability': 0.0
            }

        for job in sorted_jobs:
            best_resource = None
            best_score = -float('inf')

            for resource in resources:
                # Calculate resource score based on current load and job fit
                load_penalty = resource_scores[resource]['load'] * 0.5
                priority_bonus = job.priority * 0.3
                resource_fit = min(job.requirements.cpu_cores / 32.0, 1.0) * 0.2

                score = priority_bonus + resource_fit - load_penalty

                if score > best_score:
                    best_score = score
                    best_resource = resource

            if best_resource:
                allocations.append((job, best_resource))
                resource_scores[best_resource]['load'] += job.requirements.cpu_cores / 32.0

        return allocations


class AlgorithmBenchmark:
    """Comprehensive benchmarking system for orchestration algorithms."""

    def __init__(self):
        self.quantum_orchestrator = QuantumEntanglementOrchestrator()
        self.info_theory_orchestrator = InformationTheoryOrchestrator()
        self.classical_orchestrators = ClassicalOrchestrators()

        # Simulated GPU resources
        self.resources = [f"gpu_cluster_{i}" for i in range(8)]

        # Benchmark parameters
        self.benchmark_duration = 30.0  # seconds
        self.warmup_jobs = 50
        self.measurement_jobs = 200

    def create_benchmark_workload(self,
                                workload_type: str = "mixed",
                                job_count: int = 200) -> List[Job]:
        """Create representative workload for benchmarking."""

        jobs = []

        if workload_type == "compute_intensive":
            # High CPU, moderate memory
            for i in range(job_count):
                job = Job(
                    job_id=f"compute_{i}",
                    name=f"compute_job_{i}",
                    command="python compute_intensive.py",
                    requirements=ResourceRequirements(
                        cpu_cores=np.random.randint(8, 32),
                        memory_mb=np.random.randint(4000, 16000)
                    ),
                    priority=np.random.randint(1, 10)
                )
                jobs.append(job)

        elif workload_type == "memory_intensive":
            # Moderate CPU, high memory
            for i in range(job_count):
                job = Job(
                    job_id=f"memory_{i}",
                    name=f"memory_job_{i}",
                    command="python memory_intensive.py",
                    requirements=ResourceRequirements(
                        cpu_cores=np.random.randint(2, 16),
                        memory_mb=np.random.randint(16000, 64000)
                    ),
                    priority=np.random.randint(1, 10)
                )
                jobs.append(job)

        elif workload_type == "ml_training":
            # ML training workload characteristics
            for i in range(job_count):
                job = Job(
                    job_id=f"ml_train_{i}",
                    name=f"ml_training_{i}",
                    command="python train_model.py",
                    requirements=ResourceRequirements(
                        cpu_cores=np.random.randint(4, 16),
                        memory_mb=np.random.randint(8000, 32000)
                    ),
                    priority=np.random.randint(3, 8),
                    metadata={"workload_type": "ml_training"}
                )
                jobs.append(job)

        else:  # mixed workload
            # Combination of different job types
            for i in range(job_count):
                job_type = np.random.choice(["compute", "memory", "ml", "inference"])

                if job_type == "compute":
                    cpu_cores = np.random.randint(8, 32)
                    memory_mb = np.random.randint(4000, 16000)
                elif job_type == "memory":
                    cpu_cores = np.random.randint(2, 16)
                    memory_mb = np.random.randint(16000, 64000)
                elif job_type == "ml":
                    cpu_cores = np.random.randint(4, 16)
                    memory_mb = np.random.randint(8000, 32000)
                else:  # inference
                    cpu_cores = np.random.randint(1, 8)
                    memory_mb = np.random.randint(2000, 8000)

                job = Job(
                    job_id=f"{job_type}_{i}",
                    name=f"{job_type}_job_{i}",
                    command=f"python {job_type}_task.py",
                    requirements=ResourceRequirements(
                        cpu_cores=cpu_cores,
                        memory_mb=memory_mb
                    ),
                    priority=np.random.randint(1, 10),
                    metadata={"workload_type": job_type}
                )
                jobs.append(job)

        return jobs

    def benchmark_algorithm(self,
                          algorithm_type: AlgorithmType,
                          workload: List[Job],
                          iterations: int = 5) -> BenchmarkMetrics:
        """Benchmark a specific algorithm with given workload."""

        all_metrics = []

        for iteration in range(iterations):
            logger.info(f"Running benchmark iteration {iteration + 1}/{iterations}",
                       algorithm=algorithm_type.value)

            metrics = self._single_benchmark_run(algorithm_type, workload.copy())
            all_metrics.append(metrics)

        # Aggregate metrics across iterations
        return self._aggregate_metrics(all_metrics)

    def _single_benchmark_run(self,
                            algorithm_type: AlgorithmType,
                            workload: List[Job]) -> BenchmarkMetrics:
        """Run single benchmark iteration."""

        start_time = time.time()

        # Schedule jobs using specified algorithm
        with time_operation(f"benchmark_{algorithm_type.value}"):
            if algorithm_type == AlgorithmType.CLASSICAL_ROUND_ROBIN:
                allocations = self.classical_orchestrators.round_robin_schedule(workload, self.resources)

            elif algorithm_type == AlgorithmType.CLASSICAL_SJF:
                allocations = self.classical_orchestrators.shortest_job_first(workload, self.resources)

            elif algorithm_type == AlgorithmType.CLASSICAL_BIN_PACKING:
                allocations = self.classical_orchestrators.bin_packing_schedule(workload, self.resources)

            elif algorithm_type == AlgorithmType.CLASSICAL_KUBERNETES:
                allocations = self.classical_orchestrators.kubernetes_style_schedule(workload, self.resources)

            elif algorithm_type == AlgorithmType.INFORMATION_THEORY:
                allocations = [(job, "gpu_cluster_0") for job in workload]  # Placeholder
                # In reality: allocations = self.info_theory_orchestrator.schedule(workload)

            elif algorithm_type == AlgorithmType.QUANTUM_ENTANGLEMENT:
                quantum_allocations = self.quantum_orchestrator.quantum_orchestrate(workload)
                allocations = quantum_allocations

            else:
                raise ValueError(f"Unknown algorithm type: {algorithm_type}")

        scheduling_time = time.time() - start_time

        # Simulate execution and measure metrics
        metrics = self._measure_execution_metrics(allocations, scheduling_time, algorithm_type)

        return metrics

    def _measure_execution_metrics(self,
                                 allocations: List[Tuple[Job, str]],
                                 scheduling_time: float,
                                 algorithm_type: AlgorithmType) -> BenchmarkMetrics:
        """Measure execution metrics for given allocations."""

        # Simulate job execution times and resource usage
        job_latencies = []
        resource_utilization = {resource: 0.0 for resource in self.resources}

        for job, resource in allocations:
            # Simulate execution time based on job requirements
            base_time = job.requirements.cpu_cores * 0.1 + job.requirements.memory_mb / 10000
            execution_time = base_time + np.random.exponential(base_time * 0.2)
            job_latencies.append(execution_time)

            # Track resource utilization
            resource_utilization[resource] += job.requirements.cpu_cores

        # Calculate metrics
        metrics = BenchmarkMetrics()

        # Throughput metrics
        total_execution_time = max(job_latencies) if job_latencies else 0
        metrics.jobs_per_second = len(allocations) / max(total_execution_time, 0.001)
        metrics.total_execution_time = total_execution_time
        metrics.scheduling_overhead = scheduling_time

        # Latency metrics
        if job_latencies:
            metrics.mean_latency = statistics.mean(job_latencies)
            metrics.median_latency = statistics.median(job_latencies)
            metrics.p95_latency = np.percentile(job_latencies, 95)
            metrics.p99_latency = np.percentile(job_latencies, 99)

        # Resource utilization
        utilizations = list(resource_utilization.values())
        if utilizations:
            metrics.cpu_utilization = statistics.mean(utilizations) / 32.0  # Normalize by max CPU
            metrics.resource_efficiency = 1.0 - (statistics.stdev(utilizations) / max(statistics.mean(utilizations), 1))

        # Fairness (Jain's fairness index)
        if utilizations:
            sum_utilization = sum(utilizations)
            sum_squares = sum(u**2 for u in utilizations)
            n = len(utilizations)
            if sum_squares > 0:
                metrics.fairness_index = (sum_utilization**2) / (n * sum_squares)

        # Load balance
        if utilizations:
            max_util = max(utilizations)
            min_util = min(utilizations)
            metrics.load_balance_score = 1.0 - ((max_util - min_util) / max(max_util, 1))

        # Algorithm-specific metrics
        if algorithm_type == AlgorithmType.QUANTUM_ENTANGLEMENT:
            quantum_metrics = self.quantum_orchestrator.get_quantum_metrics()
            metrics.algorithm_specific.update(quantum_metrics)

        # Cost efficiency (simulated)
        metrics.cost_efficiency = metrics.resource_efficiency * metrics.fairness_index

        # Optimization score (composite)
        metrics.optimization_score = (
            metrics.fairness_index * 0.3 +
            metrics.load_balance_score * 0.3 +
            metrics.resource_efficiency * 0.4
        )

        return metrics

    def _aggregate_metrics(self, metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
        """Aggregate metrics across multiple benchmark runs."""

        if not metrics_list:
            return BenchmarkMetrics()

        aggregated = BenchmarkMetrics()

        # Aggregate each metric field
        aggregated.jobs_per_second = statistics.mean(m.jobs_per_second for m in metrics_list)
        aggregated.total_execution_time = statistics.mean(m.total_execution_time for m in metrics_list)
        aggregated.scheduling_overhead = statistics.mean(m.scheduling_overhead for m in metrics_list)

        aggregated.cpu_utilization = statistics.mean(m.cpu_utilization for m in metrics_list)
        aggregated.memory_utilization = statistics.mean(m.memory_utilization for m in metrics_list)
        aggregated.resource_efficiency = statistics.mean(m.resource_efficiency for m in metrics_list)

        aggregated.mean_latency = statistics.mean(m.mean_latency for m in metrics_list)
        aggregated.median_latency = statistics.mean(m.median_latency for m in metrics_list)
        aggregated.p95_latency = statistics.mean(m.p95_latency for m in metrics_list)
        aggregated.p99_latency = statistics.mean(m.p99_latency for m in metrics_list)

        aggregated.fairness_index = statistics.mean(m.fairness_index for m in metrics_list)
        aggregated.load_balance_score = statistics.mean(m.load_balance_score for m in metrics_list)
        aggregated.optimization_score = statistics.mean(m.optimization_score for m in metrics_list)

        aggregated.cost_efficiency = statistics.mean(m.cost_efficiency for m in metrics_list)

        # Aggregate algorithm-specific metrics
        if metrics_list[0].algorithm_specific:
            for key in metrics_list[0].algorithm_specific:
                values = [m.algorithm_specific.get(key, 0) for m in metrics_list]
                aggregated.algorithm_specific[key] = statistics.mean(values)

        return aggregated

    def compare_algorithms(self,
                         algorithms: List[AlgorithmType],
                         workload_types: List[str] = None,
                         job_counts: List[int] = None) -> Dict[str, ComparisonResult]:
        """
        Comprehensive comparison of multiple algorithms across different workloads.

        Returns detailed analysis of which algorithms perform best for different scenarios.
        """
        if workload_types is None:
            workload_types = ["mixed", "compute_intensive", "memory_intensive", "ml_training"]

        if job_counts is None:
            job_counts = [100, 200, 500]

        results = {}

        for workload_type in workload_types:
            for job_count in job_counts:
                test_name = f"{workload_type}_{job_count}_jobs"

                logger.info("Running algorithm comparison",
                           test_name=test_name,
                           algorithms=[a.value for a in algorithms])

                # Create workload
                workload = self.create_benchmark_workload(workload_type, job_count)

                # Benchmark each algorithm
                algorithm_results = {}
                for algorithm in algorithms:
                    metrics = self.benchmark_algorithm(algorithm, workload)
                    algorithm_results[algorithm] = metrics

                # Analyze results
                analysis = self._analyze_comparison_results(algorithm_results, test_name)
                results[test_name] = analysis

        return results

    def _analyze_comparison_results(self,
                                  algorithm_results: Dict[AlgorithmType, BenchmarkMetrics],
                                  test_name: str) -> Dict[str, Any]:
        """Analyze comparison results and provide recommendations."""

        analysis = {
            'test_name': test_name,
            'algorithms': {},
            'rankings': {},
            'recommendations': {},
            'statistical_analysis': {}
        }

        # Store individual algorithm results
        for algorithm, metrics in algorithm_results.items():
            analysis['algorithms'][algorithm.value] = metrics.to_dict()

        # Calculate rankings for key metrics
        ranking_metrics = [
            'jobs_per_second', 'resource_efficiency', 'fairness_index',
            'optimization_score', 'cost_efficiency'
        ]

        for metric in ranking_metrics:
            # Sort algorithms by metric performance
            sorted_algs = sorted(
                algorithm_results.items(),
                key=lambda x: getattr(x[1], metric),
                reverse=True
            )

            analysis['rankings'][metric] = [
                {
                    'algorithm': alg.value,
                    'value': getattr(metrics, metric),
                    'rank': i + 1
                }
                for i, (alg, metrics) in enumerate(sorted_algs)
            ]

        # Overall recommendation
        best_overall = self._calculate_overall_winner(algorithm_results)
        analysis['recommendations']['best_overall'] = best_overall

        # Specific use case recommendations
        analysis['recommendations']['use_cases'] = self._generate_use_case_recommendations(algorithm_results)

        return analysis

    def _calculate_overall_winner(self,
                                algorithm_results: Dict[AlgorithmType, BenchmarkMetrics]) -> Dict[str, Any]:
        """Calculate overall best performing algorithm."""

        # Composite score weighing different factors
        scores = {}

        for algorithm, metrics in algorithm_results.items():
            # Weighted composite score
            score = (
                metrics.jobs_per_second * 0.2 +
                metrics.resource_efficiency * 0.25 +
                metrics.fairness_index * 0.2 +
                metrics.optimization_score * 0.2 +
                metrics.cost_efficiency * 0.15
            )

            scores[algorithm] = {
                'composite_score': score,
                'metrics': metrics
            }

        # Find best performing algorithm
        best_algorithm = max(scores.keys(), key=lambda a: scores[a]['composite_score'])

        return {
            'algorithm': best_algorithm.value,
            'composite_score': scores[best_algorithm]['composite_score'],
            'confidence': 0.85,  # Placeholder - would calculate from statistical analysis
            'improvement_over_baseline': self._calculate_improvement(scores, best_algorithm)
        }

    def _calculate_improvement(self, scores: Dict[AlgorithmType, Dict], best_algorithm: AlgorithmType) -> float:
        """Calculate improvement over baseline (round robin)."""

        baseline_score = scores.get(AlgorithmType.CLASSICAL_ROUND_ROBIN, {}).get('composite_score', 0)
        best_score = scores[best_algorithm]['composite_score']

        if baseline_score > 0:
            return ((best_score - baseline_score) / baseline_score) * 100
        else:
            return 0.0

    def _generate_use_case_recommendations(self,
                                         algorithm_results: Dict[AlgorithmType, BenchmarkMetrics]) -> Dict[str, str]:
        """Generate specific use case recommendations."""

        recommendations = {}

        # Best for throughput
        best_throughput = max(algorithm_results.items(), key=lambda x: x[1].jobs_per_second)
        recommendations['high_throughput'] = best_throughput[0].value

        # Best for resource efficiency
        best_efficiency = max(algorithm_results.items(), key=lambda x: x[1].resource_efficiency)
        recommendations['resource_efficiency'] = best_efficiency[0].value

        # Best for fairness
        best_fairness = max(algorithm_results.items(), key=lambda x: x[1].fairness_index)
        recommendations['fairness'] = best_fairness[0].value

        # Best for low latency
        best_latency = min(algorithm_results.items(), key=lambda x: x[1].mean_latency)
        recommendations['low_latency'] = best_latency[0].value

        return recommendations


def create_algorithm_benchmark() -> AlgorithmBenchmark:
    """Factory function to create algorithm benchmark instance."""
    return AlgorithmBenchmark()
