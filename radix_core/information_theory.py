"""
Information Theory for GPU Orchestration

Applies information theory principles to optimize GPU orchestration,
including entropy-based scheduling, information-theoretic resource allocation,
and mutual information analysis for workload optimization.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .logging import get_logger, trace_operation
from .types import Job

logger = get_logger(__name__)


@dataclass
class InformationMetrics:
    """Information-theoretic metrics for analysis."""

    entropy: float
    mutual_information: float
    conditional_entropy: float
    relative_entropy: float
    cross_entropy: float
    information_gain: float


@dataclass
class ResourceInformation:
    """Information content of resource utilization patterns."""

    resource_type: str
    entropy_bits: float
    predictability_score: float
    compression_ratio: float
    pattern_complexity: float
    information_density: float


class InformationTheoryOrchestrator:
    """GPU orchestration system using information theory principles."""

    def __init__(self):
        self.logger = get_logger("radix.information_theory")

        # Historical data for learning
        self.job_history: List[Job] = []
        self.resource_utilization_history: List[Dict[str, float]] = []
        self.performance_history: List[Dict[str, float]] = []

        # Information-theoretic models
        self.entropy_cache: Dict[str, float] = {}
        self.mutual_info_cache: Dict[Tuple[str, str], float] = {}

    @trace_operation("entropy_based_scheduling")
    def entropy_based_scheduling(self, jobs: List[Job]) -> List[Job]:
        """
        Schedule jobs based on entropy minimization.

        Lower entropy in resource allocation leads to more predictable
        and efficient GPU utilization patterns.
        """
        if not jobs:
            return []

        # Calculate entropy for different scheduling arrangements
        best_order = jobs.copy()
        best_entropy = float("inf")

        # Try different permutations (limited for performance)
        import itertools

        max_permutations = min(math.factorial(len(jobs)), 1000)

        for i, perm in enumerate(itertools.permutations(jobs)):
            if i >= max_permutations:
                break

            entropy = self._calculate_schedule_entropy(list(perm))
            if entropy < best_entropy:
                best_entropy = entropy
                best_order = list(perm)

        self.logger.info(
            "Entropy-based scheduling completed",
            original_entropy=self._calculate_schedule_entropy(jobs),
            optimized_entropy=best_entropy,
            improvement=self._calculate_schedule_entropy(jobs) - best_entropy,
        )

        return best_order

    def mutual_information_resource_allocation(
        self, jobs: List[Job], available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Allocate resources based on mutual information between job characteristics
        and resource requirements to maximize information efficiency.
        """
        allocations = {}

        for job in jobs:
            # Calculate mutual information between job features and resource types
            job_features = self._extract_job_features(job)

            optimal_allocation = {}
            for resource_type, available_amount in available_resources.items():
                # Find allocation that maximizes mutual information
                mi_scores = []
                allocation_candidates = np.linspace(0, available_amount, 10)

                for candidate_allocation in allocation_candidates:
                    mi = self._calculate_mutual_information(
                        job_features, resource_type, candidate_allocation
                    )
                    mi_scores.append((candidate_allocation, mi))

                # Select allocation with highest mutual information
                best_allocation = max(mi_scores, key=lambda x: x[1])[0]
                optimal_allocation[resource_type] = best_allocation

            allocations[job.job_id] = optimal_allocation

        return allocations

    def information_theoretic_batching(
        self, jobs: List[Job], max_batch_size: int = 32
    ) -> List[List[Job]]:
        """
        Create batches that minimize information loss and maximize compression efficiency.
        """
        if not jobs:
            return []

        # Calculate information content for each job
        job_info_content = [(job, self._calculate_job_information_content(job)) for job in jobs]

        # Sort by information content
        job_info_content.sort(key=lambda x: x[1])

        batches = []
        current_batch = []
        current_batch_entropy = 0.0

        for job, info_content in job_info_content:
            # Calculate entropy if we add this job to current batch
            test_batch = current_batch + [job]
            test_entropy = self._calculate_batch_entropy(test_batch)

            # Add to current batch if it doesn't increase entropy too much
            # or if current batch is empty
            if (
                len(current_batch) == 0
                or test_entropy - current_batch_entropy < self._get_entropy_threshold()
                or len(current_batch) >= max_batch_size
            ):

                if len(current_batch) >= max_batch_size:
                    # Start new batch
                    batches.append(current_batch)
                    current_batch = [job]
                    current_batch_entropy = self._calculate_batch_entropy([job])
                else:
                    # Add to current batch
                    current_batch.append(job)
                    current_batch_entropy = test_entropy
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [job]
                current_batch_entropy = self._calculate_batch_entropy([job])

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        self.logger.info(
            "Information-theoretic batching completed",
            total_jobs=len(jobs),
            num_batches=len(batches),
            avg_batch_size=len(jobs) / len(batches) if batches else 0,
        )

        return batches

    def kolmogorov_complexity_optimization(
        self, workload_pattern: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize workload execution based on Kolmogorov complexity estimation.

        Aims to find the shortest program (lowest complexity) that can generate
        the desired workload execution pattern.
        """
        # Estimate Kolmogorov complexity using compression
        pattern_str = str(workload_pattern)

        # Try different compression approaches
        import lzma
        import zlib

        original_size = len(pattern_str.encode())
        zlib_compressed = len(zlib.compress(pattern_str.encode()))
        lzma_compressed = len(lzma.compress(pattern_str.encode()))

        # Use best compression as complexity estimate
        best_compression = min(zlib_compressed, lzma_compressed)
        complexity_ratio = best_compression / original_size

        # Generate optimization recommendations
        optimization_strategy = {
            "complexity_ratio": complexity_ratio,
            "compressibility_score": 1.0 - complexity_ratio,
            "optimization_potential": (
                "high" if complexity_ratio > 0.7 else "medium" if complexity_ratio > 0.4 else "low"
            ),
            "recommended_actions": self._generate_complexity_optimizations(complexity_ratio),
        }

        return optimization_strategy

    def shannon_entropy_load_balancing(
        self, nodes: List[Dict[str, Any]], jobs: List[Job]
    ) -> Dict[str, List[str]]:
        """
        Distribute jobs across nodes to minimize Shannon entropy of load distribution.
        """
        # Calculate current load entropy for each node
        node_loads = {node["id"]: node.get("current_load", 0.0) for node in nodes}

        # Assignment that minimizes entropy
        assignments = defaultdict(list)

        for job in jobs:
            job_load = self._estimate_job_load(job)

            # Find assignment that minimizes total entropy
            best_node = None
            best_entropy = float("inf")

            for node_id in node_loads:
                # Calculate entropy if we assign job to this node
                test_loads = node_loads.copy()
                test_loads[node_id] += job_load

                entropy = self._calculate_load_distribution_entropy(list(test_loads.values()))

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_node = node_id

            # Assign job to best node
            if best_node:
                assignments[best_node].append(job.job_id)
                node_loads[best_node] += job_load

        return dict(assignments)

    def information_gain_guided_scheduling(
        self, jobs: List[Job], historical_performance: List[Dict[str, Any]]
    ) -> List[Job]:
        """
        Schedule jobs to maximize information gain about system performance.
        """
        if not historical_performance:
            return jobs  # No historical data, return as-is

        # Calculate information gain for different scheduling orders
        job_performance_correlations = self._calculate_job_performance_correlations(
            jobs, historical_performance
        )

        # Schedule jobs to maximize information gain
        scheduled_jobs = []
        remaining_jobs = jobs.copy()

        while remaining_jobs:
            best_job = None
            best_info_gain = -float("inf")

            for job in remaining_jobs:
                # Calculate expected information gain if we schedule this job next
                info_gain = self._calculate_information_gain(
                    job, scheduled_jobs, job_performance_correlations
                )

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_job = job

            if best_job:
                scheduled_jobs.append(best_job)
                remaining_jobs.remove(best_job)

        return scheduled_jobs

    def compression_based_resource_prediction(
        self, historical_usage: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Predict future resource usage based on compression patterns in historical data.
        """
        predictions = {}

        for resource_type in ["cpu", "memory", "gpu"]:
            if not historical_usage:
                continue

            # Extract time series for this resource
            time_series = [usage.get(resource_type, 0.0) for usage in historical_usage]

            if not time_series:
                continue

            # Analyze compression patterns
            compression_analysis = self._analyze_compression_patterns(time_series)

            # Generate predictions based on compression insights
            predictions[resource_type] = {
                "next_value_prediction": self._predict_next_value(
                    time_series, compression_analysis
                ),
                "pattern_complexity": compression_analysis["complexity"],
                "predictability_score": compression_analysis["predictability"],
                "trend_entropy": compression_analysis["trend_entropy"],
            }

        return predictions

    def _calculate_schedule_entropy(self, jobs: List[Job]) -> float:
        """Calculate entropy of a job schedule based on resource requirements."""
        if not jobs:
            return 0.0

        # Extract resource requirement patterns
        resource_patterns = []
        for job in jobs:
            pattern = (
                job.requirements.cpu_cores,
                job.requirements.memory_mb,
                job.requirements.gpu_count,
                job.priority,
            )
            resource_patterns.append(pattern)

        # Calculate entropy of the pattern distribution
        pattern_counts = Counter(resource_patterns)
        total = len(resource_patterns)

        entropy = 0.0
        for count in pattern_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _extract_job_features(self, job: Job) -> Dict[str, float]:
        """Extract numerical features from a job for information analysis."""
        return {
            "cpu_cores": float(job.requirements.cpu_cores),
            "memory_gb": job.requirements.memory_mb,
            "gpu_count": float(job.requirements.gpu_count),
            "priority": float(job.priority),
            "max_runtime": float(job.max_runtime_seconds),
            "name_length": float(len(job.name)),
            "has_dependencies": float(len(job.dependencies) > 0),
        }

    def _calculate_mutual_information(
        self, job_features: Dict[str, float], resource_type: str, allocation: float
    ) -> float:
        """Calculate mutual information between job features and resource allocation."""
        # Simplified mutual information calculation
        # In practice, this would use historical data and proper MI estimation

        feature_entropy = sum(abs(v) for v in job_features.values()) / len(job_features)
        allocation_entropy = -allocation * math.log2(allocation + 1e-10)

        # Estimate mutual information (simplified)
        joint_entropy = feature_entropy + allocation_entropy
        mi = feature_entropy + allocation_entropy - joint_entropy

        return mi

    def _calculate_job_information_content(self, job: Job) -> float:
        """Calculate information content of a job."""
        features = self._extract_job_features(job)

        # Calculate entropy of job features
        feature_values = list(features.values())
        if not feature_values:
            return 0.0

        # Normalize features
        max_val = max(feature_values) if feature_values else 1.0
        normalized_features = [v / max_val for v in feature_values]

        # Calculate entropy
        entropy = 0.0
        for value in normalized_features:
            if value > 0:
                entropy -= value * math.log2(value)

        return entropy

    def _calculate_batch_entropy(self, batch: List[Job]) -> float:
        """Calculate entropy of a job batch."""
        if not batch:
            return 0.0

        # Combine all job features in batch
        all_features = []
        for job in batch:
            features = self._extract_job_features(job)
            all_features.extend(features.values())

        if not all_features:
            return 0.0

        # Calculate entropy of combined features
        total = sum(all_features)
        if total == 0:
            return 0.0

        probabilities = [v / total for v in all_features if v > 0]

        entropy = 0.0
        for p in probabilities:
            entropy -= p * math.log2(p)

        return entropy

    def _get_entropy_threshold(self) -> float:
        """Get entropy threshold for batching decisions."""
        return 1.5  # Configurable threshold

    def _estimate_job_load(self, job: Job) -> float:
        """Estimate computational load of a job."""
        return (
            job.requirements.cpu_cores * 1.0
            + job.requirements.memory_mb * 0.5
            + job.requirements.gpu_count * 2.0
            + job.max_runtime_seconds / 3600.0
        )

    def _calculate_load_distribution_entropy(self, loads: List[float]) -> float:
        """Calculate entropy of load distribution across nodes."""
        if not loads:
            return 0.0

        total_load = sum(loads)
        if total_load == 0:
            return 0.0

        probabilities = [load / total_load for load in loads if load > 0]

        entropy = 0.0
        for p in probabilities:
            entropy -= p * math.log2(p)

        return entropy

    def _calculate_job_performance_correlations(
        self, jobs: List[Job], historical_performance: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate correlations between job characteristics and performance."""
        correlations = {}

        # Simplified correlation calculation
        for job in jobs:
            features = self._extract_job_features(job)
            correlation_score = 0.0

            for perf_data in historical_performance:
                # Calculate similarity and correlation
                similarity = self._calculate_feature_similarity(features, perf_data)
                correlation_score += similarity

            correlations[job.job_id] = (
                correlation_score / len(historical_performance) if historical_performance else 0.0
            )

        return correlations

    def _calculate_information_gain(
        self, job: Job, scheduled_jobs: List[Job], correlations: Dict[str, float]
    ) -> float:
        """Calculate expected information gain from scheduling a job."""
        base_entropy = self._calculate_schedule_entropy(scheduled_jobs)
        new_entropy = self._calculate_schedule_entropy(scheduled_jobs + [job])

        # Information gain is reduction in entropy
        info_gain = base_entropy - new_entropy

        # Weight by performance correlation
        correlation_weight = correlations.get(job.job_id, 0.0)

        return info_gain * (1.0 + correlation_weight)

    def _analyze_compression_patterns(self, time_series: List[float]) -> Dict[str, float]:
        """Analyze compression patterns in time series data."""
        if not time_series:
            return {"complexity": 0.0, "predictability": 0.0, "trend_entropy": 0.0}

        # Convert to string for compression analysis
        series_str = ",".join(f"{x:.3f}" for x in time_series)

        # Calculate compression ratio
        import zlib

        original_size = len(series_str.encode())
        compressed_size = len(zlib.compress(series_str.encode()))
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        # Calculate trend entropy
        diffs = [time_series[i + 1] - time_series[i] for i in range(len(time_series) - 1)]
        trend_entropy = self._calculate_sequence_entropy(diffs)

        return {
            "complexity": compression_ratio,
            "predictability": 1.0 - compression_ratio,
            "trend_entropy": trend_entropy,
        }

    def _predict_next_value(
        self, time_series: List[float], compression_analysis: Dict[str, float]
    ) -> float:
        """Predict next value in time series based on compression patterns."""
        if len(time_series) < 2:
            return time_series[-1] if time_series else 0.0

        # Simple prediction based on trend and complexity
        recent_trend = time_series[-1] - time_series[-2]
        complexity = compression_analysis["complexity"]

        # Adjust prediction based on complexity
        if complexity < 0.5:  # Low complexity, high predictability
            prediction = time_series[-1] + recent_trend
        else:  # High complexity, use average
            prediction = sum(time_series[-min(5, len(time_series)) :]) / min(5, len(time_series))

        return prediction

    def _calculate_feature_similarity(
        self, features1: Dict[str, float], features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between feature sets."""
        # Convert features2 to numerical format
        numerical_features2 = {}
        for key, value in features2.items():
            try:
                numerical_features2[key] = float(value)
            except (ValueError, TypeError):
                numerical_features2[key] = 0.0

        # Calculate cosine similarity
        common_keys = set(features1.keys()) & set(numerical_features2.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(features1[key] * numerical_features2[key] for key in common_keys)
        norm1 = math.sqrt(sum(features1[key] ** 2 for key in common_keys))
        norm2 = math.sqrt(sum(numerical_features2[key] ** 2 for key in common_keys))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_sequence_entropy(self, sequence: List[float]) -> float:
        """Calculate entropy of a numerical sequence."""
        if not sequence:
            return 0.0

        # Discretize sequence for entropy calculation
        if len(sequence) == 1:
            return 0.0

        # Use quantiles for discretization
        sorted_seq = sorted(sequence)
        n_bins = min(10, len(sequence))

        # Create bins
        bins = []
        for i in range(n_bins):
            idx = int(i * len(sorted_seq) / n_bins)
            bins.append(sorted_seq[idx])

        # Count occurrences in bins
        bin_counts = [0] * n_bins
        for value in sequence:
            bin_idx = min(n_bins - 1, sum(1 for b in bins if value >= b) - 1)
            bin_counts[bin_idx] += 1

        # Calculate entropy
        total = len(sequence)
        entropy = 0.0
        for count in bin_counts:
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy

    def _generate_complexity_optimizations(self, complexity_ratio: float) -> List[str]:
        """Generate optimization recommendations based on complexity analysis."""
        recommendations = []

        if complexity_ratio > 0.8:
            recommendations.extend(
                [
                    "High complexity detected - consider workload regularization",
                    "Implement pattern-based batching",
                    "Use entropy-guided scheduling",
                ]
            )
        elif complexity_ratio > 0.5:
            recommendations.extend(
                [
                    "Medium complexity - optimize resource allocation",
                    "Consider temporal batching strategies",
                ]
            )
        else:
            recommendations.extend(
                ["Low complexity - system is well-optimized", "Focus on throughput maximization"]
            )

        return recommendations


# Global orchestrator instance
_info_theory_orchestrator: Optional[InformationTheoryOrchestrator] = None


def get_information_theory_orchestrator() -> InformationTheoryOrchestrator:
    """Get the global information theory orchestrator instance."""
    global _info_theory_orchestrator
    if _info_theory_orchestrator is None:
        _info_theory_orchestrator = InformationTheoryOrchestrator()
    return _info_theory_orchestrator


# Convenience functions
def entropy_based_scheduling(jobs: List[Job]) -> List[Job]:
    """Apply entropy-based scheduling to job list."""
    return get_information_theory_orchestrator().entropy_based_scheduling(jobs)


def information_theoretic_batching(jobs: List[Job], max_batch_size: int = 32) -> List[List[Job]]:
    """Create information-optimal batches."""
    return get_information_theory_orchestrator().information_theoretic_batching(
        jobs, max_batch_size
    )
