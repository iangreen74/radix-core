"""Statistical analysis utilities for benchmarking."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    data: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    statistic_func=np.mean
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Sample data
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        statistic_func: Function to compute statistic (default: mean)

    Returns:
        Tuple of (statistic, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0

    data_array = np.array(data)
    original_stat = statistic_func(data_array)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return original_stat, lower_bound, upper_bound


def cohens_d_effect_size(group1: list[float], group2: list[float]) -> float:
    """
    Cohen's d with guards for tiny samples and zero variance.
    If either group has <2 samples, or the pooled variance is non-positive,
    return 0.0 (undefined / no effect at this granularity).
    """
    import numpy as np
    arr1 = np.asarray(group1, dtype=float)
    arr2 = np.asarray(group2, dtype=float)
    n1 = int(arr1.size)
    n2 = int(arr2.size)

    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = float(np.mean(arr1))
    mean2 = float(np.mean(arr2))
    var1 = float(np.var(arr1, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(arr2, ddof=1)) if n2 > 1 else 0.0

    denom = (n1 + n2 - 2)
    if denom <= 0:
        return 0.0
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / denom
    if pooled_var <= 0.0:
        return 0.0
    pooled_std = float(np.sqrt(pooled_var))
    if pooled_std == 0.0:
        return 0.0
    return (mean1 - mean2) / pooled_std

def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """
    Perform Welch's t-test for unequal variances.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if not group1 or not group2:
        return 0.0, 1.0

    arr1 = np.array(group1)
    arr2 = np.array(group2)

    # Use scipy's implementation
    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)

    return t_stat, p_value


def calculate_win_rate(baseline_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
    """
    Calculate win rate statistics between baseline and treatment.

    Args:
        baseline_data: Baseline performance data
        treatment_data: Treatment performance data

    Returns:
        Dictionary with win/loss/tie rates and confidence intervals
    """
    if not baseline_data or not treatment_data:
        return {
            "win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0,
            "win_rate_ci_lower": 0.0, "win_rate_ci_upper": 0.0
        }

    # Pairwise comparisons (assuming paired data or cross-product)
    wins = 0
    losses = 0
    ties = 0
    total_comparisons = 0

    for baseline_val in baseline_data:
        for treatment_val in treatment_data:
            total_comparisons += 1
            if treatment_val > baseline_val:
                wins += 1
            elif treatment_val < baseline_val:
                losses += 1
            else:
                ties += 1

    if total_comparisons == 0:
        return {
            "win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0,
            "win_rate_ci_lower": 0.0, "win_rate_ci_upper": 0.0
        }

    win_rate = wins / total_comparisons
    loss_rate = losses / total_comparisons
    tie_rate = ties / total_comparisons

    # Bootstrap confidence interval for win rate
    win_indicators = [1 if treatment_val > baseline_val else 0
                     for baseline_val in baseline_data
                     for treatment_val in treatment_data]

    _, win_rate_ci_lower, win_rate_ci_upper = bootstrap_confidence_interval(
        win_indicators, confidence_level=0.95
    )

    return {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "win_rate_ci_lower": win_rate_ci_lower,
        "win_rate_ci_upper": win_rate_ci_upper
    }


def multi_seed_analysis(
    results_by_seed: Dict[int, Dict[str, List[float]]],
    baseline_scheduler: str = "fifo"
) -> Dict[str, Dict]:
    """
    Perform comprehensive statistical analysis across multiple seeds.

    Args:
        results_by_seed: Dictionary mapping seed -> scheduler -> metric_values
        baseline_scheduler: Name of baseline scheduler for comparisons

    Returns:
        Dictionary with statistical analysis results
    """
    analysis = {}

    # Aggregate data across seeds
    aggregated_data = {}
    for seed, seed_results in results_by_seed.items():
        for scheduler, metrics in seed_results.items():
            if scheduler not in aggregated_data:
                aggregated_data[scheduler] = []
            aggregated_data[scheduler].extend(metrics)

    # Get baseline data
    baseline_data = aggregated_data.get(baseline_scheduler, [])

    for scheduler, data in aggregated_data.items():
        scheduler_analysis = {}

        # Basic statistics with confidence intervals
        mean_val, ci_lower, ci_upper = bootstrap_confidence_interval(data)
        scheduler_analysis["mean"] = mean_val
        scheduler_analysis["ci_lower"] = ci_lower
        scheduler_analysis["ci_upper"] = ci_upper
        scheduler_analysis["std"] = np.std(data) if data else 0.0
        scheduler_analysis["n_samples"] = len(data)

        # Comparison with baseline
        if scheduler != baseline_scheduler and baseline_data:
            # Effect size
            effect_size = cohens_d_effect_size(data, baseline_data)
            scheduler_analysis["cohens_d"] = effect_size
            scheduler_analysis["effect_magnitude"] = interpret_effect_size(effect_size)

            # Statistical significance
            t_stat, p_value = welch_t_test(data, baseline_data)
            scheduler_analysis["t_statistic"] = t_stat
            scheduler_analysis["p_value"] = p_value
            scheduler_analysis["significant"] = p_value < 0.05

            # Win rate analysis
            win_rate_stats = calculate_win_rate(baseline_data, data)
            scheduler_analysis.update(win_rate_stats)

            # Improvement percentage
            baseline_mean = np.mean(baseline_data) if baseline_data else 0.0
            if baseline_mean != 0:
                improvement = ((mean_val - baseline_mean) / baseline_mean) * 100
                scheduler_analysis["improvement_percent"] = improvement

        analysis[scheduler] = scheduler_analysis

    return analysis


def generate_league_table(analysis_results: Dict[str, Dict]) -> List[Dict]:
    """
    Generate a league table ranking schedulers by performance.

    Args:
        analysis_results: Results from multi_seed_analysis

    Returns:
        List of scheduler rankings with statistics
    """
    league_entries = []

    for scheduler, stats in analysis_results.items():
        entry = {
            "scheduler": scheduler,
            "mean_performance": stats.get("mean", 0.0),
            "ci_lower": stats.get("ci_lower", 0.0),
            "ci_upper": stats.get("ci_upper", 0.0),
            "improvement_percent": stats.get("improvement_percent", 0.0),
            "effect_size": stats.get("cohens_d", 0.0),
            "effect_magnitude": stats.get("effect_magnitude", "unknown"),
            "win_rate": stats.get("win_rate", 0.0),
            "significant": stats.get("significant", False),
            "n_samples": stats.get("n_samples", 0)
        }
        league_entries.append(entry)

    # Sort by mean performance (descending for metrics where higher is better)
    league_entries.sort(key=lambda x: x["mean_performance"], reverse=True)

    # Add rankings
    for i, entry in enumerate(league_entries):
        entry["rank"] = i + 1

    return league_entries
