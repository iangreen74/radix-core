"""
Radix scheduler policy implementation with entropy-weighted scoring, aging, and softmax sampling.
Combines predictive model outputs with information-theoretic principles.
"""

import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """
    Configuration for Radix scheduler policy.
    
    Defaults selected via policy sweep on 2025-10-19:
    - temperature: 0.0 (greedy selection for highest throughput: 0.0428)
    - aging_alpha: 0.0005 (optimal balance of fairness vs throughput)
    - entropy_weight: 0.1 (diversity bonus for model families)
    """
    temperature: float = 0.0      # Softmax temperature (0.0 = greedy, >0 = stochastic)
    aging_alpha: float = 0.0005   # Aging weight factor (data-driven default)
    entropy_weight: float = 0.1   # Weight for entropy regularization term


def score_jobs(queue_df: pd.DataFrame, 
               preds: pd.DataFrame, 
               now_s: float, 
               cfg: PolicyConfig) -> pd.Series:
    """
    Score jobs using entropy-weighted policy with aging and cost awareness.
    
    Formula:
    score = -normalized_cost + aging_bonus + entropy_regularizer
    
    Where:
    - normalized_cost = (runtime_pred + energy_pred/1000) / robust_scale
    - aging_bonus = aging_alpha * wait_time_seconds
    - entropy_regularizer = entropy_weight * model_family_diversity_bonus
    
    Args:
        queue_df: DataFrame with job information including submit_time
        preds: DataFrame with runtime_ms_pred and energy_j_pred columns
        now_s: Current time in seconds
        cfg: Policy configuration
        
    Returns:
        Series with job scores (higher = better priority)
    """
    if queue_df.empty or preds.empty:
        return pd.Series(dtype=float, index=queue_df.index)
    
    # Ensure indices align
    common_idx = queue_df.index.intersection(preds.index)
    if len(common_idx) == 0:
        logger.warning("No common indices between queue_df and preds")
        return pd.Series(dtype=float, index=queue_df.index)
    
    queue_subset = queue_df.loc[common_idx]
    preds_subset = preds.loc[common_idx]
    
    # 1. Cost proxy: combine runtime and energy predictions
    # Convert energy from Joules to equivalent milliseconds (rough approximation)
    runtime_ms = preds_subset['runtime_ms_pred']
    energy_equiv_ms = preds_subset['energy_j_pred'] / 1000.0  # Simple conversion
    
    combined_cost = runtime_ms + energy_equiv_ms
    
    # Robust normalization using median and MAD (median absolute deviation)
    median_cost = combined_cost.median()
    mad_cost = (combined_cost - median_cost).abs().median()
    robust_scale = max(mad_cost, 1.0)  # Avoid division by zero
    
    normalized_cost = (combined_cost - median_cost) / robust_scale
    
    # 2. Aging bonus: prevent starvation
    submit_times = queue_subset.get('submit_time', queue_subset.get('submit_ts', now_s))
    wait_times = now_s - submit_times
    aging_bonus = cfg.aging_alpha * wait_times
    
    # 3. Entropy regularizer: encourage diversity across model families
    entropy_bonus = pd.Series(0.0, index=common_idx)
    
    if 'model_family' in queue_subset.columns:
        model_families = queue_subset['model_family']
        family_counts = model_families.value_counts()
        
        # Bonus inversely proportional to family frequency (encourage rare families)
        total_jobs = len(model_families)
        for idx in common_idx:
            family = model_families.loc[idx]
            family_freq = family_counts[family] / total_jobs
            # Higher bonus for less frequent families
            entropy_bonus.loc[idx] = cfg.entropy_weight * (1.0 - family_freq)
    
    # 4. Combine components
    # Higher score = higher priority
    scores = -normalized_cost + aging_bonus + entropy_bonus
    
    # Extend to full queue index with default scores for missing jobs
    full_scores = pd.Series(0.0, index=queue_df.index)
    full_scores.loc[common_idx] = scores
    
    return full_scores


def softmax_sample(scores: pd.Series, temperature: float, seed: int) -> str:
    """
    Sample job ID using softmax distribution over scores.
    
    Args:
        scores: Series with job scores (index = job_ids)
        temperature: Softmax temperature (0.0 = greedy, higher = more random)
        seed: Random seed for deterministic sampling
        
    Returns:
        Selected job ID
    """
    if scores.empty:
        raise ValueError("Cannot sample from empty scores")
    
    if temperature <= 0.0:
        # Greedy selection: return highest scoring job
        # Use deterministic tie-breaking by job_id string
        max_score = scores.max()
        max_jobs = scores[scores == max_score]
        return sorted(max_jobs.index)[0]  # Lexicographic tie-breaking
    
    # Stochastic softmax sampling
    # Create deterministic RNG from seed and scores
    scores_bytes = str(sorted(scores.index)).encode('utf-8')
    combined_seed = seed ^ int(hashlib.sha1(scores_bytes).hexdigest()[:8], 16)
    rng = np.random.RandomState(combined_seed)
    
    # Apply softmax
    exp_scores = np.exp(scores.values / temperature)
    probabilities = exp_scores / exp_scores.sum()
    
    # Sample according to probabilities
    selected_idx = rng.choice(len(scores), p=probabilities)
    return scores.index[selected_idx]


def evaluate_preemption_gain(current_job: Dict[str, Any], 
                           candidate_job: Dict[str, Any],
                           current_preds: Dict[str, float],
                           candidate_preds: Dict[str, float],
                           threshold: float = 0.03) -> bool:
    """
    Evaluate whether preempting current job for candidate provides sufficient gain.
    
    Args:
        current_job: Currently running job metadata
        candidate_job: Candidate job for preemption
        current_preds: Predictions for current job
        candidate_preds: Predictions for candidate job  
        threshold: Minimum relative improvement required (default 3%)
        
    Returns:
        True if preemption is beneficial
    """
    # Calculate remaining runtime for current job
    current_remaining = current_preds.get('runtime_ms_pred', 300000)  # Default 5 min
    candidate_runtime = candidate_preds.get('runtime_ms_pred', 300000)
    
    # Simple heuristic: preempt if candidate is significantly shorter
    if candidate_runtime < current_remaining * (1 - threshold):
        return True
    
    # Consider energy efficiency
    current_energy_rate = current_preds.get('energy_j_pred', 450000) / current_remaining
    candidate_energy_rate = candidate_preds.get('energy_j_pred', 450000) / candidate_runtime
    
    if candidate_energy_rate < current_energy_rate * (1 - threshold):
        return True
    
    return False


class PreemptionGate:
    """Lightweight preemption gate with budget control."""
    
    def __init__(self, budget_per_hour: int = 2, gain_threshold: float = 0.03):
        self.budget_per_hour = budget_per_hour
        self.gain_threshold = gain_threshold
        self.preemption_log = []  # Track preemptions for budget
        
    def should_preempt(self, 
                      current_job: Dict[str, Any],
                      candidate_job: Dict[str, Any], 
                      current_preds: Dict[str, float],
                      candidate_preds: Dict[str, float],
                      current_time: float) -> bool:
        """
        Determine if preemption should occur based on gain and budget.
        
        Args:
            current_job: Currently running job
            candidate_job: Candidate for preemption
            current_preds: Predictions for current job
            candidate_preds: Predictions for candidate job
            current_time: Current simulation time
            
        Returns:
            True if preemption should proceed
        """
        # Check budget: count preemptions in last hour
        hour_ago = current_time - 3600
        recent_preemptions = [t for t in self.preemption_log if t > hour_ago]
        
        if len(recent_preemptions) >= self.budget_per_hour:
            return False
        
        # Check gain threshold
        should_preempt = evaluate_preemption_gain(
            current_job, candidate_job,
            current_preds, candidate_preds,
            self.gain_threshold
        )
        
        if should_preempt:
            self.preemption_log.append(current_time)
            # Keep log bounded
            if len(self.preemption_log) > 100:
                self.preemption_log = self.preemption_log[-50:]
        
        return should_preempt
