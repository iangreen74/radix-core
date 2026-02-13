"""Radix Softmax Scheduler - Softmax variant of the Radix information-theoretic scheduler."""

from __future__ import annotations

import os
import math
import random
from typing import Any, Dict, List

from .radix_info_theory import RadixInfoTheoryScheduler


class RadixSoftmaxScheduler(RadixInfoTheoryScheduler):
    """
    Softmax variant of the Radix information-theoretic scheduler.
    
    This scheduler extends RadixInfoTheoryScheduler with default softmax sampling
    enabled, making it easier to use stochastic job selection without needing
    to set environment variables.
    """
    
    NAME = "radix_softmax"
    
    def __init__(self, alpha=None, enable_backfill=None, softmax_tau=None, 
                 preempt_enable=None, preempt_budget_per_hour=None, 
                 preempt_gain_threshold=None, rng_seed=1337, jitter=1e-6):
        """
        Initialize RadixSoftmaxScheduler with softmax enabled by default.
        
        Args:
            alpha: Aging weight (default: 0.001)
            enable_backfill: Enable backfill scheduling (default: True)
            softmax_tau: Softmax temperature (default: 0.5)
            preempt_enable: Enable preemption (default: False)
            preempt_budget_per_hour: Preemption budget per hour (default: 2)
            preempt_gain_threshold: Minimum gain threshold for preemption (default: 0.20)
            rng_seed: Random seed (default: 1337)
            jitter: Small random jitter for tie-breaking (default: 1e-6)
        """
        # Set softmax-friendly defaults
        if alpha is None:
            alpha = float(os.getenv("RADIX_ALPHA", "0.001"))
        if enable_backfill is None:
            enable_backfill = os.getenv("RADIX_ENABLE_BACKFILL", "1") not in ("0", "false", "False")
        if softmax_tau is None:
            softmax_tau = float(os.getenv("RADIX_SOFTMAX_TAU", "0.5"))  # Default enabled
        if preempt_enable is None:
            preempt_enable = os.getenv("RADIX_PREEMPT_ENABLE", "0") not in ("0", "false", "False")
        if preempt_budget_per_hour is None:
            preempt_budget_per_hour = int(float(os.getenv("RADIX_PREEMPT_BUDGET_PER_HOUR", "2")))
        if preempt_gain_threshold is None:
            preempt_gain_threshold = float(os.getenv("RADIX_PREEMPT_GAIN_THRESHOLD", "0.20"))
            
        super().__init__(
            alpha=alpha,
            enable_backfill=enable_backfill,
            softmax_tau=softmax_tau,
            preempt_enable=preempt_enable,
            preempt_budget_per_hour=preempt_budget_per_hour,
            preempt_gain_threshold=preempt_gain_threshold,
            rng_seed=rng_seed,
            jitter=jitter
        )
        
        if self._debug:
            print(f"[RADIX_SOFTMAX] Initialized with tau={self.softmax_tau}, backfill={self.enable_backfill}")
