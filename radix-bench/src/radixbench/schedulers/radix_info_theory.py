from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from .base import Scheduler, Assignment as BaseAssignment
except Exception:  # pragma: no cover
    class Scheduler:
        def __init__(self) -> None:
            self.submitted_jobs = []
    
    class BaseAssignment:
        def __init__(self, job, gpu, start_time=None):
            self.job = job
            self.gpu = gpu
            self.start_time = start_time

# Try to import JobStatus enum if present, but don't depend on it.
try:
    from radixbench.core.types import JobStatus  # path may differ in your tree
except Exception:
    JobStatus = None

def _env_flag(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0","false","False","no","NO")

def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass
class _SimpleAssignment:
    job: object
    gpu: object

def _make_assignment_factory():
    try:
        return lambda j,g,t: BaseAssignment(job=j, gpu=g, start_time=t)
    except Exception:
        return lambda j,g,t: _SimpleAssignment(job=j, gpu=g)

_PRED_MIN_RUNTIME = 1e-6
_PRED_CLIP_LOW = 0.05
_PRED_CLIP_HIGH = 20.0


class RadixInfoTheoryScheduler(Scheduler):
    NAME = "radix"

    def __init__(self, alpha=None, enable_backfill=None, softmax_tau=None, preempt_enable=None, preempt_budget_per_hour=None, preempt_gain_threshold=None, rng_seed=1337, jitter=1e-6):
        super().__init__()
        self._debug = os.environ.get("RADIX_DEBUG", "") not in ("", "0", "false", "False")
        
        # New configuration parameters with environment variable support
        self.alpha = _env_float("RADIX_ALPHA", 0.001) if alpha is None else float(alpha)
        self.enable_backfill = _env_flag("RADIX_ENABLE_BACKFILL", False) if enable_backfill is None else bool(enable_backfill)
        self.softmax_tau = _env_float("RADIX_SOFTMAX_TAU", 0.0) if softmax_tau is None else float(softmax_tau)
        self.preempt_enable = _env_flag("RADIX_PREEMPT_ENABLE", False) if preempt_enable is None else bool(preempt_enable)
        self.preempt_budget_per_hour = int(_env_float("RADIX_PREEMPT_BUDGET_PER_HOUR", 2)) if preempt_budget_per_hour is None else int(preempt_budget_per_hour)
        self.preempt_gain_threshold = _env_float("RADIX_PREEMPT_GAIN_THRESHOLD", 0.20) if preempt_gain_threshold is None else float(preempt_gain_threshold)
        self.jitter = jitter
        # Optional predictive hook (bench CLI may inject via set_predictor)
        self._predictor = None
        random.seed(rng_seed)
        self._assign = _make_assignment_factory()
        self._preempted_in_hour = {}

    def set_predictor(self, predictor: Any) -> None:
        """Attach a predictor with method predict_runtime(features)->seconds.
        Deterministic influence with bounded scaling. If predictor lacks the
        expected method or raises, the scheduler behaves identically.
        """
        self._predictor = predictor

    def submit(self, job: Any) -> None:
        self.submitted_jobs.append(job)
        if self._debug:
            print(f"[RADIX] submit: {getattr(job,'job_id','?')} at t={getattr(job,'submit_time','?')}")

    def _fits(self, job, gpu):
        return getattr(gpu, "is_available", False) and getattr(job, "memory_gb", 0) <= getattr(gpu, "memory_gb", 0)

    def _wait_time(self, job, t):
        st = getattr(job, "submit_time", 0.0) or 0.0
        return max(0.0, t - st)

    def _base_score(self, job, gpu, t):
        s = getattr(job, "it_score", None)
        if s is None:
            rt = getattr(job, "runtime_estimate", 1.0) or 1.0
            s = 1.0 / max(1e-6, rt)
        return float(s)

    def _score(self, job, gpu, t):
        s = self._base_score(job, gpu, t) + self.alpha * self._wait_time(job, t) + random.uniform(0.0, self.jitter)
        # Optional predictive scaling (bounded, deterministic)
        if getattr(self, "_predictor", None) is not None:
            try:
                # Construct minimal feature dict; do not depend on external helpers
                base_rt = float(getattr(job, "runtime_estimate", 0.0) or 0.0)
                if base_rt < _PRED_MIN_RUNTIME:
                    base_rt = _PRED_MIN_RUNTIME
                feats = {
                    "req_runtime_sec": base_rt,
                    "req_mem_gb": float(getattr(job, "memory_gb", 16.0)),
                    "queue_age_sec": float(t - float(getattr(job, "submit_time", 0.0) or 0.0)),
                    "job_size_hint": float(getattr(job, "size_hint", 1.0)),
                }
                pr_fun = getattr(self._predictor, "predict_runtime", None)
                if callable(pr_fun):
                    pred = float(pr_fun(feats))
                    if pred < _PRED_MIN_RUNTIME:
                        pred = _PRED_MIN_RUNTIME
                    runtime_scale = base_rt / pred
                    if runtime_scale < _PRED_CLIP_LOW:
                        runtime_scale = _PRED_CLIP_LOW
                    if runtime_scale > _PRED_CLIP_HIGH:
                        runtime_scale = _PRED_CLIP_HIGH
                    s *= runtime_scale
            except Exception:
                # Silent fallback on any predictor error
                pass
        return s

    def _argmax_pick(self, candidates, gpu, t):
        return max(candidates, key=lambda j: self._score(j, gpu, t))

    def _softmax_pick(self, candidates, gpu, t, tau):
        vals = [self._score(j, gpu, t) for j in candidates]
        m = max(vals)
        exps = [math.exp((v - m)/max(1e-6, tau)) for v in vals]
        s = sum(exps)
        r = random.random() * s
        c = 0.0
        for j, w in zip(candidates, exps):
            c += w
            if c >= r:
                return j
        return candidates[-1]

    def _hour_bucket(self, t):
        return int(t // 3600.0)

    def _remaining_preempt_budget(self, t):
        hb = self._hour_bucket(t)
        used = self._preempted_in_hour.get(hb, 0)
        return max(0, self.preempt_budget_per_hour - used)

    def _record_preempt(self, t):
        hb = self._hour_bucket(t)
        self._preempted_in_hour[hb] = self._preempted_in_hour.get(hb, 0) + 1

    def _maybe_preempt(self, pending_jobs, gpus, current_time):
        if not self.preempt_enable:
            return []
        supported = any(hasattr(g, "current_job") for g in gpus)
        if not supported or self._remaining_preempt_budget(current_time) <= 0:
            return []
        actions = []
        for g in gpus:
            if getattr(g, "is_available", False):
                continue
            rj = getattr(g, "current_job", None)
            if rj is None:
                continue
            fits = [j for j in pending_jobs if getattr(j, "memory_gb", 0) <= getattr(g, "memory_gb", 0)]
            if not fits:
                continue
            cand = min(fits, key=lambda j: getattr(j, "runtime_estimate", float("inf")))
            rt_running = getattr(rj, "remaining_time", None)
            rt_running = rt_running if rt_running is not None else getattr(rj, "runtime_estimate", float("inf"))
            rt_new = getattr(cand, "runtime_estimate", float("inf"))
            if rt_running <= 0 or rt_new <= 0:
                continue
            gain = 1.0 - (rt_new / max(1e-6, rt_running))
            preemptible = getattr(rj, "preemptible", False)
            if preemptible and gain >= self.preempt_gain_threshold and self._remaining_preempt_budget(current_time) > 0:
                actions.append(("preempt", rj, cand, g))
                self._record_preempt(current_time)
        return actions

    def schedule(self, cluster: List[Any], now: float, **kwargs: Dict[str, Any]) -> List[BaseAssignment]:
        # Get pending jobs from submitted_jobs
        pending = [
            j for j in self.submitted_jobs
            if (float(getattr(j, "submit_time", 0.0)) <= float(now))
            and (getattr(j, "start_time", None) is None)
        ]
        
        # Get idle GPUs
        idle = [g for g in cluster if getattr(g, "is_available", False)]
        
        if not idle or not pending:
            return []
            
        assignments = []
        chosen = set()
        
        # Main assignment loop - one job per GPU
        for gpu in idle:
            cands = [j for j in pending if getattr(j, "job_id", "") not in chosen and self._fits(j, gpu)]
            if not cands:
                continue
                
            # Choose job using either softmax or argmax
            if self.softmax_tau and self.softmax_tau > 0.0:
                pick = self._softmax_pick(cands, gpu, now, self.softmax_tau)
            else:
                pick = self._argmax_pick(cands, gpu, now)
                
            assignments.append(self._assign(pick, gpu, now))
            chosen.add(getattr(pick, "job_id", ""))
            
        # Backfill: if enabled and we have remaining idle GPUs, fill with shortest jobs
        if self.enable_backfill and len(assignments) < len(idle):
            already = {getattr(a.job, "job_id", "") for a in assignments}
            used = {getattr(a.gpu, "gpu_id", "") for a in assignments}
            remaining = [g for g in idle if getattr(g, "gpu_id", "") not in used]
            
            for gpu in remaining:
                fits = [j for j in pending if getattr(j, "job_id", "") not in already and self._fits(j, gpu)]
                if not fits:
                    continue
                j_best = min(fits, key=lambda j: getattr(j, "runtime_estimate", float("inf")))
                assignments.append(self._assign(j_best, gpu, now))
                already.add(getattr(j_best, "job_id", ""))
                
        # Check for preemption opportunities (prepare for future simulator hooks)
        preempt_actions = self._maybe_preempt(pending, cluster, now)
        if preempt_actions:
            # For now, just log preemption opportunities - actual preemption requires simulator support
            if self._debug:
                print(f"[RADIX] preemption opportunities: {len(preempt_actions)}")
                
        return assignments

    def preempt(self, cluster: List[Any], current_time: float) -> List[Any]:
        """
        Public interface for preemption decisions.
        
        Returns a list of preemption actions that the simulator can execute.
        Each action is a tuple of (action_type, victim_job, replacement_job, gpu).
        
        Args:
            cluster: List of GPU objects
            current_time: Current simulation time
            
        Returns:
            List of preemption actions to execute
        """
        if not self.preempt_enable:
            return []
            
        # Get pending jobs from submitted_jobs
        pending = [
            j for j in self.submitted_jobs
            if (float(getattr(j, "submit_time", 0.0)) <= float(current_time))
            and (getattr(j, "start_time", None) is None)
        ]
        
        return self._maybe_preempt(pending, cluster, current_time)

