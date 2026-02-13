"""FastAPI application for information-theoretic GPU scheduler."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scheduler_agent_lib'))

from config import get_config
from model import SchedulerModel
from scoring import ScoringService
from schemas import ScoreRequest, ScoreResponse, ObserveRequest


# Prometheus metrics
DECISIONS_COUNTER = Counter(
    "scheduler_decisions_total",
    "Total number of scheduling decisions made",
    ["gpu_type", "decision_type"]
)

EXPLORATION_RATIO = Gauge(
    "scheduler_exploration_ratio",
    "Current exploration ratio"
)

AVG_UNCERTAINTY = Gauge(
    "scheduler_avg_uncertainty",
    "Average uncertainty across all job-GPU pairs"
)

REGRET_ESTIMATE = Gauge(
    "scheduler_regret_estimate",
    "Estimated regret from suboptimal decisions"
)

OBSERVE_COUNTER = Counter(
    "scheduler_observe_events_total",
    "Total number of runtime observations recorded",
    ["job_type", "gpu_type"]
)

SCORING_LATENCY = Histogram(
    "scheduler_scoring_duration_seconds",
    "Time spent scoring jobs"
)


# Global state
model: SchedulerModel = None
scoring_service: ScoringService = None
checkpoint_task: asyncio.Task = None


async def periodic_checkpoint():
    """Periodically checkpoint the model state."""
    config = get_config()
    while True:
        try:
            await asyncio.sleep(config.checkpoint_interval)
            if model:
                model.checkpoint()
                logging.info("Model state checkpointed")
        except Exception as e:
            logging.error(f"Error during checkpoint: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global model, scoring_service, checkpoint_task

    # Startup
    config = get_config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))

    # Initialize model and service
    model = SchedulerModel(config, config.sqlite_path)
    scoring_service = ScoringService(model)

    # Start background tasks
    checkpoint_task = asyncio.create_task(periodic_checkpoint())

    logging.info("Scheduler agent started")

    yield

    # Shutdown
    if checkpoint_task:
        checkpoint_task.cancel()
        try:
            await checkpoint_task
        except asyncio.CancelledError:
            pass

    if model:
        model.checkpoint()

    logging.info("Scheduler agent stopped")


# FastAPI app
app = FastAPI(
    title="Information-Theoretic GPU Scheduler",
    description="GPU scheduler using information theory for exploration vs exploitation",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/v1/score", response_model=ScoreResponse)
async def score_job(request: ScoreRequest):
    """Score a job for GPU assignment using information-theoretic objective."""
    if not scoring_service:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    start_time = time.time()

    try:
        # Score the job
        response = scoring_service.score_job(request)

        # Record metrics
        DECISIONS_COUNTER.labels(
            gpu_type=response.terms.get("chosen_gpu", "unknown"),
            decision_type="exploration" if response.terms.get("sigma", 0) > 1.0 else "exploitation"
        ).inc()

        SCORING_LATENCY.observe(time.time() - start_time)

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error scoring job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/observe")
async def observe_runtime(request: ObserveRequest):
    """Record an observed runtime for model learning."""
    if not scoring_service:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        scoring_service.observe_runtime(request)

        # Record metrics
        OBSERVE_COUNTER.labels(
            job_type=request.job_type,
            gpu_type=request.gpu_type
        ).inc()

        return {"status": "ok"}

    except Exception as e:
        logging.error(f"Error recording observation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update model metrics
        if model:
            metrics = model.get_metrics()
            EXPLORATION_RATIO.set(metrics.get("exploration_ratio", 0.0))
            AVG_UNCERTAINTY.set(metrics.get("avg_uncertainty", 0.0))

            # Simple regret estimate based on exploration ratio
            regret = max(0.0, metrics.get("exploration_ratio", 0.0) - 0.1)
            REGRET_ESTIMATE.set(regret)

        return generate_latest()

    except Exception as e:
        logging.error(f"Error generating metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics generation failed")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if not model or not scoring_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}


@app.get("/readyz")
async def readiness_check():
    """Readiness check endpoint."""
    if not model or not scoring_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Verify model is functional
    try:
        # Simple test score
        test_features = {"gpu_mem_gb": 40}
        test_request = ScoreRequest(
            job_type="test",
            features=test_features,
            candidate_gpu_types=["A100-80GB"]
        )
        scoring_service.score_job(test_request)
        return {"status": "ready"}

    except Exception as e:
        logging.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    config = get_config()
    return {
        "service": "information-theoretic-gpu-scheduler",
        "version": "1.0.0",
        "config": {
            "lambda_uncertainty": config.lambda_uncertainty,
            "beta_exploration": config.beta_exploration,
            "gamma_interference": config.gamma_interference,
            "exploration_cap": config.exploration_cap,
            "enable_interference": config.enable_interference,
            "enable_sinkhorn": config.enable_sinkhorn
        }
    }


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.host, port=config.port)
