"""Overview page — cluster summary, GPU utilization, cost tracking."""

import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

logger = logging.getLogger("radix.dashboard.overview")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(tags=["overview"])


async def _fetch_preview(client: httpx.AsyncClient, observer_url: str) -> dict:
    """Fetch cluster preview from observer."""
    try:
        r = await client.get(f"{observer_url}/v1/preview")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to fetch preview: %s", e)
        return {
            "gpu_nodes": 0, "pending": 0, "running": 0,
            "completed_last_hour": 0,
            "avg_wait_time_seconds": 0, "avg_completion_time_seconds": 0,
            "baseline_completion_time_seconds": 0,
            "efficiency_pct": 0, "throughput_jobs_per_hour": 0,
            "gpu_utilization_pct": 0, "error": str(e),
        }


async def _fetch_timeseries_summary(client: httpx.AsyncClient, observer_url: str) -> dict:
    """Fetch recent timeseries summary for the overview cards."""
    try:
        r = await client.get(f"{observer_url}/v1/timeseries")
        r.raise_for_status()
        data = r.json().get("data", [])
        return {
            "total_points": len(data),
            "latest": data[-1] if data else None,
        }
    except Exception as e:
        logger.warning("Failed to fetch timeseries: %s", e)
        return {"total_points": 0, "latest": None, "error": str(e)}


async def _fetch_scheduler_info(client: httpx.AsyncClient, scheduler_url: str) -> dict:
    """Fetch scheduler service info."""
    try:
        r = await client.get(f"{scheduler_url}/")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to fetch scheduler info: %s", e)
        return {"service": "unavailable", "error": str(e)}


def _compute_system_status(preview: dict, scheduler_info: dict) -> str:
    """Determine overall system status from health data.

    Returns 'operational', 'degraded', or 'down'.
    """
    has_observer_error = "error" in preview
    has_scheduler_error = "error" in scheduler_info
    scheduler_unavailable = scheduler_info.get("service") == "unavailable"

    if has_observer_error and (has_scheduler_error or scheduler_unavailable):
        return "down"
    if has_observer_error or has_scheduler_error or scheduler_unavailable:
        return "degraded"
    return "operational"


def _compute_radix_score(
    preview: dict,
    profile: str = "balanced",
) -> int:
    """Compute composite Radix Score (0-100) from cluster metrics.

    Profiles adjust the weight distribution:
      - balanced:        GPU 40%, memory 30%, throughput 30%
      - performance:     GPU 30%, memory 20%, throughput 50%
      - cost-optimized:  GPU 50%, memory 40%, throughput 10%
    """
    weights = {
        "balanced":       (0.40, 0.30, 0.30),
        "performance":    (0.30, 0.20, 0.50),
        "cost-optimized": (0.50, 0.40, 0.10),
    }
    w_gpu, w_mem, w_tp = weights.get(profile, weights["balanced"])

    gpu_util = float(preview.get("gpu_utilization_pct", 0))
    efficiency = float(preview.get("efficiency_pct", 0))
    throughput_raw = float(preview.get("throughput_jobs_per_hour", 0))

    # Normalize throughput to 0-100 (cap at 100 jobs/hr)
    throughput_score = min(throughput_raw, 100.0)

    score = gpu_util * w_gpu + efficiency * w_mem + throughput_score * w_tp
    return max(0, min(100, int(round(score))))


@router.get("/overview", response_class=HTMLResponse)
async def overview_page(request: Request):
    """Main dashboard overview page."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    preview = await _fetch_preview(client, settings.observer_url)
    ts_summary = await _fetch_timeseries_summary(client, settings.observer_url)
    scheduler_info = await _fetch_scheduler_info(client, settings.scheduler_url)

    system_status = _compute_system_status(preview, scheduler_info)
    queue_depth = preview.get("pending", 0)
    radix_score = _compute_radix_score(preview)

    return templates.TemplateResponse("overview.html", {
        "request": request,
        "settings": settings,
        "page": "overview",
        "preview": preview,
        "timeseries": ts_summary,
        "scheduler": scheduler_info,
        "system_status": system_status,
        "queue_depth": queue_depth,
        "radix_score": radix_score,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@router.get("/api/overview", tags=["api"])
async def overview_api(request: Request):
    """JSON API for overview data (used by auto-refresh)."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    preview = await _fetch_preview(client, settings.observer_url)
    ts_summary = await _fetch_timeseries_summary(client, settings.observer_url)
    scheduler_info = await _fetch_scheduler_info(client, settings.scheduler_url)

    system_status = _compute_system_status(preview, scheduler_info)
    queue_depth = preview.get("pending", 0)
    radix_score = _compute_radix_score(preview)

    return {
        "preview": preview,
        "timeseries": ts_summary,
        "scheduler": scheduler_info,
        "system_status": system_status,
        "queue_depth": queue_depth,
        "radix_score": radix_score,
        "radix_score_profiles": {
            "balanced": _compute_radix_score(preview, "balanced"),
            "performance": _compute_radix_score(preview, "performance"),
            "cost-optimized": _compute_radix_score(preview, "cost-optimized"),
        },
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
