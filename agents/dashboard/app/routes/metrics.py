"""Metrics page — timeseries charts, scheduler performance."""

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

logger = logging.getLogger("radix.dashboard.metrics")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """Metrics and timeseries page."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    # Fetch timeseries data
    timeseries_data = []
    try:
        r = await client.get(f"{settings.observer_url}/v1/timeseries")
        r.raise_for_status()
        timeseries_data = r.json().get("data", [])
    except Exception as e:
        logger.warning("Failed to fetch timeseries: %s", e)

    # Fetch scheduler prometheus metrics (raw text)
    scheduler_metrics_raw = ""
    scheduler_metrics = {}
    try:
        r = await client.get(f"{settings.scheduler_url}/metrics")
        r.raise_for_status()
        scheduler_metrics_raw = r.text
        # Parse some key metrics from prometheus format
        for line in scheduler_metrics_raw.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split(" ")
                if len(parts) == 2:
                    scheduler_metrics[parts[0]] = parts[1]
    except Exception as e:
        logger.warning("Failed to fetch scheduler metrics: %s", e)

    return templates.TemplateResponse("metrics.html", {
        "request": request,
        "settings": settings,
        "page": "metrics",
        "timeseries_data": timeseries_data,
        "scheduler_metrics": scheduler_metrics,
        "total_points": len(timeseries_data),
    })


@router.get("/api/timeseries", tags=["api"])
async def timeseries_api(request: Request):
    """JSON API for timeseries data (used by Chart.js)."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        r = await client.get(f"{settings.observer_url}/v1/timeseries")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"data": [], "error": str(e)}
