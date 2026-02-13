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
        return {"gpu_nodes": 0, "pending": 0, "estimated_improvement_now_pct": 0, "error": str(e)}


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


@router.get("/overview", response_class=HTMLResponse)
async def overview_page(request: Request):
    """Main dashboard overview page."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    preview = await _fetch_preview(client, settings.observer_url)
    ts_summary = await _fetch_timeseries_summary(client, settings.observer_url)
    scheduler_info = await _fetch_scheduler_info(client, settings.scheduler_url)

    return templates.TemplateResponse("overview.html", {
        "request": request,
        "settings": settings,
        "page": "overview",
        "preview": preview,
        "timeseries": ts_summary,
        "scheduler": scheduler_info,
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

    return {
        "preview": preview,
        "timeseries": ts_summary,
        "scheduler": scheduler_info,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
