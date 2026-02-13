"""Health page — component status, connectivity checks."""

import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

logger = logging.getLogger("radix.dashboard.health")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/health", tags=["health"])


async def _check_service(
    client: httpx.AsyncClient, name: str, base_url: str
) -> dict:
    """Check health and readiness of a backend service."""
    result = {
        "name": name,
        "url": base_url,
        "healthy": False,
        "ready": False,
        "details": {},
        "error": None,
    }
    try:
        r = await client.get(f"{base_url}/healthz")
        result["healthy"] = r.status_code == 200
        result["details"]["healthz"] = r.json() if r.status_code == 200 else r.text
    except Exception as e:
        result["error"] = str(e)
        return result

    try:
        r = await client.get(f"{base_url}/readyz")
        result["ready"] = r.status_code == 200
        result["details"]["readyz"] = r.json() if r.status_code == 200 else r.text
    except Exception:
        # readyz might not exist on all services
        result["ready"] = result["healthy"]

    return result


@router.get("", response_class=HTMLResponse)
async def health_page(request: Request):
    """Component health status page."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    services = [
        ("Observer", settings.observer_url),
        ("Scheduler Agent", settings.scheduler_url),
        ("Dashboard", "http://localhost:8080"),
    ]

    checks = []
    for name, url in services:
        check = await _check_service(client, name, url)
        checks.append(check)

    all_healthy = all(c["healthy"] for c in checks)
    all_ready = all(c["ready"] for c in checks)

    return templates.TemplateResponse("health.html", {
        "request": request,
        "settings": settings,
        "page": "health",
        "checks": checks,
        "all_healthy": all_healthy,
        "all_ready": all_ready,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@router.get("/api", tags=["api"])
async def health_api(request: Request):
    """JSON API for health data."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    services = [
        ("Observer", settings.observer_url),
        ("Scheduler Agent", settings.scheduler_url),
    ]

    checks = []
    for name, url in services:
        check = await _check_service(client, name, url)
        checks.append(check)

    return {"checks": checks, "all_healthy": all(c["healthy"] for c in checks)}
