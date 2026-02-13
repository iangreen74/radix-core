"""Radix Core Dashboard — FastAPI + Jinja2."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .routes import overview, jobs, metrics, health

logger = logging.getLogger("radix.dashboard")

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the shared httpx client."""
    settings = get_settings()
    app.state.settings = settings
    app.state.http_client = httpx.AsyncClient(timeout=10.0)
    logging.basicConfig(level=logging.INFO)
    logger.info("Dashboard started — observer=%s scheduler=%s",
                settings.observer_url, settings.scheduler_url)
    yield
    await app.state.http_client.aclose()
    logger.info("Dashboard stopped")


app = FastAPI(
    title="Radix Core Dashboard",
    description="GPU Orchestration Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include route modules
app.include_router(overview.router)
app.include_router(jobs.router)
app.include_router(metrics.router)
app.include_router(health.router)


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/readyz")
async def readyz(request: Request):
    """Readiness check — verifies backend connectivity."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client
    checks = {}
    try:
        r = await client.get(f"{settings.observer_url}/healthz")
        checks["observer"] = r.status_code == 200
    except Exception:
        checks["observer"] = False
    try:
        r = await client.get(f"{settings.scheduler_url}/healthz")
        checks["scheduler"] = r.status_code == 200
    except Exception:
        checks["scheduler"] = False

    all_ok = all(checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}


@app.get("/")
async def root():
    """Redirect root to overview."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/overview", status_code=302)
