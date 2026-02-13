"""Jobs page — job queue, scheduling status, submission."""

import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

logger = logging.getLogger("radix.dashboard.jobs")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory job store (in production, this would be backed by a database)
_jobs: list[dict] = []


@router.get("", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Job listing page."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    # Fetch scheduler metrics if available
    scheduler_metrics = {}
    try:
        r = await client.get(f"{settings.scheduler_url}/")
        r.raise_for_status()
        scheduler_metrics = r.json()
    except Exception as e:
        logger.warning("Failed to fetch scheduler metrics: %s", e)

    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "settings": settings,
        "page": "jobs",
        "jobs": _jobs,
        "scheduler_metrics": scheduler_metrics,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@router.post("/submit")
async def submit_job(
    request: Request,
    job_name: str = Form(...),
    job_type: str = Form("training"),
    gpu_type: str = Form("A100-80GB"),
    gpu_count: int = Form(1),
    priority: int = Form(5),
):
    """Submit a new job via the dashboard."""
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    job = {
        "name": job_name,
        "type": job_type,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "priority": priority,
        "status": "pending",
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "score": None,
    }

    # Try to score via scheduler
    try:
        r = await client.post(f"{settings.scheduler_url}/v1/score", json={
            "job_type": job_type,
            "features": {"gpu_mem_gb": 80 if "80" in gpu_type else 40},
            "candidate_gpu_types": [gpu_type],
        })
        if r.status_code == 200:
            score_data = r.json()
            job["score"] = score_data.get("score")
            job["status"] = "scored"
    except Exception as e:
        logger.warning("Failed to score job: %s", e)

    _jobs.insert(0, job)
    return RedirectResponse(url="/jobs", status_code=303)


@router.get("/api", tags=["api"])
async def jobs_api(request: Request):
    """JSON API for job data."""
    return {"jobs": _jobs, "total": len(_jobs)}
