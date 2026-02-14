"""Tests for the dashboard application."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root_redirects_to_overview(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/overview"


def test_overview_page(client):
    r = client.get("/overview")
    assert r.status_code == 200
    assert "Cluster Overview" in r.text


def test_jobs_page(client):
    r = client.get("/jobs")
    assert r.status_code == 200
    assert "Job Queue" in r.text


def test_metrics_page(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "Metrics" in r.text


def test_health_page(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "Component Health" in r.text


def test_jobs_api(client):
    r = client.get("/jobs/api")
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data
    assert "total" in data


def test_overview_api(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.json()
    assert "preview" in data
    assert "scheduler" in data
    # Verify new observer response shape
    p = data["preview"]
    assert "efficiency_pct" in p
    assert "gpu_nodes" in p
    assert "throughput_jobs_per_hour" in p


def test_health_api(client):
    r = client.get("/health/api")
    assert r.status_code == 200
    data = r.json()
    assert "checks" in data
    assert "all_healthy" in data


def test_submit_job(client):
    r = client.post("/jobs/submit", data={
        "job_name": "test-job",
        "job_type": "training",
        "gpu_type": "A100-80GB",
        "gpu_count": 1,
        "priority": 5,
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/jobs"

    # Verify job appears in list
    r = client.get("/jobs/api")
    jobs = r.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["name"] == "test-job"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ready", "degraded")
    assert "checks" in data
