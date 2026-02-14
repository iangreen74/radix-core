"""Dashboard gap analyzer — compares current dashboard state against production targets.

Inspects the actual files on disk to determine which features are present,
partial, or missing, then returns a prioritized list of gaps.
"""

from pathlib import Path
from typing import Dict, List


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml (the monorepo root)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def _file_contains(path: Path, *needles: str, case_sensitive: bool = False) -> bool:
    """Return True if the file exists and contains ANY of the needle strings."""
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    if not case_sensitive:
        text = text.lower()
        needles = tuple(n.lower() for n in needles)
    return any(needle in text for needle in needles)


def _file_has_field(path: Path, field_name: str) -> bool:
    """Return True if a Python file defines a field (e.g. `field_name:` or `field_name =`)."""
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    return f"{field_name}:" in text or f"{field_name} :" in text or f"{field_name} =" in text


def analyze_gaps() -> List[Dict]:
    """Analyze the dashboard codebase and return a list of feature gaps.

    Each gap dict contains:
        priority    - P0 / P1 / P2 / P3
        feature     - short feature name
        status      - "missing" or "partial"
        complexity  - "trivial", "low", "medium", "high"
        files       - list of file paths (relative to project root) that need changes
        description - human-readable explanation of what needs to be done
    """
    root = _find_project_root()
    dash = root / "agents" / "dashboard"
    app = dash / "app"
    config_py = app / "config.py"
    base_html = app / "templates" / "base.html"
    overview_html = app / "templates" / "overview.html"
    metrics_html = app / "templates" / "metrics.html"
    dockerfile = dash / "Dockerfile"
    routes_dir = app / "routes"
    main_py = app / "main.py"

    gaps: List[Dict] = []

    # ── P0: trivial ──────────────────────────────────────────────────────

    # 1. 5-second refresh interval
    config_text = config_py.read_text(errors="replace") if config_py.is_file() else ""
    refresh_is_5 = "refresh_interval_seconds: int = 5" in config_text or \
                   "refresh_interval_seconds:int = 5" in config_text or \
                   "refresh_interval_seconds = 5" in config_text
    if not refresh_is_5:
        gaps.append({
            "priority": "P0",
            "feature": "5s refresh",
            "status": "partial",
            "complexity": "trivial",
            "files": [str(config_py.relative_to(root))],
            "description": (
                "Change refresh_interval_seconds default from 30 to 5 in config.py. "
                "Production dashboards need near-real-time updates for GPU cluster monitoring."
            ),
        })

    # 2. Build metadata fields in config
    build_fields = ["build_commit", "build_branch", "build_timestamp"]
    missing_build_fields = [f for f in build_fields if not _file_has_field(config_py, f)]
    if missing_build_fields:
        gaps.append({
            "priority": "P0",
            "feature": "Build metadata",
            "status": "missing" if len(missing_build_fields) == len(build_fields) else "partial",
            "complexity": "trivial",
            "files": [str(config_py.relative_to(root))],
            "description": (
                f"Add build metadata fields to Settings: {', '.join(missing_build_fields)}. "
                "These are injected at build time via Docker ARGs so operators can identify "
                "the exact deployed version."
            ),
        })

    # 3. Build info shown in footer
    if not _file_contains(base_html, "build_commit", "build_branch", "build_timestamp"):
        gaps.append({
            "priority": "P0",
            "feature": "Build info in footer",
            "status": "missing",
            "complexity": "trivial",
            "files": [str(base_html.relative_to(root))],
            "description": (
                "Display build commit, branch, and timestamp in the sidebar footer "
                "so operators can verify which version is deployed at a glance."
            ),
        })

    # 4. Dockerfile BUILD_COMMIT ARG
    if not _file_contains(dockerfile, "BUILD_COMMIT", case_sensitive=True):
        gaps.append({
            "priority": "P0",
            "feature": "Build Dockerfile args",
            "status": "missing",
            "complexity": "trivial",
            "files": [str(dockerfile.relative_to(root))],
            "description": (
                "Add ARG BUILD_COMMIT, BUILD_BRANCH, BUILD_TIMESTAMP to the Dockerfile "
                "and pass them as ENV vars so the running container exposes build metadata."
            ),
        })

    # ── P1: medium ───────────────────────────────────────────────────────

    # 5. Cluster name in config
    if not _file_has_field(config_py, "cluster_name"):
        gaps.append({
            "priority": "P1",
            "feature": "Cluster name",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(config_py.relative_to(root)),
                str(base_html.relative_to(root)),
            ],
            "description": (
                "Add a cluster_name field to Settings and display it in the sidebar header "
                "so multi-cluster operators can distinguish which cluster they are viewing."
            ),
        })

    # 6. Status banner
    if not _file_contains(base_html, "status-banner"):
        gaps.append({
            "priority": "P1",
            "feature": "Status banner",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(base_html.relative_to(root)),
                str((app / "static" / "css" / "style.css").relative_to(root)),
            ],
            "description": (
                "Add a dismissible status banner at the top of the page (above the content area) "
                "for system-wide alerts such as degraded mode, license warnings, or maintenance notices."
            ),
        })

    # 7. Queue depth on overview
    if not _file_contains(overview_html, "Queue Depth", "queue_depth"):
        gaps.append({
            "priority": "P1",
            "feature": "Queue Depth",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(overview_html.relative_to(root)),
                str((routes_dir / "overview.py").relative_to(root)),
            ],
            "description": (
                "Add a Queue Depth card to the overview page showing the number of jobs "
                "waiting in the scheduler queue, broken down by priority tier."
            ),
        })

    # 8. Radix Score
    score_route = routes_dir / "score.py"
    has_score_route = score_route.is_file()
    has_score_in_overview = _file_contains(overview_html, "radix_score", "radix score")
    if not has_score_route and not has_score_in_overview:
        gaps.append({
            "priority": "P1",
            "feature": "Radix Score",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(score_route.relative_to(root)),
                str(overview_html.relative_to(root)),
                str(main_py.relative_to(root)),
            ],
            "description": (
                "Create a Radix Score page (or widget on overview) that displays the "
                "composite scheduling score for the cluster, showing how effectively "
                "Radix is optimizing GPU placement."
            ),
        })
    elif not has_score_route or not has_score_in_overview:
        gaps.append({
            "priority": "P1",
            "feature": "Radix Score",
            "status": "partial",
            "complexity": "medium",
            "files": [
                str(score_route.relative_to(root)),
                str(overview_html.relative_to(root)),
                str(main_py.relative_to(root)),
            ],
            "description": (
                "Radix Score is partially implemented. Ensure both a dedicated route "
                "and an overview widget exist for the composite scheduling score."
            ),
        })

    # 9. Scoring profile dropdown
    has_scoring_profile = any(
        _file_contains(p, "scoring_profile", "scoring-profile", "profile-dropdown",
                       "scoring profile")
        for p in [overview_html, base_html, metrics_html]
        if p.is_file()
    )
    if not has_scoring_profile:
        gaps.append({
            "priority": "P1",
            "feature": "Scoring Profile",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(overview_html.relative_to(root)),
                str(config_py.relative_to(root)),
                str((routes_dir / "overview.py").relative_to(root)),
            ],
            "description": (
                "Add a scoring profile dropdown that lets operators switch between "
                "scheduling profiles (e.g. throughput-optimized, latency-optimized, "
                "balanced) and see the impact on cluster metrics."
            ),
        })

    # ── P2: substantial ──────────────────────────────────────────────────

    # 10. Control Plane Advice
    if not _file_contains(overview_html, "advice", "advisory"):
        gaps.append({
            "priority": "P2",
            "feature": "Control Plane Advice",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(overview_html.relative_to(root)),
                str((routes_dir / "overview.py").relative_to(root)),
            ],
            "description": (
                "Add a Control Plane Advice section to the overview page that surfaces "
                "actionable recommendations from the scheduler (e.g. 'Consider adding "
                "A100 nodes — queue depth exceeds capacity by 3x')."
            ),
        })

    # 11. Health score timeseries chart
    has_health_chart = _file_contains(
        metrics_html,
        "health_score", "health-score", "healthscore", "health score chart",
        "health_chart", "health-chart",
    )
    if not has_health_chart:
        gaps.append({
            "priority": "P2",
            "feature": "Health Score timeseries",
            "status": "missing",
            "complexity": "medium",
            "files": [
                str(metrics_html.relative_to(root)),
                str((routes_dir / "metrics.py").relative_to(root)),
            ],
            "description": (
                "Add a dedicated health score timeseries chart to the metrics page "
                "showing cluster health over time (composite of GPU utilization, "
                "error rates, and queue pressure)."
            ),
        })

    # ── P3: complex ──────────────────────────────────────────────────────

    # 12. Authentication
    auth_indicators = [
        routes_dir / "auth.py",
        routes_dir / "login.py",
        app / "middleware" / "auth.py",
        app / "auth.py",
    ]
    has_auth_file = any(p.is_file() for p in auth_indicators)
    has_auth_in_main = _file_contains(main_py, "auth", "login", "middleware")
    if not has_auth_file and not has_auth_in_main:
        gaps.append({
            "priority": "P3",
            "feature": "Authentication",
            "status": "missing",
            "complexity": "high",
            "files": [
                str((routes_dir / "auth.py").relative_to(root)),
                str(main_py.relative_to(root)),
                str(base_html.relative_to(root)),
            ],
            "description": (
                "Add authentication middleware and a login route. Production deployments "
                "require at minimum SSO/OIDC integration to prevent unauthorized access "
                "to cluster controls and job submission."
            ),
        })
    elif not has_auth_file or not has_auth_in_main:
        gaps.append({
            "priority": "P3",
            "feature": "Authentication",
            "status": "partial",
            "complexity": "high",
            "files": [
                str((routes_dir / "auth.py").relative_to(root)),
                str(main_py.relative_to(root)),
                str(base_html.relative_to(root)),
            ],
            "description": (
                "Authentication is partially implemented. Ensure auth middleware is "
                "registered in main.py and a login/callback route exists."
            ),
        })

    # 13. Persistent jobs (database-backed instead of in-memory)
    jobs_py = routes_dir / "jobs.py"
    if jobs_py.is_file():
        jobs_text = jobs_py.read_text(errors="replace")
        uses_db = any(kw in jobs_text.lower() for kw in [
            "sqlalchemy", "database", "asyncpg", "sqlite", "postgresql",
            "create_engine", "sessionmaker", "tortoise", "prisma",
        ])
        uses_in_memory = "_jobs: list" in jobs_text or "_jobs = [" in jobs_text
        if uses_in_memory and not uses_db:
            gaps.append({
                "priority": "P3",
                "feature": "Persistent jobs",
                "status": "partial",
                "complexity": "high",
                "files": [
                    str(jobs_py.relative_to(root)),
                    str(config_py.relative_to(root)),
                ],
                "description": (
                    "Jobs are stored in an in-memory list and lost on restart. "
                    "Migrate to a persistent store (SQLite for single-node, PostgreSQL "
                    "for production) so job history survives dashboard restarts."
                ),
            })

    # 14. LLM Workflows
    workflows_py = routes_dir / "workflows.py"
    if not workflows_py.is_file():
        gaps.append({
            "priority": "P3",
            "feature": "LLM Workflows",
            "status": "missing",
            "complexity": "high",
            "files": [
                str(workflows_py.relative_to(root)),
                str(main_py.relative_to(root)),
                str((app / "templates" / "workflows.html").relative_to(root)),
            ],
            "description": (
                "Create a workflows page for defining and managing multi-step LLM "
                "workflows (e.g. fine-tune then evaluate, or batch inference pipelines) "
                "with DAG visualization and status tracking."
            ),
        })

    # 15. Diagnostics tab
    diagnostics_py = routes_dir / "diagnostics.py"
    if not diagnostics_py.is_file():
        gaps.append({
            "priority": "P3",
            "feature": "Diagnostics tab",
            "status": "missing",
            "complexity": "high",
            "files": [
                str(diagnostics_py.relative_to(root)),
                str(main_py.relative_to(root)),
                str((app / "templates" / "diagnostics.html").relative_to(root)),
                str(base_html.relative_to(root)),
            ],
            "description": (
                "Add a Diagnostics tab for deep-diving into cluster issues: GPU error logs, "
                "NCCL failure traces, node-level resource breakdowns, and scheduler decision "
                "audit trails."
            ),
        })

    return gaps


def print_report() -> None:
    """Print a human-readable gap analysis report to stdout."""
    gaps = analyze_gaps()
    if not gaps:
        print("No gaps found — dashboard is fully aligned with production targets.")
        return

    # Group by priority
    by_priority: Dict[str, List[Dict]] = {}
    for gap in gaps:
        by_priority.setdefault(gap["priority"], []).append(gap)

    total = len(gaps)
    missing = sum(1 for g in gaps if g["status"] == "missing")
    partial = total - missing

    print(f"Dashboard Gap Analysis: {total} gaps found ({missing} missing, {partial} partial)\n")
    print("=" * 72)

    for priority in ["P0", "P1", "P2", "P3"]:
        items = by_priority.get(priority, [])
        if not items:
            continue
        label = {
            "P0": "P0 - Critical (trivial fixes)",
            "P1": "P1 - Important (medium effort)",
            "P2": "P2 - Desirable (substantial effort)",
            "P3": "P3 - Future (complex)",
        }[priority]
        print(f"\n{label}")
        print("-" * 72)
        for gap in items:
            marker = "MISSING" if gap["status"] == "missing" else "PARTIAL"
            print(f"  [{marker}] {gap['feature']}  ({gap['complexity']})")
            print(f"           {gap['description']}")
            print(f"           Files: {', '.join(gap['files'])}")
            print()

    print("=" * 72)
    print(f"Total: {total} gaps | Run analyze_gaps() for programmatic access.")


if __name__ == "__main__":
    print_report()
