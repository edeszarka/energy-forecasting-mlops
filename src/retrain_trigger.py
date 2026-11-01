"""Starts the retraining job from the hourly error monitor (spec 011, spec 018).

Fire-and-forget: the hourly job must not wait ~1h for training to finish,
otherwise `max_concurrent_runs: 1` would block the next hourly forecast.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def trigger_retraining_job(
    job_id: int, workspace_client: Any | None = None
) -> tuple[int | None, str]:
    """Starts `job_id` unless a run of it is already active.

    Returns (run_id, message); run_id is None when no new run was started.
    `workspace_client` is injectable for tests; on Databricks it defaults to
    `databricks.sdk.WorkspaceClient()`, which picks up notebook auth.
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    active = list(workspace_client.jobs.list_runs(job_id=job_id, active_only=True))
    if active:
        return None, f"Retraining job {job_id} already running (run {active[0].run_id})."

    # Not waiting on the returned handle is what makes this fire-and-forget.
    run = workspace_client.jobs.run_now(job_id=job_id)
    return run.run_id, f"Started retraining job {job_id} (run {run.run_id})."


def read_pending_since(flag_path: str | Path) -> datetime | None:
    """When the outstanding retrain request was written, or None.

    An unreadable flag returns None: it must not block triggering forever.
    """
    path = Path(flag_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return datetime.fromisoformat(json.load(f)["triggered_at"])
    except Exception:
        return None


def write_retrain_flag(
    flag_path: str | Path,
    reason: str,
    consecutive_hours: int,
    job_run_id: int | None = None,
    trigger: str = "forecast_error",
    now: datetime | None = None,
) -> str:
    """Writes the pending-retrain flag; 08_promote_model deletes it after a retrain.

    `drifted_features` stays in the payload (empty) because 08 copies it into
    promotion_log.
    """
    payload = {
        "triggered_at": (now or datetime.now(UTC)).isoformat(),
        "trigger": trigger,
        "reason": reason,
        "consecutive_hours": consecutive_hours,
        "drifted_features": [],
        "job_run_id": job_run_id,
        "pipeline_version": "3.0",
    }
    path = Path(flag_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)
