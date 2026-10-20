"""Starts the retraining job from the hourly drift check (spec 011).

Fire-and-forget: the hourly job must not wait ~1h for training to finish,
otherwise `max_concurrent_runs: 1` would block the next hourly forecast.
"""

from __future__ import annotations

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
