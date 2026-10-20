"""
Unit tests for src/retrain_trigger.py (spec 011).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrain_trigger import trigger_retraining_job


def make_client(active_runs):
    client = MagicMock()
    client.jobs.list_runs.return_value = iter(active_runs)
    client.jobs.run_now.return_value = SimpleNamespace(run_id=987)
    return client


def test_starts_job_when_idle():
    client = make_client([])
    run_id, msg = trigger_retraining_job(123, workspace_client=client)

    assert run_id == 987
    client.jobs.list_runs.assert_called_once_with(job_id=123, active_only=True)
    client.jobs.run_now.assert_called_once_with(job_id=123)
    assert "Started" in msg


def test_does_not_start_second_run_when_already_running():
    client = make_client([SimpleNamespace(run_id=555)])
    run_id, msg = trigger_retraining_job(123, workspace_client=client)

    assert run_id is None
    client.jobs.run_now.assert_not_called()
    assert "already running" in msg
