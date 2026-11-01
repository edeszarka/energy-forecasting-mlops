"""
Unit tests for src/retrain_trigger.py (spec 011).
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrain_trigger import read_pending_since, trigger_retraining_job, write_retrain_flag


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


# ───────────── spec 018: pending flag written by 09_error_monitor ─────────────

WRITTEN_AT = datetime(2026, 11, 20, 5, 0, tzinfo=UTC)


def test_flag_round_trip(tmp_path):
    path = tmp_path / "flags" / "retrain_requested.flag"

    write_retrain_flag(path, "degraded", 24, job_run_id=987, now=WRITTEN_AT)

    payload = json.loads(path.read_text())
    assert payload["trigger"] == "forecast_error"
    assert payload["consecutive_hours"] == 24
    assert payload["drifted_features"] == []  # 08 copies it into promotion_log
    assert payload["job_run_id"] == 987
    assert read_pending_since(path) == WRITTEN_AT


def test_no_flag_means_nothing_pending(tmp_path):
    assert read_pending_since(tmp_path / "missing.flag") is None


def test_unreadable_flag_does_not_block(tmp_path):
    path = tmp_path / "retrain_requested.flag"
    path.write_text("not json")

    assert read_pending_since(path) is None
