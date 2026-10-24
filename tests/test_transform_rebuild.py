"""02_transform rebuild mode and dq_mode (spec 014 T4, T5, T6; AC6).

`rebuild_silver` is extracted from the notebook via ``ast`` as in
``tests/test_predict_timing_integration.py``; bronze reads and Delta writes are
replaced by recorders.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_quality import DataQualityError, apply_dq_mode, check_load_quality
from src.silver_build import build_silver_frame, plan_rebuild_chunks

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02_transform.py"
START = pd.Timestamp("2026-03-01T00:00Z")


def _bronze(stretched: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range(START - pd.Timedelta(days=8), periods=80 * 24, freq="h")
    hours = np.arange(len(ts)) / (4 if stretched else 1)
    load = pd.DataFrame(
        {
            "timestamp": ts,
            "value_mwh": 5000 + 1200 * np.sin(2 * np.pi * hours / 24),
            "is_gap": False,
            "source": "entsoe_api",
            "fetched_at": ts,
        }
    )
    temp = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_c": 10.0,
            "humidity_pct": 60.0,
            "cloud_cover_pct": 20.0,
            "is_weather_imputed": False,
        }
    )
    return load, temp


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name):
        return lambda *args: self.calls.append((name, *args))


def _rebuild_silver(stretched: bool, recorder: _Recorder):
    load, temp = _bronze(stretched)

    def read_bronze_window(spark, start, end):
        keep = lambda df: df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]  # noqa: E731
        return keep(load), keep(temp)

    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rebuild_silver"
    ]
    namespace = {
        "pd": pd,
        "plan_rebuild_chunks": plan_rebuild_chunks,
        "build_silver_frame": build_silver_frame,
        "check_load_quality": check_load_quality,
        "apply_dq_mode": apply_dq_mode,
        "read_bronze_window": read_bronze_window,
        "ensure_silver_table": recorder("ensure"),
        "write_silver_replace": recorder("write"),
        "set_silver_data_version": recorder("data_version"),
        "CURRENT_DATA_VERSION": "2",
    }
    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["rebuild_silver"]


END = START + pd.Timedelta(days=60)


def test_enforce_raises_and_writes_nothing():
    recorder = _Recorder()
    rebuild = _rebuild_silver(stretched=True, recorder=recorder)

    with pytest.raises(DataQualityError, match="daily cycle"):
        rebuild(None, START, END, "enforce", False, "run-1")

    assert recorder.calls == []


def test_warn_writes_every_chunk_then_bumps_data_version():
    recorder = _Recorder()
    rebuild = _rebuild_silver(stretched=True, recorder=recorder)

    summary = rebuild(None, START, END, "warn", False, "run-1")

    names = [c[0] for c in recorder.calls]
    assert names == ["ensure", "write", "write", "data_version"]
    assert recorder.calls[-1] == ("data_version", None, "2")
    _, _, frame, core_start, core_end, run_id = recorder.calls[1]
    assert (core_start, core_end, run_id) == (START, START + pd.Timedelta(days=30), "run-1")
    assert frame["timestamp"].min() == START and len(frame) == 30 * 24
    assert [c["rows"] for c in summary] == [720, 720]


def test_clean_bronze_passes_enforce():
    recorder = _Recorder()
    rebuild = _rebuild_silver(stretched=False, recorder=recorder)

    rebuild(None, START, END, "enforce", False, "run-1")

    assert [c[0] for c in recorder.calls] == ["ensure", "write", "write", "data_version"]


def test_dry_run_writes_nothing_and_keeps_data_version():
    recorder = _Recorder()
    rebuild = _rebuild_silver(stretched=False, recorder=recorder)

    summary = rebuild(None, START, END, "enforce", True, "run-1")

    assert recorder.calls == []
    assert len(summary) == 2
