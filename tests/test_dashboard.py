"""
Unit tests for the Streamlit dashboard module.

External dependencies (streamlit, databricks, plotly) are mocked at module
level so that src.dashboard can be imported in environments where those
packages are not installed (e.g. CI).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Mock external deps BEFORE any import of src.dashboard
# ──────────────────────────────────────────────────────────────────────

# -- Streamlit ---------------------------------------------------------
_mock_st = MagicMock()
_mock_st.secrets = {"databricks": {"host": "h", "http_path": "p", "token": "t"}}
_mock_st.cache_data = lambda *a, **kw: a[0] if a and callable(a[0]) else lambda f: f
_mock_st.cache_resource = lambda f: f
_mock_st.columns = lambda n, **kw: [MagicMock() for _ in range(n)]
_mock_st.tabs = lambda labels: [MagicMock() for _ in labels]
_mock_st.sidebar = MagicMock()
_mock_st.sidebar.slider = lambda *a, **kw: 720
sys.modules["streamlit"] = _mock_st

# -- Databricks SQL ----------------------------------------------------
_empty_fcst_df = pd.DataFrame(
    columns=[
        "timestamp",
        "predicted_mwh",
        "actual_mwh",
        "model_name",
        "horizon_hours",
        "model_version",
    ]
)
_mock_cursor = MagicMock()
_mock_cursor.fetchall_arrow.return_value = MagicMock()
_mock_cursor.fetchall_arrow.return_value.to_pandas.return_value = _empty_fcst_df
_mock_cm = MagicMock()
_mock_cm.__enter__.return_value = _mock_cursor
_mock_cm.__exit__.return_value = None
_mock_conn = MagicMock()
_mock_conn.cursor.return_value = _mock_cm
_mock_sql = MagicMock()
_mock_sql.connect.return_value = _mock_conn
_mock_db = MagicMock()
_mock_db.sql = _mock_sql
sys.modules["databricks"] = _mock_db
sys.modules["databricks.sql"] = _mock_sql

# -- Plotly ------------------------------------------------------------
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["plotly.subplots"] = MagicMock()

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestR2Score:
    """Tests for the r2_score helper."""

    def test_perfect(self):
        from src.dashboard import r2_score

        s = pd.Series([1.0, 2.0, 3.0])
        assert r2_score(s, s) == 1.0

    def test_zero_variance(self):
        from src.dashboard import r2_score

        actual = pd.Series([5.0, 5.0, 5.0])
        predicted = pd.Series([3.0, 3.0, 3.0])
        assert r2_score(actual, predicted) == 0.0

    def test_negative(self):
        from src.dashboard import r2_score

        actual = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([10.0, 20.0, 30.0])
        assert r2_score(actual, predicted) < 0

    def test_actual_predicted_swapped(self):
        from src.dashboard import r2_score

        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([1.0, 2.0, 3.0])
        result = r2_score(actual, predicted)
        assert isinstance(result, float)


class TestQuery:
    """Tests for the query helper."""

    def test_calls_connect_and_returns_dataframe(self):
        from src.dashboard import query

        mock_conn_inner = MagicMock()
        mock_cursor_inner = MagicMock()
        expected = pd.DataFrame({"x": [1]})
        mock_cursor_inner.fetchall_arrow.return_value = MagicMock()
        mock_cursor_inner.fetchall_arrow.return_value.to_pandas.return_value = expected
        mock_cm_inner = MagicMock()
        mock_cm_inner.__enter__.return_value = mock_cursor_inner
        mock_cm_inner.__exit__.return_value = None
        mock_conn_inner.cursor.return_value = mock_cm_inner

        with patch("src.dashboard.get_connection", return_value=mock_conn_inner):
            result = query("SELECT 1")

        pd.testing.assert_frame_equal(result, expected)
        mock_cursor_inner.execute.assert_called_once_with("SELECT 1")


class TestLoadForecasts:
    """Tests for the load_forecasts function."""

    @patch("src.dashboard.query")
    def test_returns_dataframe_with_converted_timestamps(self, mock_query):
        from src.dashboard import load_forecasts

        raw = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "predicted_mwh": [100.0, 200.0],
                "actual_mwh": [110.0, 210.0],
                "model_name": ["m1", "m1"],
                "horizon_hours": [24, 24],
                "model_version": ["v1", "v1"],
            }
        )
        mock_query.return_value = raw

        df = load_forecasts(48)

        assert len(df) == 2
        assert df["predicted_mwh"].iloc[0] == 100.0
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        mock_query.assert_called_once()

    @patch("src.dashboard.query")
    def test_returns_empty_when_no_data(self, mock_query):
        from src.dashboard import load_forecasts

        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "predicted_mwh",
                "actual_mwh",
                "model_name",
                "horizon_hours",
                "model_version",
            ]
        )
        mock_query.return_value = empty

        df = load_forecasts(48)
        assert df.empty


class TestLoadDrift:
    """Tests for the load_drift function."""

    @patch("src.dashboard.query")
    def test_returns_dataframe_with_converted_timestamps(self, mock_query):
        from src.dashboard import load_drift

        raw = pd.DataFrame(
            {
                "check_timestamp": ["2024-01-01 00:00:00"],
                "window_start": ["2024-01-01"],
                "window_end": ["2024-01-02"],
                "data_drift_detected": [False],
                "prediction_drift_detected": [False],
                "any_drift_detected": [False],
                "n_drifted_features": [0],
                "drifted_features": [""],
                "drift_score_value_mwh": [0.05],
                "drift_score_temp": [0.03],
                "prediction_mae_current": [10.0],
                "prediction_mae_reference": [9.0],
                "consecutive_drift_hours": [0],
                "retrain_triggered": [False],
            }
        )
        mock_query.return_value = raw

        df = load_drift(48)

        assert len(df) == 1
        assert pd.api.types.is_datetime64_any_dtype(df["check_timestamp"])
        mock_query.assert_called_once()

    @patch("src.dashboard.query")
    def test_returns_empty_when_no_data(self, mock_query):
        from src.dashboard import load_drift

        empty = pd.DataFrame(
            columns=[
                "check_timestamp",
                "window_start",
                "window_end",
                "data_drift_detected",
                "prediction_drift_detected",
                "any_drift_detected",
                "n_drifted_features",
                "drifted_features",
                "drift_score_value_mwh",
                "drift_score_temp",
                "prediction_mae_current",
                "prediction_mae_reference",
                "consecutive_drift_hours",
                "retrain_triggered",
            ]
        )
        mock_query.return_value = empty

        df = load_drift(48)
        assert df.empty


class TestModuleLevel:
    """Verifies module-level code (decorators, sidebar, tabs) runs safely."""

    def test_module_imports_and_executes_top_level(self):
        from src.dashboard import (
            CATALOG,
            DRIFT_TABLE,
            GOLD_TABLE,
            HOST,
            HTTP_PATH,
            SCHEMA,
            TOKEN,
            get_connection,
            load_drift,
            load_forecasts,
            query,
            r2_score,
        )

        assert HOST == "h"
        assert HTTP_PATH == "p"
        assert TOKEN == "t"
        assert CATALOG == "workspace"
        assert SCHEMA == "energy_forecasting"
        assert "gold_forecasts" in GOLD_TABLE
        assert "drift_control" in DRIFT_TABLE
        assert callable(get_connection)
        assert callable(load_forecasts)
        assert callable(load_drift)
        assert callable(query)
        assert callable(r2_score)
