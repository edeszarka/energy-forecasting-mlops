"""
Streamlit dashboard for energy forecasting.

Displays actual vs predicted values, MAE trends, and data drift
summaries by querying Databricks Delta tables via the SQL connector.

Usage:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from databricks import sql
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Secrets ──────────────────────────────────────────────────────────────────

HOST = st.secrets["databricks"]["host"]
HTTP_PATH = st.secrets["databricks"]["http_path"]
TOKEN = st.secrets["databricks"]["token"]

CATALOG = "workspace"
SCHEMA = "energy_forecasting"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_forecasts"
DRIFT_TABLE = f"{CATALOG}.{SCHEMA}.drift_control"


# ── Connection ───────────────────────────────────────────────────────────────


@st.cache_resource
def get_connection() -> sql.Connection:
    return sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN)


def query(sql_query: str) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        return cursor.fetchall_arrow().to_pandas()


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner="Loading forecasts…")
def load_forecasts(hours_back: int = 720) -> pd.DataFrame:
    cutoff = (datetime.now(UTC) - timedelta(hours=hours_back)).isoformat()
    sql_query = f"""
        SELECT timestamp, predicted_mwh, actual_mwh, model_name, horizon_hours,
               model_version
        FROM {GOLD_TABLE}
        WHERE timestamp >= '{cutoff}'
        ORDER BY timestamp
    """
    df = query(sql_query)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=300, show_spinner="Loading drift control…")
def load_drift(hours_back: int = 720) -> pd.DataFrame:
    cutoff = (datetime.now(UTC) - timedelta(hours=hours_back)).isoformat()
    sql_query = f"""
        SELECT check_timestamp, window_start, window_end,
               data_drift_detected, prediction_drift_detected, any_drift_detected,
               n_drifted_features, drifted_features,
               drift_score_value_mwh, drift_score_temp,
               prediction_mae_current, prediction_mae_reference,
               consecutive_drift_hours, retrain_triggered
        FROM {DRIFT_TABLE}
        WHERE check_timestamp >= '{cutoff}'
        ORDER BY check_timestamp
    """
    df = query(sql_query)
    if not df.empty:
        df["check_timestamp"] = pd.to_datetime(df["check_timestamp"])
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def forecast_tab(horizon: int, title: str) -> None:
    hours_back = st.sidebar.slider(
        f"{title} — lookback hours", min_value=48, max_value=2160, value=720, step=24
    )
    df = load_forecasts(hours_back)
    df_h = df[df["horizon_hours"] == horizon].copy()

    if df_h.empty:
        st.info(f"No {title} forecasts found in the selected window.")
        return

    model_names = sorted(df_h["model_name"].unique())
    selected_model = st.selectbox(f"{title} — model", model_names)
    df_m = df_h[df_h["model_name"] == selected_model].copy().sort_values("timestamp")

    has_actuals = df_m["actual_mwh"].notna().any()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_m["timestamp"],
                y=df_m["predicted_mwh"],
                mode="lines",
                name="Predicted",
                line=dict(color="#1f77b4"),
            )
        )
        if has_actuals:
            fig.add_trace(
                go.Scatter(
                    x=df_m["timestamp"],
                    y=df_m["actual_mwh"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="#ff7f0e"),
                )
            )
        fig.update_layout(
            title=f"{title} — Forecast vs Actual",
            xaxis_title="Time",
            yaxis_title="MW",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        actual = df_m["actual_mwh"].dropna()
        predicted = df_m.loc[actual.index, "predicted_mwh"]
        if len(actual) > 1:
            r2 = r2_score(actual, predicted)
            mae_v = float((actual - predicted).abs().mean())
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode="markers",
                    marker=dict(color="#2ca02c", size=4),
                    name="Actual vs Predicted",
                )
            )
            max_val = max(actual.max(), predicted.max())
            fig2.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name="Perfect fit",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig2.update_layout(
                title=f"Predicted vs Actual (R²={r2:.3f}, MAE={mae_v:.1f} MW)",
                xaxis_title="Actual MW",
                yaxis_title="Predicted MW",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("MAE", f"{mae_v:.1f} MW", delta=None)
        else:
            st.info("Not enough actuals yet for scatter plot.")


def mae_trend_tab() -> None:
    hours_back = st.sidebar.slider(
        "MAE lookback hours", min_value=48, max_value=2160, value=720, step=24
    )
    df = load_drift(hours_back)
    if df.empty:
        st.info("No drift control data found.")
        return

    latest = df.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Current MAE",
        f"{latest['prediction_mae_current']:.1f} MW"
        if pd.notna(latest["prediction_mae_current"])
        else "N/A",
    )
    k2.metric(
        "Reference MAE",
        f"{latest['prediction_mae_reference']:.1f} MW"
        if pd.notna(latest["prediction_mae_reference"])
        else "N/A",
    )
    k3.metric("Consecutive Drift Hours", str(latest["consecutive_drift_hours"]))
    k4.metric("Retrain Triggered", "Yes" if latest["retrain_triggered"] else "No")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["check_timestamp"],
            y=df["prediction_mae_current"],
            mode="lines+markers",
            name="MAE current",
            line=dict(color="#d62728"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["check_timestamp"],
            y=df["prediction_mae_reference"],
            mode="lines+markers",
            name="MAE reference",
            line=dict(color="#7f7f7f", dash="dot"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["check_timestamp"],
            y=df["consecutive_drift_hours"],
            name="Consecutive drift",
            marker_color="#ff9896",
            opacity=0.5,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Prediction MAE & Consecutive Drift Over Time",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(title_text="MAE (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Consecutive drift hours", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw drift control table"):
        st.dataframe(df.sort_values("check_timestamp", ascending=False), use_container_width=True)


def drift_tab() -> None:
    hours_back = st.sidebar.slider(
        "Drift lookback hours", min_value=48, max_value=2160, value=720, step=24
    )
    df = load_drift(hours_back)
    if df.empty:
        st.info("No drift control data found.")
        return

    display_cols = [
        "check_timestamp",
        "data_drift_detected",
        "prediction_drift_detected",
        "n_drifted_features",
        "drifted_features",
        "drift_score_value_mwh",
        "drift_score_temp",
    ]
    display = df[display_cols].copy().sort_values("check_timestamp", ascending=False)

    def color_drift(val: object) -> str:
        if isinstance(val, bool) and val:
            return "background-color: #ffcccc"
        return ""

    st.dataframe(
        display.style.map(color_drift, subset=["data_drift_detected", "prediction_drift_detected"]),
        use_container_width=True,
        height=min(60 * len(display) + 40, 600),
    )

    if not df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["check_timestamp"],
                y=df["drift_score_value_mwh"],
                mode="lines+markers",
                name="Drift score — value_mwh",
                line=dict(color="#9467bd"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["check_timestamp"],
                y=df["drift_score_temp"],
                mode="lines+markers",
                name="Drift score — temperature_c",
                line=dict(color="#8c564b"),
            )
        )
        fig.add_hline(
            y=0.15,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold (0.15)",
        )
        fig.update_layout(
            title="Feature Drift Scores Over Time",
            yaxis_title="Jensen-Shannon divergence",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ Energy Forecast")
st.sidebar.markdown(f"**Catalog:** `{CATALOG}.{SCHEMA}`")
st.sidebar.markdown(f"**Gold table:** `{GOLD_TABLE}`")
st.sidebar.markdown(f"**Drift table:** `{DRIFT_TABLE}`")
st.sidebar.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["24h Forecast", "168h Forecast", "MAE Trend", "Data Drift"])

with tab1:
    forecast_tab(24, "24h")
with tab2:
    forecast_tab(168, "168h")
with tab3:
    mae_trend_tab()
with tab4:
    drift_tab()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
