# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

# %% [markdown]
# # 07_evaluate
# **Purpose:** Compare new Challenger models against the current Production run, scoring
# both on the same out-of-sample rows (spec 011).
# **Inputs:** MLflow runs + `backtest_predictions.json` artifacts, `workspace.energy_forecasting.silver_features`
# **Outputs:** `workspace.energy_forecasting.model_evaluation` Delta table
# **Last Updated:** 2026-10-20

# COMMAND ----------

import logging
from datetime import UTC, datetime

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config import CATALOG, EVALUATED_MODELS, MODEL_INPUT_FEATURES, PATHS, SCHEMA
from src.promotion import (
    BACKTEST_ARTIFACT,
    MIN_COMMON_ROWS,
    apply_benchmark_policy,
    calculate_mape,
    decide_promotion,
    is_promotable,
    pick_production_run,
    select_common_window,
    served_data_end,
    to_naive_utc,
)
from src.target_construction import build_horizon_target

# COMMAND ----------

dbutils.widgets.text("mape_improvement_threshold", "0.01")
THRESHOLD = float(dbutils.widgets.get("mape_improvement_threshold"))

CONFIG = {"eval_table": PATHS.table_eval, "silver_table": PATHS.table_silver}

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("07_evaluate")
client = MlflowClient()
EXPERIMENT_IDS = [e.experiment_id for e in client.search_experiments()]

# COMMAND ----------


def get_challenger_run(model_name: str):
    """The challenger is the latest valid run for this model (just trained by 05/06)."""
    runs = client.search_runs(
        experiment_ids=EXPERIMENT_IDS,
        filter_string=f"tags.model_name = '{model_name}' AND tags.leak_affected = 'false'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_champion_run(model_name: str, challenger_run_id: str):
    """The champion is the run actually serving (production=true) — the same run
    04_predict loads — not the best historical MAPE."""
    runs = client.search_runs(
        experiment_ids=EXPERIMENT_IDS,
        filter_string=(
            f"tags.model_name = '{model_name}' AND tags.production = 'true' "
            "AND tags.leak_affected = 'false'"
        ),
        max_results=50,
    )
    return pick_production_run(runs, exclude_run_id=challenger_run_id)


def load_backtest(run) -> pd.DataFrame | None:
    """Per-row backtest predictions logged by 05/06 (None for pre-spec-011 runs)."""
    try:
        data = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/{BACKTEST_ARTIFACT}")
    except Exception as e:
        logger.warning(f"No backtest artifact on run {run.info.run_id}: {e}")
        return None
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).map(to_naive_utc)
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"]).map(to_naive_utc)
    return df


_SILVER_CACHE: dict[str, pd.DataFrame] = {}


def load_silver() -> pd.DataFrame:
    if "pdf" not in _SILVER_CACHE:
        pdf = spark.read.table(CONFIG["silver_table"]).toPandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"]).map(to_naive_utc)
        _SILVER_CACHE["pdf"] = pdf
    return _SILVER_CACHE["pdf"]


def predict_with_champion(champion, model_name: str, horizon: int, rows: pd.DataFrame) -> pd.Series:
    """Scores the champion's served model on exactly the challenger's backtest rows,
    building inputs the same way 05/06 built them for the challenger."""
    model_uri = f"runs:/{champion.info.run_id}/model"
    silver = load_silver()

    if "lgbm" in model_name:
        model = mlflow.lightgbm.load_model(model_uri)
        feats = build_horizon_target(silver, horizon).set_index("timestamp")
        trained = model.booster_.feature_name()
        X = feats.reindex(rows["timestamp"])[[c for c in trained if c in MODEL_INPUT_FEATURES]]
        complete = X.notna().all(axis=1).values
        X = X.astype({c: bool for c in ("is_weekend", "is_holiday") if c in X.columns})
        preds = pd.Series(float("nan"), index=rows.index)
        if complete.any():
            preds[complete] = model.predict(X[complete])
        return preds

    model = mlflow.prophet.load_model(model_uri)
    temps = silver.set_index("timestamp")["temperature_c"].reindex(rows["target_timestamp"])
    complete = temps.notna().values
    preds = pd.Series(float("nan"), index=rows.index)
    if complete.any():
        p_df = pd.DataFrame(
            {
                "ds": rows["target_timestamp"].values[complete],
                "temperature_c": temps.values[complete],
            }
        )
        preds[complete] = model.predict(p_df)["yhat"].values
    return preds


def compare_on_common_window(challenger, champion, model_name: str, horizon: int) -> dict:
    """Same-window MAPE for both models. Falls back to each run's logged metric only
    when too few rows are out-of-sample for BOTH models (e.g. pre-spec-011 runs)."""
    logged = {
        "challenger_mape": challenger.data.metrics.get("mape"),
        "champion_mape": champion.data.metrics.get("mape"),
        "comparison_method": "logged_metrics",
        "n_common_rows": 0,
    }

    backtest = load_backtest(challenger)
    if backtest is None:
        return logged

    family = "lgbm" if "lgbm" in model_name else "prophet"
    common = select_common_window(backtest, served_data_end(champion))
    if len(common) < MIN_COMMON_ROWS[family]:
        logger.warning(
            f"{model_name}: only {len(common)} backtest rows are out-of-sample for the "
            f"champion (need {MIN_COMMON_ROWS[family]}). Falling back to logged metrics."
        )
        return logged

    champion_pred = predict_with_champion(champion, model_name, horizon, common)
    valid = champion_pred.notna() & common["actual"].notna()
    if valid.sum() < MIN_COMMON_ROWS[family]:
        logger.warning(f"{model_name}: champion could only score {valid.sum()} common rows.")
        return logged

    return {
        "challenger_mape": calculate_mape(
            common.loc[valid, "actual"], common.loc[valid, "predicted"]
        ),
        "champion_mape": calculate_mape(common.loc[valid, "actual"], champion_pred[valid]),
        "comparison_method": "same_window",
        "n_common_rows": int(valid.sum()),
    }


# COMMAND ----------

# Main execution
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")

eval_rows = []

for name in EVALUATED_MODELS:
    horizon = 24  # ADR-0001: 24h only

    challenger = get_challenger_run(name)
    if challenger is None:
        logger.warning(f"No challenger run found for {name}. Skipping.")
        continue
    # Benchmarks always get a row: historic Prophet runs keep their production tag.
    if is_promotable(name) and challenger.data.tags.get("production") == "true":
        logger.info(f"{name}: latest run {challenger.info.run_id} is already production. Skipping.")
        continue

    champion = get_champion_run(name, challenger.info.run_id)
    first_run = champion is None

    if first_run:
        comparison = {
            "challenger_mape": challenger.data.metrics.get("mape"),
            "champion_mape": None,
            "comparison_method": "first_run",
            "n_common_rows": 0,
        }
    else:
        comparison = compare_on_common_window(challenger, champion, name, horizon)

    challenger_wins, decision_reason = decide_promotion(
        comparison["champion_mape"], comparison["challenger_mape"], THRESHOLD, first_run
    )
    challenger_wins, decision_reason = apply_benchmark_policy(
        name, challenger_wins, decision_reason
    )
    logger.info(f"{name} [{comparison['comparison_method']}]: {decision_reason}")

    metrics = challenger.data.metrics
    baseline_mape = metrics.get("naive_mape")
    beats_baseline = None
    if metrics.get("mape") is not None and baseline_mape is not None:
        beats_baseline = metrics["mape"] < baseline_mape

    champion_metrics = champion.data.metrics if champion else {}
    eval_rows.append(
        {
            "model_name": name,
            "horizon_hours": horizon,
            "challenger_run_id": challenger.info.run_id,
            "challenger_mae": metrics.get("mae"),
            "challenger_rmse": metrics.get("rmse"),
            "challenger_mape": comparison["challenger_mape"],
            "baseline_mae": metrics.get("naive_mae"),
            "baseline_rmse": metrics.get("naive_rmse"),
            "baseline_mape": baseline_mape,
            "beats_baseline": beats_baseline,
            "champion_run_id": champion.info.run_id if champion else None,
            "champion_mae": champion_metrics.get("mae"),
            "champion_rmse": champion_metrics.get("rmse"),
            "champion_mape": comparison["champion_mape"],
            "challenger_wins": challenger_wins,
            "first_run": first_run,
            "comparison_method": comparison["comparison_method"],
            "n_common_rows": comparison["n_common_rows"],
            "decision_reason": decision_reason,
            "evaluated_at": datetime.now(UTC),
            "promoted": False,  # Set to true by 08 once the row has been processed
        }
    )

if not eval_rows:
    logger.info("No new challenger models to evaluate.")
    dbutils.notebook.exit("NO_CHALLENGERS")

# Write results to Delta
EVAL_SCHEMA = StructType(
    [
        StructField("model_name", StringType(), False),
        StructField("horizon_hours", IntegerType(), False),
        StructField("challenger_run_id", StringType(), False),
        StructField("challenger_mae", DoubleType(), True),
        StructField("challenger_rmse", DoubleType(), True),
        StructField("challenger_mape", DoubleType(), True),
        StructField("baseline_mae", DoubleType(), True),
        StructField("baseline_rmse", DoubleType(), True),
        StructField("baseline_mape", DoubleType(), True),
        StructField("beats_baseline", BooleanType(), True),
        StructField("champion_run_id", StringType(), True),
        StructField("champion_mae", DoubleType(), True),
        StructField("champion_rmse", DoubleType(), True),
        StructField("champion_mape", DoubleType(), True),
        StructField("challenger_wins", BooleanType(), False),
        StructField("first_run", BooleanType(), False),
        StructField("comparison_method", StringType(), True),
        StructField("n_common_rows", IntegerType(), True),
        StructField("decision_reason", StringType(), True),
        StructField("evaluated_at", TimestampType(), False),
        StructField("promoted", BooleanType(), False),
    ]
)

eval_df = spark.createDataFrame(eval_rows, schema=EVAL_SCHEMA)
eval_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
    CONFIG["eval_table"]
)

# Print recommendation summary
print("\nModel Evaluation Summary:")
pdf = eval_df.toPandas()
print(
    pdf[
        [
            "model_name",
            "challenger_mape",
            "baseline_mape",
            "champion_mape",
            "beats_baseline",
            "challenger_wins",
            "comparison_method",
        ]
    ].to_string(index=False)
)

for _, row in pdf[pdf["beats_baseline"].eq(False)].iterrows():
    print(
        f"WARNING: {row['model_name']} does NOT beat the naive baseline "
        f"(model {row['challenger_mape']:.2f}% vs baseline {row['baseline_mape']:.2f}%)."
    )

print("\nFinal Recommendations:")
for _, row in pdf.iterrows():
    status = "PROMOTE" if row["challenger_wins"] else "KEEP CHAMPION"
    print(f"{status}: {row['model_name']} — {row['decision_reason']}")

dbutils.notebook.exit("SUCCESS")
