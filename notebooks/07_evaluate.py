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
# **Purpose:** The promotion gate (spec 021, ADR-0009). An as-of model, built by the
# challenger's code on the rows up to the champion's cutoff, is compared with the champion on
# the rows after it, both on the same point-in-time features as training and serving
# (spec 019). Promote if the as-of model is within PROMOTION_NON_INFERIORITY_DELTA of the
# champion and the challenger's backtest beats naive. This tests the procedure, not the data.
# **Inputs:** MLflow runs + `training_params.json` artifacts, `silver_features` (load),
# `bronze_weather_forecast`
# **Outputs:** `workspace.energy_forecasting.model_evaluation` Delta table

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

from src.backtest import fit_final_lgbm
from src.config import (
    CATALOG,
    CURRENT_RUN_FILTER,
    EVALUATED_MODELS,
    PATHS,
    PROMOTION_NON_INFERIORITY_DELTA,
    SCHEMA,
)
from src.pit_features import FEATURES, build_features, issuance_hours
from src.promotion import (
    MIN_COMMON_ROWS,
    TRAINING_PARAMS_ARTIFACT,
    apply_benchmark_policy,
    as_of_train_rows,
    calculate_mape,
    decide_as_of_promotion,
    is_promotable,
    pick_production_run,
    select_as_of_window,
    served_data_end,
)

# COMMAND ----------

CONFIG = {
    "eval_table": PATHS.table_eval,
    "silver_table": PATHS.table_silver,
    "weather_table": PATHS.table_weather_forecast,
}

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
        filter_string=(
            f"tags.model_name = '{model_name}' AND tags.leak_affected = 'false' "
            f"AND {CURRENT_RUN_FILTER}"
        ),
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
            "AND tags.leak_affected = 'false' "
            f"AND {CURRENT_RUN_FILTER}"
        ),
        max_results=50,
    )
    return pick_production_run(runs, exclude_run_id=challenger_run_id)


_INPUT_CACHE: dict[str, pd.DataFrame] = {}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual load and as-issued weather forecasts, read once."""
    if not _INPUT_CACHE:
        load = spark.read.table(CONFIG["silver_table"]).select("timestamp", "value_mwh").toPandas()
        weather = spark.read.table(CONFIG["weather_table"]).toPandas()
        for frame in (load, weather):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        _INPUT_CACHE.update(load=load, weather=weather)
    return _INPUT_CACHE["load"], _INPUT_CACHE["weather"]


def score_as_of(challenger, champion) -> dict:
    """As-of model and champion on the rows after the champion's cutoff (spec 021)."""
    cutoff = served_data_end(champion)
    load, weather = load_inputs()
    data = build_features(load, weather, issuance_hours(load, weather))
    data = data.dropna(subset=["target_mwh"])
    common = select_as_of_window(data, cutoff)
    if len(common) < MIN_COMMON_ROWS["lgbm"]:
        return {
            "as_of_mape": None,
            "champion_mape": None,
            "comparison_method": "insufficient_common_rows",
            "n_common_rows": len(common),
        }

    run_id = challenger.info.run_id
    params = mlflow.artifacts.load_dict(f"runs:/{run_id}/{TRAINING_PARAMS_ARTIFACT}")
    as_of_model, _ = fit_final_lgbm(as_of_train_rows(data, cutoff), params)

    champion_model = mlflow.lightgbm.load_model(f"runs:/{champion.info.run_id}/model")
    if list(champion_model.booster_.feature_name()) != list(FEATURES):
        raise ValueError(f"Champion {champion.info.run_id} was trained on other features.")

    return {
        "as_of_mape": calculate_mape(common["target_mwh"], as_of_model.predict(common[FEATURES])),
        "champion_mape": calculate_mape(
            common["target_mwh"], champion_model.predict(common[FEATURES])
        ),
        "comparison_method": "as_of",
        "n_common_rows": len(common),
    }


def compare_as_of(challenger, champion, model_name: str) -> dict:
    """score_as_of, except that any failure leaves no result, so the champion stays."""
    try:
        return score_as_of(challenger, champion)
    except Exception as e:
        logger.warning(f"{model_name}: as-of comparison failed, keeping the champion: {e}")
        return {
            "as_of_mape": None,
            "champion_mape": None,
            "comparison_method": "not_scored",
            "n_common_rows": 0,
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
            "as_of_mape": None,
            "champion_mape": None,
            "comparison_method": "first_run",
            "n_common_rows": 0,
        }
    else:
        comparison = compare_as_of(challenger, champion, name)

    challenger_wins, decision_reason = decide_as_of_promotion(
        comparison["champion_mape"],
        comparison["as_of_mape"],
        challenger.data.metrics.get("mape"),
        challenger.data.metrics.get("naive_mape"),
        comparison["n_common_rows"],
        PROMOTION_NON_INFERIORITY_DELTA,
        first_run,
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
            # The as-of model on the common window; otherwise the logged backtest (spec 021).
            "challenger_mape": comparison["as_of_mape"]
            if comparison["comparison_method"] == "as_of"
            else metrics.get("mape"),
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
        f"WARNING: {row['model_name']}'s backtest does NOT beat the naive baseline "
        f"({row['baseline_mape']:.2f}%)."
    )

print("\nFinal Recommendations:")
for _, row in pdf.iterrows():
    status = "PROMOTE" if row["challenger_wins"] else "KEEP CHAMPION"
    print(f"{status}: {row['model_name']} — {row['decision_reason']}")

dbutils.notebook.exit("SUCCESS")
