# Databricks notebook source
# MAGIC %md
# MAGIC # Remediate pre-014 MLflow runs (spec 014 T6, ADR-0003)
# MAGIC
# MAGIC One-time. Every `energy_lgbm_24h` / `energy_prophet_24h` / legacy 168h run without a
# MAGIC `data_version` tag gets `data_version=1`, `invalidated_reason=spec014_pt15m_parsing`
# MAGIC and `production=false`. Prints the before/after tags of every run it touches.
# MAGIC
# MAGIC `dry_run` (default `true`) prints the plan and changes nothing. Re-running is safe:
# MAGIC runs that already carry `data_version` are skipped.

# COMMAND ----------

import os
import sys
from pathlib import Path

# The script lives in scripts/; add the project root so `src` imports work.
root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

from mlflow.tracking import MlflowClient  # noqa: E402 (after sys.path setup)

from src.data_version import REMEDIATION_MODELS, plan_data_version_remediation  # noqa: E402

# COMMAND ----------


def search_model_runs(client, experiment_ids: list[str], model_names: tuple[str, ...]) -> list:
    """All runs tagged with one of `model_names`, following pagination."""
    runs = []
    for model_name in model_names:
        page_token = None
        while True:
            page = client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=f"tags.model_name = '{model_name}'",
                max_results=1000,
                page_token=page_token,
            )
            runs.extend(page)
            page_token = getattr(page, "token", None)
            if not page_token:
                break
    return runs


def remediate(client, experiment_ids: list[str], dry_run: bool = True) -> list:
    """Prints the before/after tags of every run to fix; sets them only when not dry_run."""
    plan = plan_data_version_remediation(
        search_model_runs(client, experiment_ids, REMEDIATION_MODELS)
    )
    print(f"{'DRY RUN — no changes' if dry_run else 'EXECUTE'}: {len(plan)} runs to remediate")
    for item in plan:
        print(f"{item.run_id} [{item.model_name}]")
        for key, value in item.after.items():
            print(f"    {key}: {item.before[key]!r} -> {value!r}")
        if not dry_run:
            for key, value in item.after.items():
                client.set_tag(item.run_id, key, value)
    return plan


# COMMAND ----------

if __name__ == "__main__":
    try:
        dbutils.widgets.text("dry_run", "true")
        dry_run = dbutils.widgets.get("dry_run").lower() != "false"
    except NameError:  # outside Databricks
        dry_run = "--execute" not in sys.argv

    mlflow_client = MlflowClient()
    experiment_ids = [e.experiment_id for e in mlflow_client.search_experiments()]
    remediate(mlflow_client, experiment_ids, dry_run=dry_run)
