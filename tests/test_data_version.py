"""Data versioning of models (spec 014 T6, ADR-0003; AC7).

Notebook and script functions are extracted via ``ast`` as in
``tests/test_predict_timing_integration.py``. The fake MLflow client evaluates
the `tags.<key> = '<value>'` clauses of each filter string, so a reader that
forgets the data_version clause sees the pre-014 runs and fails the test.
"""

import ast
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data_version import (
    DATA_VERSION_PROPERTY,
    INVALIDATED_REASON_014,
    plan_data_version_remediation,
    read_data_version,
    table_properties,
)
from src.promotion import pick_production_run

ROOT = Path(__file__).resolve().parents[1]
CLAUSE = re.compile(r"tags\.(\w+) = '([^']*)'")


def make_run(run_id, model_name="energy_lgbm_24h", start_ms=0, **tags):
    tags = {"model_name": model_name, "leak_affected": "false", **tags}
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_ms),
        data=SimpleNamespace(tags=tags, metrics={"mape": 5.0}),
    )


class _PagedList(list):
    token = None


class _FilteringMlflowClient:
    def __init__(self, runs):
        self.runs = runs
        self.filters = []
        self.tags_set = []

    def search_experiments(self):
        return [SimpleNamespace(experiment_id="1")]

    def search_runs(self, experiment_ids, filter_string, order_by=None, max_results=None, **kw):
        self.filters.append(filter_string)
        clauses = CLAUSE.findall(filter_string)
        hits = [r for r in self.runs if all(r.data.tags.get(k) == v for k, v in clauses)]
        if order_by == ["attributes.start_time DESC"]:
            hits.sort(key=lambda r: r.info.start_time, reverse=True)
        return _PagedList(hits[:max_results] if max_results else hits)

    def set_tag(self, run_id, key, value):
        self.tags_set.append((run_id, key, value))


def _functions(relative_path: str, *names: str, **namespace) -> dict:
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "pd": pd,
        "datetime": datetime,
        "timedelta": timedelta,
        "UTC": UTC,
        "logger": logging.getLogger("data-version-test"),
        "MlflowClient": object,
        "SparkSession": object,
        "pick_production_run": pick_production_run,
        "CURRENT_DATA_VERSION": "2",
        **namespace,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


# Pre-014 production run (no data_version, promoted later) vs a version-2 run.
OLD_PROD = make_run(
    "old", start_ms=2_000, production="true", promoted_at="2026-10-21T21:15:00+00:00"
)
NEW_PROD = make_run(
    "new",
    start_ms=1_000,
    production="true",
    promoted_at="2026-10-01T00:00:00+00:00",
    data_version="2",
    training_data_start="2026-09-01T00:00:00",
    training_data_end="2026-09-30T00:00:00",
)


# --- table property and tagging ------------------------------------------------


def test_data_version_defaults_to_1_without_the_property():
    assert read_data_version({}) == "1"
    assert read_data_version({DATA_VERSION_PROPERTY: "2", "delta.x": "y"}) == "2"


def test_table_properties_from_show_tblproperties_rows():
    rows = [
        {"key": DATA_VERSION_PROPERTY, "value": "2"},
        {"key": "delta.minReaderVersion", "value": "1"},
    ]
    assert read_data_version(table_properties(rows)) == "2"


def test_training_notebooks_tag_data_version():
    for notebook in ("notebooks/05_train_prophet.py", "notebooks/06_train_lgbm.py"):
        source = (ROOT / notebook).read_text(encoding="utf-8")
        assert 'mlflow.set_tag("data_version", CONFIG["data_version"])' in source
        assert "SHOW TBLPROPERTIES" in source


# --- AC7: readers ignore runs without data_version='2' ---------------------------


class _ModelNotFound(Exception):
    pass


def test_04_loader_ignores_pre_014_production_run():
    client = _FilteringMlflowClient([OLD_PROD, NEW_PROD])
    mlflow = SimpleNamespace(lightgbm=SimpleNamespace(load_model=lambda uri: uri))
    load = _functions(
        "notebooks/04_predict.py",
        "load_production_model",
        mlflow=mlflow,
        ModelNotFoundError=_ModelNotFound,
    )["load_production_model"]

    model, _, run_id = load("energy_lgbm_24h", client)

    assert run_id == "new"
    assert model == "runs:/new/model"


def test_04_loader_finds_nothing_when_only_pre_014_runs_exist():
    client = _FilteringMlflowClient([OLD_PROD])
    load = _functions(
        "notebooks/04_predict.py",
        "load_production_model",
        mlflow=SimpleNamespace(),
        ModelNotFoundError=_ModelNotFound,
    )["load_production_model"]

    try:
        load("energy_lgbm_24h", client)
    except _ModelNotFound:
        pass
    else:
        raise AssertionError("a pre-014 run was loaded")


def _evaluate_functions(client):
    return _functions(
        "notebooks/07_evaluate.py",
        "get_challenger_run",
        "get_champion_run",
        client=client,
        EXPERIMENT_IDS=["1"],
    )


def test_07_challenger_ignores_newer_pre_014_run():
    old_latest = make_run("old-latest", start_ms=9_000)
    new = make_run("new-run", start_ms=1_000, data_version="2")
    fns = _evaluate_functions(_FilteringMlflowClient([old_latest, new]))

    assert fns["get_challenger_run"]("energy_lgbm_24h").info.run_id == "new-run"


def test_07_champion_ignores_pre_014_production_run():
    fns = _evaluate_functions(_FilteringMlflowClient([OLD_PROD, NEW_PROD]))

    assert fns["get_champion_run"]("energy_lgbm_24h", "challenger").info.run_id == "new"


def test_07_no_version_2_champion_means_first_run():
    fns = _evaluate_functions(_FilteringMlflowClient([OLD_PROD]))

    assert fns["get_champion_run"]("energy_lgbm_24h", "challenger") is None


def test_08_production_lookup_ignores_pre_014_runs():
    client = _FilteringMlflowClient([OLD_PROD, NEW_PROD])
    get_runs = _functions("notebooks/08_promote_model.py", "get_production_runs")[
        "get_production_runs"
    ]

    assert [r.info.run_id for r in get_runs("energy_lgbm_24h", client)] == ["new"]


class _FakeFrame:
    def __init__(self, calls):
        self.calls = calls

    def filter(self, condition):
        self.calls.append(condition)
        return self

    def select(self, cols):
        return self

    def toPandas(self):
        return pd.DataFrame()


class _FakeCol:
    def __init__(self, name):
        self.name = name

    def between(self, start, end):
        return (self.name, start, end)


def test_03_reference_window_uses_only_version_2_runs():
    client = _FilteringMlflowClient([OLD_PROD, NEW_PROD])
    calls = []
    spark = SimpleNamespace(table=lambda name: _FakeFrame(calls))
    get_reference_window = _functions(
        "notebooks/03_drift_check.py", "get_reference_window", F=SimpleNamespace(col=_FakeCol)
    )["get_reference_window"]
    config = {
        "primary_model_name": "energy_lgbm_24h",
        "silver_table": "silver",
        "target_column": "value_mwh",
        "feature_columns": [],
    }

    _, source, start, end = get_reference_window(spark, client, config)

    assert source == "mlflow_training_window (run new)"
    assert start == datetime(2026, 9, 1, tzinfo=UTC)
    assert all("tags.data_version = '2'" in f for f in client.filters)


# --- remediation ----------------------------------------------------------------


def test_plan_marks_every_unversioned_run_of_the_known_models():
    runs = [
        make_run("lgbm-old", production="true"),
        make_run("prophet-old", model_name="energy_prophet_24h"),
        make_run("lgbm168-old", model_name="energy_lgbm_168h", production="false"),
        make_run("prophet168-old", model_name="energy_prophet_168h"),
        make_run("already-versioned", data_version="2"),
        make_run("other-model", model_name="something_else"),
    ]

    plan = plan_data_version_remediation(runs)

    assert [p.run_id for p in plan] == ["lgbm-old", "prophet-old", "lgbm168-old", "prophet168-old"]
    assert plan[0].before == {
        "data_version": None,
        "invalidated_reason": None,
        "production": "true",
    }
    assert plan[0].after == {
        "data_version": "1",
        "invalidated_reason": INVALIDATED_REASON_014,
        "production": "false",
    }


def _remediate():
    from src.data_version import REMEDIATION_MODELS

    return _functions(
        "scripts/remediate_014_data_version.py",
        "search_model_runs",
        "remediate",
        REMEDIATION_MODELS=REMEDIATION_MODELS,
        plan_data_version_remediation=plan_data_version_remediation,
    )["remediate"]


def test_remediation_dry_run_prints_tags_and_touches_nothing(capsys):
    client = _FilteringMlflowClient([make_run("lgbm-old", production="true"), NEW_PROD])

    plan = _remediate()(client, ["1"])  # dry_run is the default

    out = capsys.readouterr().out
    assert client.tags_set == []
    assert [p.run_id for p in plan] == ["lgbm-old"]
    assert "DRY RUN" in out
    assert "lgbm-old [energy_lgbm_24h]" in out
    assert "production: 'true' -> 'false'" in out
    assert "data_version: None -> '1'" in out


def test_remediation_execute_sets_the_tags():
    client = _FilteringMlflowClient([make_run("lgbm-old", production="true"), NEW_PROD])

    _remediate()(client, ["1"], dry_run=False)

    assert client.tags_set == [
        ("lgbm-old", "data_version", "1"),
        ("lgbm-old", "invalidated_reason", INVALIDATED_REASON_014),
        ("lgbm-old", "production", "false"),
    ]
