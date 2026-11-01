"""Benchmark-only guards in notebooks/08_promote_model.py (spec 013 T2, AC2).

The notebook cannot be imported (module-level ``dbutils``/``spark`` calls), so the
functions under test are extracted via ``ast`` and exec'd in a stubbed namespace,
following ``tests/test_predict_timing_integration.py``.
"""

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.promotion import BENCHMARK_ONLY_REASON, is_promotable, pick_production_run

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "08_promote_model.py"


def _load_notebook_functions(*names: str) -> dict:
    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "pd": pd,
        "datetime": datetime,
        "UTC": UTC,
        "logger": logging.getLogger("promote-test"),
        "MlflowClient": object,
        "BENCHMARK_ONLY_REASON": BENCHMARK_ONLY_REASON,
        "is_promotable": is_promotable,
        "pick_production_run": pick_production_run,
        "CURRENT_DATA_VERSION": "2",
        "CURRENT_RUN_FILTER": "tags.data_version = '2' AND tags.feature_set = 'pit_24h_v1'",
    }
    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _FakeMlflowClient:
    def __init__(self, production_runs=()):
        self.production_runs = list(production_runs)
        self.tags_set = []

    def search_experiments(self):
        return [SimpleNamespace(experiment_id="1")]

    def search_runs(self, experiment_ids, filter_string, **kwargs):
        return [r for r in self.production_runs if f"'{r.model_name}'" in filter_string]

    def set_tag(self, run_id, key, value):
        self.tags_set.append((run_id, key, value))


def _eval_row(model_name, challenger_run_id="challenger", champion_run_id=None):
    return {
        "model_name": model_name,
        "challenger_run_id": challenger_run_id,
        "challenger_mape": 4.0,
        "champion_run_id": champion_run_id,
        "champion_mape": None,
        "challenger_wins": True,  # a (pre-spec-013) winning row
        "first_run": champion_run_id is None,
        "decision_reason": "No Production run exists — promoting unconditionally",
    }


CONFIG = {"force_promote": True, "dry_run": False}


def test_benchmark_row_is_not_promoted_even_with_force():
    fns = _load_notebook_functions("get_production_runs", "decide_promotions")
    eval_df = pd.DataFrame([_eval_row("energy_prophet_24h")])

    decisions = fns["decide_promotions"](eval_df, _FakeMlflowClient(), CONFIG)

    assert decisions[0]["should_promote"] is False
    assert decisions[0]["promotion_reason"] == BENCHMARK_ONLY_REASON


def test_promotable_row_still_honours_force():
    fns = _load_notebook_functions("get_production_runs", "decide_promotions")
    eval_df = pd.DataFrame([_eval_row("energy_lgbm_24h")])

    decisions = fns["decide_promotions"](eval_df, _FakeMlflowClient(), CONFIG)

    assert decisions[0]["should_promote"] is True


def test_execute_promotions_refuses_to_tag_a_benchmark():
    fns = _load_notebook_functions("get_production_runs", "execute_promotions")
    client = _FakeMlflowClient()
    decisions = [
        {
            "model_name": "energy_prophet_24h",
            "challenger_run_id": "prophet-run",
            "should_promote": True,  # bypassed decide_promotions
        }
    ]

    result = fns["execute_promotions"](decisions, client, CONFIG)

    assert client.tags_set == []
    assert result[0]["should_promote"] is False
