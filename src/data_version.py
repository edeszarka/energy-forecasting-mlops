"""Data versioning of models (ADR-0003, spec 014 T6).

The silver table carries the Delta table property `pipeline.data_version`; only
a full rebuild from verified bronze bumps it. Training runs tag the version
they read, and every MLflow reader filters on CURRENT_DATA_VERSION.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

DATA_VERSION_PROPERTY: str = "pipeline.data_version"
# Silver without the property predates versioning.
DEFAULT_DATA_VERSION: str = "1"

INVALIDATED_REASON_014: str = "spec014_pt15m_parsing"
# Every model family ever trained on pre-014 silver, including the dropped 168h ones.
REMEDIATION_MODELS: tuple[str, ...] = (
    "energy_lgbm_24h",
    "energy_prophet_24h",
    "energy_lgbm_168h",
    "energy_prophet_168h",
)


def table_properties(rows: Iterable[Any]) -> dict[str, str]:
    """Maps `SHOW TBLPROPERTIES` rows (key, value) to a dict."""
    return {row["key"]: row["value"] for row in rows}


def read_data_version(properties: Mapping[str, str]) -> str:
    """The silver data version; absent means version 1."""
    return properties.get(DATA_VERSION_PROPERTY, DEFAULT_DATA_VERSION)


@dataclass(frozen=True)
class TagRemediation:
    """Tags one run gets, with the values it had before."""

    run_id: str
    model_name: str
    before: dict[str, str | None]
    after: dict[str, str]


def plan_data_version_remediation(
    runs: Iterable[Any], model_names: tuple[str, ...] = REMEDIATION_MODELS
) -> list[TagRemediation]:
    """Runs of `model_names` without a `data_version` tag, and the tags to set.

    Marks them `data_version=1`, `invalidated_reason=spec014_pt15m_parsing`,
    `production=false` (ADR-0003 §4). Runs that already carry a data_version
    were trained after the deploy and are left alone.
    """
    after = {
        "data_version": DEFAULT_DATA_VERSION,
        "invalidated_reason": INVALIDATED_REASON_014,
        "production": "false",
    }
    plan = []
    for run in runs:
        tags = run.data.tags
        model_name = tags.get("model_name")
        if model_name not in model_names or "data_version" in tags:
            continue
        plan.append(
            TagRemediation(
                run_id=run.info.run_id,
                model_name=model_name,
                before={key: tags.get(key) for key in after},
                after=dict(after),
            )
        )
    return plan
