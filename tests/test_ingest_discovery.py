"""Unit tests for src.ingest_discovery.select_pending_files (spec 008 AC1)."""

from src.ingest_discovery import select_pending_files

P = "/Volumes/workspace/energy_forecasting/data/raw_ingestion/load"


def _paths(*hours: str) -> list[str]:
    return [f"{P}/{h}.json" for h in hours]


def test_backlog_smaller_than_cap_selects_all_with_none_remaining():
    available = _paths("2026-09-19T14-00-00Z", "2026-09-19T15-00-00Z")

    selected, remaining = select_pending_files(available, 10)

    assert selected == available
    assert remaining == 0


def test_backlog_larger_than_cap_selects_oldest_first_and_reports_remaining():
    available = _paths(
        "2026-09-19T14-00-00Z",
        "2026-09-19T15-00-00Z",
        "2026-09-19T16-00-00Z",
        "2026-09-19T17-00-00Z",
        "2026-09-19T18-00-00Z",
    )

    selected, remaining = select_pending_files(available, 2)

    assert selected == _paths("2026-09-19T14-00-00Z", "2026-09-19T15-00-00Z")
    assert remaining == 3


def test_unsorted_input_is_normalized_to_chronological_order():
    available = _paths(
        "2026-09-20T02-00-00Z",
        "2026-09-19T22-00-00Z",
        "2026-09-20T00-00-00Z",
        "2026-09-19T23-00-00Z",
        "2026-09-20T01-00-00Z",
    )

    selected, remaining = select_pending_files(available, 3)

    assert selected == _paths(
        "2026-09-19T22-00-00Z",
        "2026-09-19T23-00-00Z",
        "2026-09-20T00-00-00Z",
    )
    assert remaining == 2


def test_empty_input_returns_empty_selection_and_no_backlog():
    selected, remaining = select_pending_files([], 24)

    assert selected == []
    assert remaining == 0


def test_exact_cap_boundary_selects_all_with_none_remaining():
    available = _paths(
        "2026-09-19T14-00-00Z",
        "2026-09-19T15-00-00Z",
        "2026-09-19T16-00-00Z",
        "2026-09-19T17-00-00Z",
    )

    selected, remaining = select_pending_files(available, 4)

    assert selected == available
    assert remaining == 0
