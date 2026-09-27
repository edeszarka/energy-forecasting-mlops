"""Pure selection logic for the ingestion Volume backlog-discovery pattern."""

from __future__ import annotations


def select_pending_files(
    available_paths: list[str],
    max_files_per_run: int,
) -> tuple[list[str], int]:
    """Given every filename currently sitting in a raw-ingestion Volume directory
    (already-archived files were moved out by Cell 11's archival step in earlier
    runs, so whatever remains here IS the backlog), returns
    (files_to_process_this_run, backlog_remaining_after_this_run).

    Files are sorted ascending — the "%Y-%m-%dT%H-00-00Z.json" naming convention
    makes lexicographic sort equivalent to chronological order — and capped to
    max_files_per_run, oldest first, so a run never exceeds its processing
    budget regardless of how large the backlog is. Anything beyond the cap is
    left for the next run; because MERGE INTO in this pipeline is idempotent,
    draining a backlog across several runs is safe and self-healing.
    """
    ordered = sorted(available_paths)
    selected = ordered[:max_files_per_run]
    backlog_remaining = max(0, len(ordered) - max_files_per_run)
    return selected, backlog_remaining
