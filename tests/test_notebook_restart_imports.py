"""Names used after `dbutils.library.restartPython()` must be defined after it.

The restart clears every name from earlier cells, but ruff sees one file and
cannot tell. 09_error_monitor shipped using `Path` from the pre-restart cell
and failed on its first live run (2026-11-01).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.py"))
DATABRICKS_GLOBALS = {"spark", "dbutils", "display", "displayHTML", "sc"}


def _after_restart(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "dbutils.library.restartPython()" in line:
            return "\n".join(lines[i + 1 :])
    return None


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_no_name_from_before_the_restart_is_used_after_it(notebook):
    code = _after_restart(notebook)
    if code is None:
        pytest.skip("no restartPython()")

    result = subprocess.run(
        [
            "ruff",
            "check",
            "--select",
            "F821",
            "--output-format",
            "json",
            "--stdin-filename",
            "nb.py",
            "-",
        ],
        input=code,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    undefined = {
        d["message"].split("`")[1] for d in json.loads(result.stdout or "[]") if d["code"] == "F821"
    }

    assert undefined - DATABRICKS_GLOBALS == set()
