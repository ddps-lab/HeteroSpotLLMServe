"""
Figure-generation smoke test.

Walks every ``ArtifactEvaluation/ReferenceData/**/figures/`` directory,
executes each notebook (.ipynb) and script (.py) in place, and asserts that
the directory contains at least one new PDF or PNG afterward.

Usage
-----
    pytest tests/test_figures.py                   # run all
    pytest tests/test_figures.py -k interruption   # filter by name
    USE_SAMPLE=1 pytest tests/test_figures.py      # use sample data if notebook supports it

Skip behavior
-------------
A notebook is SKIPPED (not failed) if either:

* its first markdown cell starts with ``SKIP:`` (hand-annotated reason), or
* it raises ``FileNotFoundError`` for a raw data path that is not present
  in the repo (typical for notebooks whose inputs are generated on GPU nodes).

Anything else — import errors, unexpected paths written outside the current
``figures/`` dir, empty output — is a failure.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = REPO_ROOT / "ArtifactEvaluation" / "ReferenceData"

FIGURE_EXTS = {".pdf", ".png", ".svg"}


def _discover() -> list[Path]:
    items: list[Path] = []
    for figures_dir in sorted(REFERENCE_DATA.rglob("figures")):
        if not figures_dir.is_dir():
            continue
        for p in sorted(figures_dir.iterdir()):
            if p.suffix in {".ipynb", ".py"} and not p.name.startswith("_"):
                items.append(p)
    return items


def _read_skip_reason(nb_path: Path) -> str | None:
    if nb_path.suffix != ".ipynb":
        return None
    try:
        nb = json.loads(nb_path.read_text())
    except Exception:
        return None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if src.lstrip().upper().startswith("SKIP:"):
                return src.strip().splitlines()[0]
            break
    return None


def _snapshot_figures(dir_path: Path) -> dict[Path, float]:
    return {
        p: p.stat().st_mtime
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in FIGURE_EXTS
    }


def _new_or_updated_figures(before: dict[Path, float], dir_path: Path) -> list[Path]:
    out: list[Path] = []
    for p in dir_path.iterdir():
        if not p.is_file() or p.suffix.lower() not in FIGURE_EXTS:
            continue
        if p not in before or p.stat().st_mtime > before[p]:
            out.append(p)
    return out


def _find_paths_written_outside(dir_path: Path, before_all: set[Path]) -> list[Path]:
    """Return figure files that appeared under REFERENCE_DATA outside dir_path."""
    strays: list[Path] = []
    for p in REFERENCE_DATA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in FIGURE_EXTS:
            continue
        if p in before_all:
            continue
        if dir_path in p.parents:
            continue
        strays.append(p)
    return strays


def _execute(path: Path, cwd: Path) -> subprocess.CompletedProcess:
    cell_timeout = int(os.environ.get("FIGURE_CELL_TIMEOUT", "120"))
    if path.suffix == ".ipynb":
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={cell_timeout}",
            str(path.name),
        ]
    else:
        cmd = [sys.executable, str(path.name)]
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
        env=os.environ.copy(), timeout=cell_timeout * 4,
    )


@pytest.mark.parametrize("script", _discover(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_figure_script(script: Path, tmp_path: Path) -> None:
    reason = _read_skip_reason(script)
    if reason:
        pytest.skip(reason)

    figures_dir = script.parent
    before = _snapshot_figures(figures_dir)
    before_all = {p for p in REFERENCE_DATA.rglob("*") if p.is_file() and p.suffix.lower() in FIGURE_EXTS}

    result = _execute(script, figures_dir)

    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-2000:]
        if "FileNotFoundError" in tail or "No such file" in tail:
            pytest.skip(f"Input data missing for {script.name}\n{tail[-800:]}")
        pytest.fail(
            f"Execution failed ({script.name}, rc={result.returncode})\n\n"
            f"--- stdout tail ---\n{result.stdout[-1000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-1000:]}"
        )

    produced = _new_or_updated_figures(before, figures_dir)
    assert produced, (
        f"{script.name} ran successfully but produced no .pdf/.png/.svg under "
        f"{figures_dir.relative_to(REPO_ROOT)}"
    )

    strays = _find_paths_written_outside(figures_dir, before_all)
    assert not strays, (
        f"{script.name} wrote figures outside its own figures/ directory: "
        + ", ".join(str(s.relative_to(REPO_ROOT)) for s in strays)
    )
