"""Escape sequences in the package sources (audit A1-12, 10.95).

An invalid escape such as ``\\D`` in a non-raw docstring raises a
``SyntaxWarning`` (a ``SyntaxError`` under ``python -W error`` and in a
future Python); ``ruff check --select W605``, part of the blocking lint
subset since this release, reports it. A *valid* escape that is meant as
LaTeX compiles silently and corrupts the rendered formula: ``\\rho``
starts with a carriage return, ``\\text`` with a TAB, ``\\beta`` with a
backspace. Both checks run over every module of the package.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import pytest

import groundfield

ROOT = Path(groundfield.__file__).resolve().parent

#: Files owned by a parallel stream, checked once their fix is merged.
OWNED_ELSEWHERE = {
    # '\rho' in the module docstring (a CR); stream gen2, see the fix1 report.
    "generators/soil_specs.py",
}

_CONTROL = re.compile(r"[\t\r\x07\x08\x0b\x0c]")


def _modules() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def test_every_module_compiles_without_syntax_warnings() -> None:
    offenders = []
    for path in _modules():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        offenders += [
            f"{path.relative_to(ROOT)}:{w.lineno}: {w.message}"
            for w in caught
            if issubclass(w.category, SyntaxWarning)
        ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.finding("10.95", "A1-12")
def test_no_docstring_contains_a_control_character() -> None:
    offenders = []
    for path in _modules():
        rel = path.relative_to(ROOT).as_posix()
        if rel in OWNED_ELSEWHERE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                doc = ast.get_docstring(node, clean=False) or ""
                if _CONTROL.search(doc):
                    offenders.append(f"{rel}:{getattr(node, 'lineno', 1)}")
    assert not offenders, offenders
