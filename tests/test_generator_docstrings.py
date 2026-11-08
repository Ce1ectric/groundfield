"""Source hygiene of the generator and geo packages.

A LaTeX command such as ``\\rho`` or ``\\text`` in a non-raw docstring is
a valid Python escape (``\\r`` is a carriage return, ``\\t`` a tab): the
module compiles without a warning, but the rendered formula is silently
corrupted. Other backslash sequences (``\\D``) are invalid escapes and
raise ``SyntaxWarning`` (a ``SyntaxError`` under ``python -W error``).
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

import groundfield

_PACKAGES = ("generators", "geo")
_CONTROL = re.compile(r"[\t\r\x07\x08\x0b\x0c]")


def _sources() -> Iterator[Path]:
    root = Path(groundfield.__file__).resolve().parent
    for package in _PACKAGES:
        yield from sorted((root / package).rglob("*.py"))


@pytest.mark.finding("10.95")
def test_generator_modules_compile_without_syntax_warnings() -> None:
    offenders = []
    for path in _sources():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        offenders += [f"{path.name}:{w.lineno}: {w.message}" for w in caught]
    assert not offenders


@pytest.mark.finding("10.95")
def test_generator_docstrings_contain_no_control_characters() -> None:
    """``generators/soil_specs.py`` had ``:math:`\\rho``` in a non-raw
    module docstring, i.e. a carriage return followed by ``ho``."""
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
            ):
                doc = ast.get_docstring(node, clean=False) or ""
                if _CONTROL.search(doc):
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', 1)}")
    assert not offenders
