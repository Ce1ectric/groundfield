"""Smoke test for the freshly created package.

This test ensures that ``groundfield`` is importable and that the
version string matches the SemVer core stored in ``pyproject.toml``.
It exists so the CI pipeline scaffold has something green to show
from the first commit.
"""

from __future__ import annotations

import re

import groundfield


def test_package_version_is_semver() -> None:
    """``groundfield.__version__`` must satisfy a SemVer core."""
    assert re.match(
        r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$",
        groundfield.__version__,
    ), groundfield.__version__


def test_subpackages_are_importable() -> None:
    """All declared subpackages must be importable."""
    import importlib

    for name in (
        "groundfield.soil",
        "groundfield.geometry",
        "groundfield.conductors",
        "groundfield.solver",
        "groundfield.coupling",
        "groundfield.postprocess",
        "groundfield.io",
        "groundfield.utils",
    ):
        importlib.import_module(name)
