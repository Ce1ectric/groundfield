"""Smoke test for the freshly created package.

This test ensures that ``groundfield`` is importable and that the
version string is a valid PEP 440 release (calendar versioning
``YYYY.MM.MICRO`` since 2026.11, SemVer before).
It exists so the CI pipeline scaffold has something green to show
from the first commit.
"""

from __future__ import annotations

import re

import groundfield


def test_package_version_is_pep440() -> None:
    """``groundfield.__version__`` is a PEP 440 version with three release
    components, optionally a pre-release (``2026.11.0``, ``2026.11.0rc1``)."""
    assert re.match(
        r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$",
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
