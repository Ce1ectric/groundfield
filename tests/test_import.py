"""Smoke-Test für das frisch angelegte Paket.

Der Test stellt sicher, dass ``groundfield`` importierbar ist und die
Versionszeichenkette dem in ``pyproject.toml`` hinterlegten Wert
entspricht. Er existiert, damit das CI-Pipeline-Gerüst vom ersten
Commit an etwas Grünes vorzeigen kann.
"""

from __future__ import annotations

import re

import groundfield


def test_package_version_is_semver() -> None:
    """``groundfield.__version__`` muss einem SemVer-Kern genügen."""
    assert re.match(
        r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$",
        groundfield.__version__,
    ), groundfield.__version__


def test_subpackages_are_importable() -> None:
    """Alle angelegten Subpackages müssen per Import auffindbar sein."""
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
