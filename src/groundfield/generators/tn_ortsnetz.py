"""Deprecated module path — kept as an alias for the renamed module.

The TN low-voltage network generator was renamed from
``TnOrtsnetzGenerator`` to :class:`TnNetworkGenerator` (and from
``TnOrtsnetzConfig`` to :class:`TnNetworkConfig`) to align with the
project's English-only naming convention. The generator was
subsequently extended with a richer spec layer (multi-electrode
grounding systems, building-type catalog, pluggable placement and
soil specs) — *that change broke the previous flat config schema*.

Importing from this module emits a :class:`DeprecationWarning` and
re-exports the new names. Update your imports to
:mod:`groundfield.generators.tn_network` (or pick the symbols up
from :mod:`groundfield.generators` directly).
"""

from __future__ import annotations

import warnings

from groundfield.generators.tn_network import (
    KvsConfig,
    PenConfig,
    SubstationConfig,
    TnNetworkConfig as TnOrtsnetzConfig,
    TnNetworkGenerator as TnOrtsnetzGenerator,
)

warnings.warn(
    "groundfield.generators.tn_ortsnetz is deprecated; import "
    "TnNetworkGenerator / TnNetworkConfig from groundfield.generators "
    "(or .tn_network).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "KvsConfig",
    "PenConfig",
    "SubstationConfig",
    "TnOrtsnetzConfig",
    "TnOrtsnetzGenerator",
]
