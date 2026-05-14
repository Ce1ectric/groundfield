"""Placement strategies for site positioning on a 2-D map.

A :class:`PlacementSpec` decides where the buildings of a generator
run end up on the horizontal plane. Two strategies are supported in
v1; further strategies (random scatter with Poisson-disk sampling,
sub-cluster placement, OSM ingest) are deferred to follow-up
generators.

* :class:`ManhattanGridPlacement` — regular street-raster layout:
  ``n_per_row`` columns, $\\lceil n / n_\\text{per\\_row}
  \\rceil$ rows centred on a configurable point. Optional
  ``jitter_m`` adds a uniform offset per site so the grid is not
  pixel-perfect.

* :class:`ExplicitPlacement` — caller-supplied list of $(x, y)$
  positions. Useful for replaying a real map slice or for hand-
  drawn small reference cases. The number of positions must match
  or exceed the requested count.

Both classes implement :meth:`generate(n, rng) -> list[(x, y)]`.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from groundfield.generators.distributions import AnyDistribution, Distribution

__all__ = [
    "ManhattanGridPlacement",
    "ExplicitPlacement",
    "PlacementSpec",
]


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    """Resolve a ``float | Distribution`` field to a float."""
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


# ---------------------------------------------------------------------
# Manhattan grid
# ---------------------------------------------------------------------


class ManhattanGridPlacement(BaseModel):
    """Regular Manhattan-grid placement.

    Houses are placed on a grid centred on ``centre_xy`` with
    ``n_per_row`` columns and as many rows as needed to fit ``n``
    sites (the last row may be partially filled).

    Spacings are configurable as fixed values or distributions; they
    are sampled once per :meth:`generate` call (one draw used for
    the whole grid). Per-site ``jitter_m`` adds an additional
    uniform offset in the box
    $[-\\text{jitter}/2, +\\text{jitter}/2]^2$ to each grid point —
    sample once per site.

    Notes
    -----
    The grid is centred so the geometric centroid of a fully filled
    grid coincides with ``centre_xy``. For a partial last row the
    centroid drifts slightly; if you need exact symmetry, request
    a count that is a multiple of ``n_per_row``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["manhattan"] = "manhattan"
    spacing_x_m: Union[float, AnyDistribution] = Field(
        default=25.0,
        description="Column-to-column spacing in m.",
    )
    spacing_y_m: Union[float, AnyDistribution] = Field(
        default=30.0,
        description="Row-to-row spacing in m.",
    )
    n_per_row: int = Field(default=10, ge=1, description="Columns per row.")
    centre_xy: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Centre of the grid in m.",
    )
    jitter_m: Union[float, AnyDistribution] = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-site uniform jitter amplitude in m (full width of the "
            "box). 0 disables."
        ),
    )

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        if n < 1:
            return []
        sx = _to_float(self.spacing_x_m, rng)
        sy = _to_float(self.spacing_y_m, rng)
        n_rows = math.ceil(n / self.n_per_row)
        cx, cy = self.centre_xy
        x_offset = cx - (self.n_per_row - 1) / 2.0 * sx
        y_offset = cy - (n_rows - 1) / 2.0 * sy
        positions: list[tuple[float, float]] = []
        for i in range(n):
            row = i // self.n_per_row
            col = i % self.n_per_row
            x = col * sx + x_offset
            y = row * sy + y_offset
            j = _to_float(self.jitter_m, rng)
            if j > 0.0:
                x += float(rng.uniform(-j / 2.0, j / 2.0))
                y += float(rng.uniform(-j / 2.0, j / 2.0))
            positions.append((x, y))
        return positions


# ---------------------------------------------------------------------
# Explicit positions
# ---------------------------------------------------------------------


class ExplicitPlacement(BaseModel):
    """Explicit caller-supplied list of $(x, y)$ positions.

    The list must be at least as long as the requested count; extras
    are ignored. Order is preserved (so the *k*-th building from
    the count list ends up at the *k*-th position).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["explicit"] = "explicit"
    positions: list[tuple[float, float]] = Field(default_factory=list)

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        if n > len(self.positions):
            raise ValueError(
                f"ExplicitPlacement: requested {n} positions but only "
                f"{len(self.positions)} provided."
            )
        return list(self.positions[:n])


# ---------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------


PlacementSpec = Annotated[
    Union[ManhattanGridPlacement, ExplicitPlacement],
    Field(discriminator="kind"),
]
"""JSON-serialisable union of placement strategies."""
