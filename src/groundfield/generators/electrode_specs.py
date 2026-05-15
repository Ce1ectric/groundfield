"""Electrode specifications for generator-built worlds.

A :class:`GroundingSystemSpec` (see
:mod:`groundfield.generators.grounding`) is composed of a *list* of
electrode specifications drawn from this module. Each spec is a
Pydantic v2 model that defines the geometry of *one* electrode and
two auxiliary fields that the generator framework consumes:

* ``presence_prob`` — Bernoulli probability that this electrode is
  actually present in any given realisation. Useful for wiring up
  *fleets* (e.g. "70 % of houses have a foundation electrode plus a
  20 % chance of an additional driven rod for upgraded
  installations"). The probability itself may also be a
  :class:`Distribution` (so it can be drawn from e.g. a
  :class:`Uniform` over the network ensemble).

* ``offset_xy_m`` — translation in the horizontal plane relative to
  the *site centre* (substation centre, KVS centre, building
  centre, …). Lets the generator place several electrodes around
  the same site without having to special-case the geometry.

The classes are wired into a discriminated union :data:`ElectrodeSpec`
keyed on ``kind``. Together with the JSON-roundtrip machinery of
:mod:`groundfield.generators.distributions` this lets a complete
multi-electrode grounding system be persisted as a single JSON
document and replayed bit-exactly.

Convenience helper :func:`rod_circle` returns a list of
:class:`RodElectrodeSpec` instances arranged on a circle, which is
the typical layout for the *Tiefenerder* inside a transformer ring
earth electrode.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from groundfield.generators.distributions import AnyDistribution

FoundationStyle = Literal["ring", "mesh"]
"""How a :class:`FoundationElectrodeSpec` is realised.

- ``"ring"`` — only the rectangle's perimeter is buried (a closed
  loop of wire); no internal cross-bracing. Equivalent to a
  :class:`groundfield.geometry.electrodes.GridMeshElectrode` with
  ``n_x = n_y = 1`` (one mesh = perimeter wires only).
- ``"mesh"`` — the perimeter plus internal horizontal and vertical
  cross-braces, dividing the foundation into ``n_x × n_y`` cells.
  The classical *Maschenerder* layout.
"""

__all__ = [
    "RodElectrodeSpec",
    "RingElectrodeSpec",
    "StripElectrodeSpec",
    "FoundationElectrodeSpec",
    "ElectrodeSpec",
    "rod_circle",
]


# ---------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------


class _ElectrodeSpecBase(BaseModel):
    """Common Pydantic configuration for every electrode spec.

    Non-discriminator fields shared across all subclasses:

    * ``presence_prob`` — probability the electrode is actually placed
      in any one realisation (Bernoulli per realisation).
    * ``offset_xy_m`` — offset in the horizontal plane, relative to
      the site centre, in metres.
    """

    model_config = ConfigDict(extra="forbid")

    presence_prob: Union[float, AnyDistribution] = Field(
        default=1.0,
        description="Bernoulli probability that this electrode is present.",
    )
    offset_xy_m: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Offset (x, y) from the site centre in m.",
    )


# ---------------------------------------------------------------------
# Concrete electrode specs
# ---------------------------------------------------------------------


class RodElectrodeSpec(_ElectrodeSpecBase):
    """Driven rod (Tiefenerder).

    Modelled as a vertical line conductor of length ``length_m``
    starting at depth ``depth_m`` (rod head, 0 m = ground surface).
    """

    kind: Literal["rod"] = "rod"
    length_m: Union[float, AnyDistribution] = Field(default=1.5, description="Rod length in m.")
    depth_m: Union[float, AnyDistribution] = Field(default=0.0, description="Rod head depth in m (0 m = surface).")
    wire_radius_m: float = Field(default=0.008, description="Rod wire radius in m.")


class RingElectrodeSpec(_ElectrodeSpecBase):
    """Ring earth electrode.

    Horizontal circle of radius ``radius_m`` at depth ``depth_m``.
    The classical foundation-ring or substation-ring electrode.
    """

    kind: Literal["ring"] = "ring"
    radius_m: Union[float, AnyDistribution] = Field(default=4.0, description="Ring radius in m.")
    depth_m: Union[float, AnyDistribution] = Field(default=0.6, description="Ring depth in m.")
    wire_radius_m: float = Field(default=0.005, description="Ring wire radius in m.")


class StripElectrodeSpec(_ElectrodeSpecBase):
    """Horizontal strip earth electrode (Banderder).

    Straight buried wire of length ``length_m`` at depth ``depth_m``,
    rotated by ``orientation_deg`` around the vertical axis (0° = +x).
    """

    kind: Literal["strip"] = "strip"
    length_m: Union[float, AnyDistribution] = Field(default=20.0, description="Strip length in m.")
    depth_m: Union[float, AnyDistribution] = Field(default=0.6, description="Burial depth in m.")
    orientation_deg: Union[float, AnyDistribution] = Field(
        default=0.0,
        description="Rotation around the vertical axis (0° = +x).",
    )
    wire_radius_m: float = Field(default=0.005, description="Strip wire radius in m.")


class FoundationElectrodeSpec(_ElectrodeSpecBase):
    """Rectangular foundation electrode (Fundamenterder).

    A rectangle of side length ``size_m`` (or ``size_xy_m`` for an
    asymmetric footprint) at depth ``depth_m``. The internal
    structure is selected by :attr:`style`:

    * ``"ring"`` — only the rectangle's perimeter is buried (a
      closed wire loop). No internal cross-bracing. ``n_x`` and
      ``n_y`` are ignored. Use this for the *Ringerder*-style
      foundation electrode in residential buildings where only the
      strip foundation is electrically connected.
    * ``"mesh"`` (default) — perimeter plus horizontal and vertical
      cross-braces, dividing the foundation into ``n_x × n_y``
      cells. The *Maschenerder*-style foundation electrode found
      in larger buildings, transformer stations and industrial
      sites. With ``n_x = n_y = 2`` (the default) the rectangle
      gets exactly one internal horizontal plus one internal
      vertical wire.

    Both styles materialise as a
    :class:`groundfield.geometry.electrodes.GridMeshElectrode`;
    the ``"ring"`` style is just the special case ``n_x = n_y = 1``
    of that primitive (one mesh = perimeter only).

    For an asymmetric footprint set ``size_xy_m`` instead of
    ``size_m`` (when both are given, ``size_xy_m`` wins).
    """

    kind: Literal["foundation"] = "foundation"
    style: FoundationStyle = Field(
        default="mesh",
        description=(
            "'ring' — perimeter only (no inner bracing). 'mesh' — "
            "perimeter plus internal cross-braces (default)."
        ),
    )
    size_m: Union[float, AnyDistribution] = Field(default=10.0, description="Side length in m (square).")
    size_xy_m: Optional[tuple[float, float]] = Field(
        default=None,
        description="(dx, dy) — overrides ``size_m`` for rectangular footprints.",
    )
    depth_m: Union[float, AnyDistribution] = Field(default=0.8, description="Burial depth in m.")
    n_x: int = Field(
        default=2,
        ge=1,
        description=(
            "Inner meshes along x. Only honoured when style='mesh'; "
            "style='ring' forces n_x = 1."
        ),
    )
    n_y: int = Field(
        default=2,
        ge=1,
        description=(
            "Inner meshes along y. Only honoured when style='mesh'; "
            "style='ring' forces n_y = 1."
        ),
    )
    wire_radius_m: float = Field(default=0.005, description="Wire radius in m.")
    orientation_deg: Optional[float] = Field(
        default=None,
        description=(
            "Rotation of the foundation rectangle around its centre, "
            "in degrees. ``None`` (the default) and ``0.0`` both keep "
            "the historic axis-aligned realisation (one "
            ":class:`GridMeshElectrode` primitive). Any other value "
            "synthesises the foundation from rotated "
            ":class:`StripElectrode` primitives in "
            ":meth:`GroundingSystemSpec.build_at` and bonds them "
            "internally, so the spec still emits a single bondable "
            "anchor for the outer grounding cluster. Useful both for "
            "footprint-driven foundations (see ADR-0011, the OSM path "
            "sets this from the polygon OMBR) and for hand-placed "
            "houses on a street that does not run E-W."
        ),
    )
    # Concrete encasement (ADR-0012). See the ADR for the physical
    # model (cylindrical Sunde shell around each wire segment).
    concrete_rho_ohm_m: Union[float, AnyDistribution, None] = Field(
        default=None,
        description=(
            "Resistivity of the concrete encasement in Ω·m (typical "
            "ranges: 30-80 wet/fresh, 80-200 earth-moist, "
            "200-2 000 cycling, 5 000-50 000 dry). ``None`` (default) "
            "keeps the historic behaviour: wire sits directly in soil "
            "— right for ring/strip/rod electrodes which always run "
            "in trenches, wrong for foundation electrodes which sit "
            "inside a Streifenfundament (DIN 18014). Set this field "
            "to enable the concrete-shell model defined in ADR-0012."
        ),
    )
    concrete_thickness_m: Union[float, AnyDistribution] = Field(
        default=0.05,
        description=(
            "Radial thickness of the concrete shell around the wire, "
            "in metres. Only honoured when ``concrete_rho_ohm_m`` is "
            "not ``None``. Default 50 mm matches a typical 30 cm wide "
            "Streifenfundament with the conductor on its central "
            "axis; for the more conservative 'edge of the strip' "
            "placement, use 15 mm; for industrial pad foundations "
            "with the conductor on a reinforcement mat, 100-200 mm "
            "is appropriate."
        ),
    )
    concrete_model: Literal["lumped", "distributed"] = Field(
        default="lumped",
        description=(
            "How the Sunde-shell impedance enters the linear system. "
            "``\"lumped\"`` (default): total shell resistance is "
            "injected as a single series resistor on the PEN service "
            "drop. Zero solver-side risk; exact for the cluster "
            "impedance when current distributes uniformly along the "
            "foundation, which holds for AP1 frequencies. "
            "``\"distributed\"``: per-segment radial impedance is "
            "added to the MoM diagonal (ADR-0012 V2). More expensive "
            "but right for non-uniform current distributions or "
            "where the surface potential right at the building "
            "matters."
        ),
    )


# ---------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------


ElectrodeSpec = Annotated[
    Union[
        RodElectrodeSpec,
        RingElectrodeSpec,
        StripElectrodeSpec,
        FoundationElectrodeSpec,
    ],
    Field(discriminator="kind"),
]
"""JSON-serialisable union of electrode specs.

Use this alias as the field type in any :class:`GeneratorConfig`
that holds a list of electrodes (e.g. ``electrodes:
list[ElectrodeSpec]``). Pydantic dispatches on ``kind``.
"""


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------


def rod_circle(
    n: int,
    radius_m: float,
    *,
    length_m: Union[float, AnyDistribution] = 1.5,
    depth_m: Union[float, AnyDistribution] = 0.0,
    wire_radius_m: float = 0.008,
    presence_prob: Union[float, AnyDistribution] = 1.0,
    angle_offset_deg: float = 0.0,
) -> list[RodElectrodeSpec]:
    """Build *n* :class:`RodElectrodeSpec` arranged on a circle.

    Typical use case: the four (or eight) driven rods inside the
    substation ring earth electrode. Each rod's ``offset_xy_m`` is
    set so the rods sit at angles
    $\\theta_k = \\text{angle\\_offset} + 2\\pi k / n$ on a circle
    of radius ``radius_m`` centred on the site.

    Parameters
    ----------
    n
        Number of rods.
    radius_m
        Circle radius in m.
    length_m, depth_m, wire_radius_m, presence_prob
        Forwarded to every :class:`RodElectrodeSpec`. Each rod
        carries the same value (a single :class:`Distribution`
        passed in is shared and *resampled per rod* by the build
        machinery — pass independent copies if you want
        independent draws).
    angle_offset_deg
        Phase offset in degrees so the first rod can be rotated
        from the +x axis.

    Returns
    -------
    list[RodElectrodeSpec]
        Length ``n``. Drop straight into a
        :class:`GroundingSystemSpec.electrodes` list.
    """
    if n < 1:
        raise ValueError(f"rod_circle: n must be >= 1, got {n}.")
    if radius_m < 0:
        raise ValueError(f"rod_circle: radius_m must be >= 0, got {radius_m}.")
    out: list[RodElectrodeSpec] = []
    for k in range(n):
        angle = math.radians(angle_offset_deg) + 2.0 * math.pi * k / n
        offset = (radius_m * math.cos(angle), radius_m * math.sin(angle))
        out.append(
            RodElectrodeSpec(
                length_m=length_m,
                depth_m=depth_m,
                wire_radius_m=wire_radius_m,
                presence_prob=presence_prob,
                offset_xy_m=offset,
            )
        )
    return out
