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
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Optional, Self, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from groundfield.generators.distributions import AnyDistribution
from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

FoundationStyle = Literal["ring", "mesh"]
"""How a :class:`FoundationElectrodeSpec` is realised.

- ``"ring"`` — only the rectangle's perimeter is buried (a closed
  loop of wire); no internal cross-bracing. The geometry of a
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

# User decision Q21 (2026.11). The numbers are the envelope of the largest
# relative difference of the input impedance between the two values over the
# scope they name (audit D2B-02, evidence of stream f2gen:
# q21_envelope_scan.py over 37 rectangles, 5 wire radii, 39 soils and
# rho_c = 30 ... 1000 ohm m; q21_envelope_verify.py checks the largest cases
# with Engine.solve and an independent BEM): 0.691 % (4 m x 1 m, t = 0.2 m,
# rho_c = 30 ohm m, 1000/30 ohm m, h_1 = 5 m) and 1.529 % (20 m x 1 m, the
# same shell, h_1 = 10 m); 0.321 % (80 m x 20 m) and 0.914 % (150 m x 7.5 m)
# bare or with the default 5 cm shell (30/200 ohm m, h_1 = 5 m).
_CONCRETE_MODEL_DEPRECATION = (
    "FoundationElectrodeSpec.concrete_model is deprecated. It no longer "
    "selects a concrete-shell model (both values put the same shell on the "
    "wires); it only chooses the uniform-current owners of an axis-aligned "
    "rectangle without cross-braces (style='ring', or style='mesh' with "
    "n_x = n_y = 1): one per side for 'distributed', one for the whole "
    "rectangle for 'lumped' (the default). Over the soils of the AP1 grid, "
    "for a 5 mm wire at 0.8 m, bare or in concrete of 30 to 1000 ohm m up to "
    "0.2 m thick, the two differ by less than 0.7 % up to aspect ratio 4 "
    "(sides up to 80 m) and 1.6 % up to aspect ratio 50 (up to 150 m, short "
    "side at least 1 m); bare or with the default 5 cm shell by less than "
    "0.33 % and 0.92 %. The field will be removed no earlier than groundfield "
    f"{DEPRECATION_REMOVAL}; drop the argument."
)


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
    length_m: Union[float, AnyDistribution] = Field(
        default=1.5, description="Rod length in m."
    )
    depth_m: Union[float, AnyDistribution] = Field(
        default=0.0, description="Rod head depth in m (0 m = surface)."
    )
    wire_radius_m: float = Field(default=0.008, description="Rod wire radius in m.")


class RingElectrodeSpec(_ElectrodeSpecBase):
    """Ring earth electrode.

    Horizontal circle of radius ``radius_m`` at depth ``depth_m``.
    The classical foundation-ring or substation-ring electrode.
    """

    kind: Literal["ring"] = "ring"
    radius_m: Union[float, AnyDistribution] = Field(
        default=4.0, description="Ring radius in m."
    )
    depth_m: Union[float, AnyDistribution] = Field(
        default=0.6, description="Ring depth in m."
    )
    wire_radius_m: float = Field(default=0.005, description="Ring wire radius in m.")


class StripElectrodeSpec(_ElectrodeSpecBase):
    """Horizontal strip earth electrode (Banderder).

    Straight buried wire of length ``length_m`` at depth ``depth_m``,
    rotated by ``orientation_deg`` around the vertical axis (0° = +x).
    """

    kind: Literal["strip"] = "strip"
    length_m: Union[float, AnyDistribution] = Field(
        default=20.0, description="Strip length in m."
    )
    depth_m: Union[float, AnyDistribution] = Field(
        default=0.6, description="Burial depth in m."
    )
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

    The image family solves one uniform leakage current per electrode
    (an *owner*); the owners of a foundation depend only on its
    structure (and, for an axis-aligned ring, on the deprecated
    :attr:`concrete_model`), never on the orientation, on how the outline
    is described or on whether a shell is configured (audit A2G-04). A
    ``"mesh"`` foundation is one straight wire per mesh line (a polyline
    split at the crossings), bonded to the first: one owner over
    perimeter and braces was 2.4 % (2 × 2) to 3.7 % (4 × 4) high against
    ``mom`` (one current per segment) with the default line kernel, one
    owner per wire is 1.1 % high (0.3–0.5 % with the point kernel, the
    kernel of ``mom``). A ``"ring"`` foundation is one
    :class:`groundfield.geometry.electrodes.GridMeshElectrode` with
    ``n_x = n_y = 1`` (one owner) when axis-aligned and one straight wire
    per side when rotated, and so is a ``"mesh"`` with ``n_x = n_y = 1``
    (a rectangle without cross-braces); ``concrete_model="distributed"``
    (deprecated) makes the axis-aligned rectangle one wire per side as
    well, less than 0.7 % apart up to aspect ratio 4 and 1.6 % up to 50
    (see :attr:`concrete_model` for the scope). A ``polygon_xy_m`` outline
    is one closed polyline.

    With a shell (:attr:`concrete_rho_ohm_m`) every wire of the
    foundation, the cross-braces of a mesh included, is encased (DIN
    18014, user decision Q20); see ADR-0012.

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
    size_m: Union[float, AnyDistribution] = Field(
        default=10.0, description="Side length in m (square)."
    )
    size_xy_m: Optional[tuple[float, float]] = Field(
        default=None,
        description="(dx, dy) — overrides ``size_m`` for rectangular footprints.",
    )
    polygon_xy_m: Optional[tuple[tuple[float, float], ...]] = Field(
        default=None,
        description=(
            "Closed outline of the foundation as (x, y) vertices in "
            "metres, **relative to the site centre** passed to "
            ":meth:`GroundingSystemSpec.build_at` (do not repeat the "
            "first vertex). When set it overrides ``size_m`` / "
            "``size_xy_m`` / ``orientation_deg``: the foundation "
            "electrode is materialised as a single closed "
            ":class:`~groundfield.geometry.electrodes.PolylineElectrode` "
            "following the real building outline (DIN 18014: the strip "
            "foundation runs along the building perimeter), instead of "
            "the oriented bounding rectangle. Only meaningful with "
            "``style='ring'`` — the internal cross-braces of "
            "``style='mesh'`` are undefined for a general polygon and "
            "are rejected. Rationale: the OMBR overestimates the "
            "perimeter of L-/U-shaped buildings and can make the "
            "rectangles of neighbouring buildings coincide or overlap, "
            "which drives distinct segments into the 1 mm singularity "
            "clamp of the solver."
        ),
    )
    depth_m: Union[float, AnyDistribution] = Field(
        default=0.8, description="Burial depth in m."
    )
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
            "in degrees. ``None`` (the default) and ``0.0`` keep it "
            "axis-aligned; a multiple of 90 degrees is the axis-aligned "
            "rectangle with swapped sides. The orientation changes only "
            "the geometry, never the owners (see the class docstring): a "
            "foundation built from several wires bonds them internally, "
            "so the spec still emits a single bondable anchor (the first "
            "wire) for the outer grounding cluster. Useful both for "
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
            "to enable the concrete-shell model defined in ADR-0012: "
            "every wire of the foundation, the cross-braces of a "
            "``style='mesh'`` foundation included, gets the outer radius "
            "r_b = r_a + t and the shell coefficient "
            "C = rho_c / (2 pi) ln(r_b / r_a) (user decision Q20 of "
            "2026.11). In 0.15.0 every wire got r_b, but C only the "
            "perimeter wires and only with ``concrete_model='distributed'`` "
            "(``'lumped'`` kept the shell in a registry that only the TN "
            "generator's service drops read); the bare braces (r_a, no C) "
            "of the 2026.11 development line before that decision were "
            "never released."
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
            "**Deprecated** since 2026.11 (user decision Q21; removal no "
            'earlier than 2027.5): drop the argument. ``"distributed"`` '
            "still works and emits a DeprecationWarning; the default "
            '``"lumped"`` is accepted silently, also from the JSON and '
            "pickles of 0.15.0. The field no longer selects a shell model: "
            "both values put the Sunde-shell coefficient "
            "C = rho_c / (2 pi) ln(r_b / r_a) (Ohm m) on every wire of an "
            "encased foundation, cross-braces included, whose radius "
            "becomes r_b = r_a + t; the solver adds C / ds to the diagonal "
            "of each segment, i.e. the lumped shell resistance C / L in "
            "series with the wire. It only chooses the uniform-current "
            "owners of an axis-aligned rectangle without cross-braces "
            "(``style='ring'``, or ``style='mesh'`` with ``n_x = n_y = 1``): "
            '``"lumped"`` one grid-mesh owner for the whole rectangle, '
            '``"distributed"`` one owner per side. Over the soils of the '
            "AP1 grid, for a 5 mm wire at 0.8 m, bare or in concrete of 30 "
            "to 1000 Ω·m up to 0.2 m thick, the two differ by less than "
            "0.7 % up to aspect ratio 4 (sides up to 80 m) and 1.6 % up to "
            "aspect ratio 50 (up to 150 m, short side at least 1 m); bare or "
            "with the default 5 cm shell by less than 0.33 % and 0.92 % "
            "(audit D2B-02). Rotated rectangles (one owner per side), "
            "meshes with cross-braces (one owner per wire) and polygon "
            "outlines (one closed wire) are the same for both values (audit "
            "A2G-04). Up to 0.15.0 the lumped shell was a registry entry "
            "that only the TN generator's PEN service drops read (review "
            "findings 10.12-10.14)."
        ),
    )

    @field_validator("concrete_model")
    @classmethod
    def _warn_deprecated_concrete_model(cls, value: str) -> str:
        """Warn for the non-default value (user decision Q21).

        Runs on construction, ``model_validate`` and JSON input; the
        default ``"lumped"`` (also in files written by 0.15.0) is silent.
        Pickles are not validated, and copies only warn when their
        ``update`` sets the value (:meth:`model_copy`), so the generators,
        which pass resolved specs on as copies, never repeat the warning.
        """
        if value != "lumped":
            warn_external(_CONCRETE_MODEL_DEPRECATION, DeprecationWarning)
        return value

    def model_copy(
        self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False
    ) -> Self:
        """Copy of the spec; validates an ``update`` of ``concrete_model``.

        Pydantic stores ``update`` without validation, so the deprecated
        :attr:`concrete_model` (user decision Q21) is validated here like
        in the constructor: a value outside ``{"lumped", "distributed"}``
        raises the constructor's :class:`pydantic.ValidationError` (no
        warning, nothing stored; audit D2B-04), ``"distributed"`` emits
        the constructor's :class:`DeprecationWarning` at the caller's line
        and ``"lumped"`` is silent. Only that field is validated (on a
        shallow copy), so the other fields, also the required fields of a
        subclass, are copied as they are (audit R2B-05).
        ``copy.replace`` (Python >= 3.13) calls this method. Copies that
        leave the field alone (every copy the generators make) are silent.

        Parameters
        ----------
        update
            Field values to change. ``concrete_model`` is validated; the
            other fields are not (as in pydantic).
        deep
            Deep copy instead of a shallow one.

        Returns
        -------
        FoundationElectrodeSpec
            The copy (the receiver is not changed).

        Raises
        ------
        pydantic.ValidationError
            If ``update`` sets ``concrete_model`` to a value other than
            ``"lumped"`` or ``"distributed"``.
        """
        if update and "concrete_model" in update:
            # Only the field is validated, on a shallow copy, so the other
            # fields of a subclass need not be rebuilt (audit R2B-05): the
            # Literal check raises, the field validator warns for the
            # deprecated value.
            probe = super().model_copy()
            type(self).__pydantic_validator__.validate_assignment(
                probe, "concrete_model", update["concrete_model"]
            )
            update = {**update, "concrete_model": probe.concrete_model}
        return super().model_copy(update=update, deep=deep)

    @model_validator(mode="after")
    def _check_polygon(self) -> "FoundationElectrodeSpec":
        if self.polygon_xy_m is None:
            return self
        if len(self.polygon_xy_m) < 3:
            raise ValueError(
                f"polygon_xy_m needs at least 3 vertices; got {len(self.polygon_xy_m)}."
            )
        if self.style != "ring":
            raise ValueError(
                "polygon_xy_m is only defined for style='ring' — the "
                "internal cross-braces of style='mesh' have no meaning "
                "on a general polygon."
            )
        return self


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
