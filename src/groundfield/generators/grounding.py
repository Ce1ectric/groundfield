"""Composable grounding-system specifications.

A :class:`GroundingSystemSpec` represents *one* electrically
connected grounding installation: a (possibly heterogeneous) set of
:class:`~groundfield.generators.electrode_specs.ElectrodeSpec`
instances that together form a single galvanic cluster. The class
is shared between

* the substation grounding (typically ring + multiple rods, optionally
  a strip and/or a foundation electrode),
* every cable cabinet (KVS) grounding (typically a single rod, but
  any combination is allowed),
* the per-house / per-building grounding (foundation, rod, mesh, or
  any AND-combination).

The materialisation method :meth:`GroundingSystemSpec.build_at`
takes a :class:`World`, a centre position ``(x, y)``, and a name
prefix; for each electrode in the list it draws a Bernoulli on
``presence_prob``, and if present, samples the geometry parameters
and registers the corresponding electrode with the world. All
created electrodes are then bonded together with bare-copper
conductors so that the world treats them as one cluster.

The first present electrode is the *anchor* — the
:meth:`build_at` method returns its name so the caller can wire
PEN cables, current sources, etc. into the cluster.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from pydantic import Field

from groundfield.api import create_conductor, create_electrode
from groundfield.generators.base import GeneratorConfig
from groundfield.generators.distributions import Distribution
from groundfield.generators.electrode_specs import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
)
from groundfield.world import World

__all__ = [
    "GroundingSystemSpec",
]


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    """Resolve a ``float | Distribution`` to ``float`` (sample once)."""
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


def _maybe_concrete_shell(
    spec: FoundationElectrodeSpec,
    *,
    dx: float,
    dy: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    r"""Compute the effective wire radius, lumped shell resistance, and
    per-segment shell coefficient for ADR-0012.

    Parameters
    ----------
    spec
        Foundation electrode spec; ``concrete_rho_ohm_m`` decides
        whether the shell model is active.
    dx, dy
        Side lengths of the foundation rectangle in metres (already
        resolved from the spec or from an OSM footprint).
    rng
        RNG used to sample ``concrete_rho_ohm_m`` and
        ``concrete_thickness_m`` when they are ``Distribution``.

    Returns
    -------
    effective_radius_m : float
        Outer radius of the concrete shell ($r_b = r_a + t$) when
        a shell is configured, otherwise the bare wire radius
        ``spec.wire_radius_m``.
    shell_total_ohm : float
        Total lumped Sunde-shell resistance in Ω used by the V1
        ("lumped") path:

        .. math::

           R_\text{shell,total}\;=\;\frac{\rho_c}{2\pi\,L_\text{perim}}\,
                                    \ln\!\frac{r_b}{r_a}.

        Perimeter $L_\text{perim} = 2 (d_x + d_y)$ — internal
        cross-braces of a ``style="mesh"`` foundation are *not*
        encased in concrete and therefore do not contribute.
    shell_coefficient_ohm_m : float
        Per-meter Sunde coefficient $C = \rho_c/(2\pi)\,\ln(r_b/r_a)$
        in Ω·m, consumed by the V2 ("distributed") path. The
        per-segment diagonal augmentation is then $C/\Delta s$ for a
        segment of length $\Delta s$.

    All three return values are zero when no shell is configured.
    """
    if spec.concrete_rho_ohm_m is None:
        return float(spec.wire_radius_m), 0.0, 0.0
    rho_c = _to_float(spec.concrete_rho_ohm_m, rng)
    t = _to_float(spec.concrete_thickness_m, rng)
    if rho_c <= 0.0 or t <= 0.0:
        return float(spec.wire_radius_m), 0.0, 0.0
    r_a = float(spec.wire_radius_m)
    r_b = r_a + t
    perimeter = 2.0 * (dx + dy)
    coefficient = rho_c / (2.0 * math.pi) * math.log(r_b / r_a)
    r_shell = coefficient / perimeter if perimeter > 0.0 else 0.0
    return r_b, float(r_shell), float(coefficient)


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


class GroundingSystemSpec(GeneratorConfig):
    """A complete grounding installation.

    Attributes
    ----------
    electrodes
        Ordered list of electrode specs. Order matters only for the
        choice of anchor (first *present* electrode). All present
        electrodes are bonded into one cluster.
    bond_conductor_type
        Conductor type used to bond the electrodes together. Default
        ``"bare_copper"`` matches the typical above-ground bonding
        strap inside a substation pit; switch to ``"pen"`` if you
        want the bond to be modelled like an insulated PEN cable.
    """

    electrodes: list[ElectrodeSpec] = Field(default_factory=list, min_length=0)
    bond_conductor_type: str = Field(
        default="bare_copper",
        description="Conductor type used to bond the electrodes together.",
    )

    def build_at(
        self,
        world: World,
        site_xy: tuple[float, float],
        name_prefix: str,
        rng: np.random.Generator,
    ) -> Optional[str]:
        """Materialise the grounding system inside ``world``.

        Parameters
        ----------
        world
            The :class:`World` to populate.
        site_xy
            Centre position ``(x, y)`` of this site (substation
            centre, KVS centre, building centre, …) in metres.
        name_prefix
            Prefix used for every created electrode name. The actual
            name is ``f"{name_prefix}_{kind}_{i}"``. Should be unique
            within the world.
        rng
            Random generator used to sample ``presence_prob`` and any
            geometric distributions.

        Returns
        -------
        str | None
            The name of the *anchor* electrode (the first present
            one). ``None`` if no electrode ended up being placed
            (every Bernoulli failed) — the caller may want to skip
            this site or raise.
        """
        cx, cy = site_xy
        created: list[str] = []
        for i, spec in enumerate(self.electrodes):
            # Bernoulli on presence
            p = _to_float(spec.presence_prob, rng)
            if p < 1.0 and rng.random() >= p:
                continue
            ex = cx + spec.offset_xy_m[0]
            ey = cy + spec.offset_xy_m[1]
            name = f"{name_prefix}_{spec.kind}_{i}"
            self._build_one(world, spec, ex, ey, name, rng)
            created.append(name)

        if not created:
            return None

        # Bond every non-anchor electrode to the anchor with a
        # bare-copper conductor. All bonds are short — the segment
        # length is set by the engine, not by us.
        anchor = created[0]
        for k, name in enumerate(created[1:], start=1):
            create_conductor(
                world,
                name=f"{name_prefix}_bond_{k}",
                start=anchor,
                end=name,
                conductor_type=self.bond_conductor_type,
            )
        return anchor

    # ------------------------------------------------------------------
    # Per-kind dispatcher
    # ------------------------------------------------------------------

    def _build_one(
        self,
        world: World,
        spec: ElectrodeSpec,
        ex: float,
        ey: float,
        name: str,
        rng: np.random.Generator,
    ) -> None:
        if isinstance(spec, RodElectrodeSpec):
            create_electrode(
                world, "rod", name=name,
                position=(ex, ey, _to_float(spec.depth_m, rng)),
                length=_to_float(spec.length_m, rng),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, RingElectrodeSpec):
            create_electrode(
                world, "ring", name=name,
                center=(ex, ey, _to_float(spec.depth_m, rng)),
                radius=_to_float(spec.radius_m, rng),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, StripElectrodeSpec):
            length = _to_float(spec.length_m, rng)
            angle = math.radians(_to_float(spec.orientation_deg, rng))
            depth = _to_float(spec.depth_m, rng)
            sx = ex - 0.5 * length * math.cos(angle)
            sy = ey - 0.5 * length * math.sin(angle)
            tx = ex + 0.5 * length * math.cos(angle)
            ty = ey + 0.5 * length * math.sin(angle)
            create_electrode(
                world, "strip", name=name,
                start=(sx, sy, depth),
                end=(tx, ty, depth),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, FoundationElectrodeSpec):
            depth = _to_float(spec.depth_m, rng)
            if spec.size_xy_m is not None:
                dx, dy = spec.size_xy_m
            else:
                size = _to_float(spec.size_m, rng)
                dx, dy = size, size
            # ``style="ring"`` forces a perimeter-only realisation —
            # GridMeshElectrode with n_x = n_y = 1 is exactly that
            # (one mesh cell = the four perimeter wires).
            if spec.style == "ring":
                nx, ny = 1, 1
            else:
                nx, ny = spec.n_x, spec.n_y

            # Concrete shell pre-computation (ADR-0012). When
            # ``concrete_rho_ohm_m`` is set, the wire's effective
            # radius for the solver is bumped to
            # ``wire_radius_m + concrete_thickness_m`` (Sunde shell:
            # the soil sees the concrete outer surface, not the
            # bare wire). Then two paths split:
            #
            # * ``concrete_model="lumped"`` (V1, default) — the
            #   total Sunde-shell resistance R_shell_total is
            #   recorded in ``world.concrete_shell_corrections``
            #   so the TN generator can inject it as a lumped
            #   series resistor on the PEN service drop. Zero
            #   solver-side change.
            # * ``concrete_model="distributed"`` (V2) — every
            #   segment of the foundation's strip electrodes
            #   carries the per-meter Sunde coefficient
            #   ``C = rho_c / (2 pi) * ln(r_b / r_a)``, which the
            #   image / image_2layer self-action augments on the
            #   MoM diagonal. The lumped registry is *not*
            #   populated; the distributed correction subsumes it.
            effective_radius, shell_total_ohm, shell_coeff = (
                _maybe_concrete_shell(spec, dx=dx, dy=dy, rng=rng)
            )
            if spec.concrete_model == "lumped":
                if shell_total_ohm > 0.0:
                    world.concrete_shell_corrections[name] = shell_total_ohm
                strip_shell_coefficient = 0.0
            elif spec.concrete_model == "distributed":
                strip_shell_coefficient = shell_coeff
            else:  # pragma: no cover - validated by the Literal
                raise ValueError(
                    f"Unknown concrete_model: {spec.concrete_model!r}"
                )

            # When the distributed shell is in play the foundation
            # is always materialised as a chain of strips (even at
            # angle = 0), because ``GridMeshElectrode`` is one
            # combined primitive whose segments would all share
            # the same diagonal — the per-segment augmentation is
            # only well-defined on individual strip primitives. For
            # the lumped path the historic GridMesh fast-path is
            # preserved.
            angle_deg = spec.orientation_deg or 0.0
            force_strip_path = strip_shell_coefficient > 0.0
            if abs(angle_deg) < 1e-9 and not force_strip_path:
                create_electrode(
                    world, "grid_mesh", name=name,
                    corner=(ex - dx / 2.0, ey - dy / 2.0, depth),
                    size=(dx, dy),
                    n_x=nx,
                    n_y=ny,
                    wire_radius=effective_radius,
                )
                return

            # Rotated path (or strip-forced for V2): synthesise the
            # foundation as a closed chain of axis-arbitrary
            # :class:`StripElectrode`s in the foundation-local
            # frame, rotate every endpoint into the world frame
            # and bond all sub-electrodes internally.
            self._build_rotated_foundation(
                world,
                spec=spec,
                centre_xy=(ex, ey),
                depth=depth,
                dx=dx, dy=dy,
                n_x=nx, n_y=ny,
                angle_deg=angle_deg,
                name=name,
                wire_radius_override=effective_radius,
                concrete_shell_coefficient=strip_shell_coefficient,
            )
            return
        raise TypeError(
            f"GroundingSystemSpec.build_at: unknown electrode spec "
            f"{type(spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Rotated foundation electrode (Phase A of ADR-0011)
    # ------------------------------------------------------------------

    def _build_rotated_foundation(
        self,
        world: World,
        *,
        spec: FoundationElectrodeSpec,
        centre_xy: tuple[float, float],
        depth: float,
        dx: float,
        dy: float,
        n_x: int,
        n_y: int,
        angle_deg: float,
        name: str,
        wire_radius_override: Optional[float] = None,
        concrete_shell_coefficient: float = 0.0,
    ) -> None:
        r"""Build a rotated rectangular foundation as a bonded
        Strip-electrode chain.

        Parameters
        ----------
        world
            Target :class:`World`.
        spec
            The originating :class:`FoundationElectrodeSpec`. Only
            ``wire_radius_m`` is read out here; size, mesh density,
            depth, and angle are passed in explicitly so the caller
            can override them (e.g. from an OSM footprint).
        centre_xy
            Foundation centre in metres (already offset by the
            grounding system's site centre and the spec's
            ``offset_xy_m``).
        depth
            Burial depth in metres (positive = into soil).
        dx, dy
            Side lengths of the rectangle in the foundation-local
            frame, in metres. ``dx`` is along the local long axis
            (``angle_deg`` direction).
        n_x, n_y
            Mesh density: ``n_x`` cells along the local ``+x``,
            ``n_y`` along the local ``+y``. ``(1, 1)`` produces the
            perimeter-only ring; higher values add internal braces.
        angle_deg
            Rotation of the foundation's local ``+x`` axis with
            respect to the world ``+x`` axis, in degrees.
        name
            Name of the *anchor* sub-electrode. The first wire (a
            full-length perimeter strip along the local ``-y`` edge)
            is registered with exactly this name so the surrounding
            :meth:`build_at` bonding loop can wire PEN, sources, or
            other electrodes to the foundation without knowing about
            the internal decomposition. Sub-electrodes get
            ``f"{name}_w{j}"`` (j = 1, 2, …).

        Notes
        -----
        Physical interpretation
        ^^^^^^^^^^^^^^^^^^^^^^^
        For a DIN-18014 Streifenfundament, the bonded wire follows
        the foundation outline at the strip-foundation depth. The
        OMBR projection (see
        :meth:`BuildingFootprint.oriented_bounding_rectangle`)
        approximates an arbitrary building outline by a rectangle
        whose perimeter conserves the dominant edge alignment.
        Modelling the resulting Streifenfundament as a closed
        chain of :class:`StripElectrode`s captures both the
        galvanic spreading (perimeter length contributes
        proportionally) and the inductive coupling to a parallel
        PEN trunk (the rectangle's long axis carries most of the
        mutual flux).
        """
        cos_a = math.cos(math.radians(angle_deg))
        sin_a = math.sin(math.radians(angle_deg))
        cx, cy = centre_xy
        half_x = 0.5 * dx
        half_y = 0.5 * dy

        def to_world(lx: float, ly: float) -> tuple[float, float]:
            """Rotate ``(lx, ly)`` from the foundation-local frame into
            the world frame, then translate to ``centre_xy``."""
            wx = cos_a * lx - sin_a * ly + cx
            wy = sin_a * lx + cos_a * ly + cy
            return wx, wy

        # Local-frame wire endpoints (open list of (start, end) pairs).
        # ``n_y + 1`` longitudinal wires along local +x at evenly
        # spaced y values; ``n_x + 1`` transverse wires along local +y.
        # For ``n_x = n_y = 1`` this is exactly the four perimeter
        # edges; for higher counts the internal braces are added.
        wires: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for k in range(n_y + 1):
            ly = -half_y + k * (dy / n_y)
            wires.append(((-half_x, ly), (half_x, ly)))
        for k in range(n_x + 1):
            lx = -half_x + k * (dx / n_x)
            wires.append(((lx, -half_y), (lx, half_y)))

        wire_radius = (
            float(wire_radius_override)
            if wire_radius_override is not None
            else spec.wire_radius_m
        )
        # V2: the per-strip Sunde coefficient. Only the perimeter
        # wires sit inside the concrete; internal cross-braces of
        # a ``style="mesh"`` foundation typically run through the
        # cellar floor and stay direct-in-soil. With ``n_y + 1``
        # longitudinal wires and ``n_x + 1`` transverse wires we
        # mark the first and last of each batch (i.e. the four
        # perimeter edges); internal braces keep ``coeff = 0``.
        perimeter_long_indices = {0, n_y}  # k=0 and k=n_y in the longitudinal batch
        perimeter_trans_offset = n_y + 1
        perimeter_trans_indices = {
            perimeter_trans_offset + 0,
            perimeter_trans_offset + n_x,
        }
        sub_names: list[str] = []
        for j, ((lsx, lsy), (lex, ley)) in enumerate(wires):
            sub_name = name if j == 0 else f"{name}_w{j}"
            sx, sy = to_world(lsx, lsy)
            tx, ty = to_world(lex, ley)
            on_perimeter = (
                j < (n_y + 1) and j in perimeter_long_indices
            ) or (
                j >= (n_y + 1) and j in perimeter_trans_indices
            )
            strip_coeff = (
                concrete_shell_coefficient if on_perimeter else 0.0
            )
            create_electrode(
                world, "strip", name=sub_name,
                start=(sx, sy, depth),
                end=(tx, ty, depth),
                wire_radius=wire_radius,
                concrete_shell_coefficient_ohm_m=strip_coeff,
            )
            sub_names.append(sub_name)

        # Internal bonding: every wire after the first is tied to the
        # anchor with a bare-copper bond so the foundation registers
        # as a single galvanic cluster on the engine side. The bond
        # conductor is short and contributes negligibly to the
        # foundation's spreading admittance.
        anchor = sub_names[0]
        for j, sn in enumerate(sub_names[1:], start=1):
            create_conductor(
                world,
                name=f"{name}_bond_{j}",
                start=anchor,
                end=sn,
                conductor_type=self.bond_conductor_type,
            )
