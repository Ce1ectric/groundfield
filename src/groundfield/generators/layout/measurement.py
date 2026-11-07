r"""Simulated fall-of-potential measurement for the layout builder.

Part of :mod:`groundfield.generators.layout`. An auxiliary current
electrode (*Hilfserder*) closes the test-current loop and a voltage probe
(*Spannungssonde*) is the potential reference of the reading

.. math::

   Z_\text{meas} = \frac{\varphi_\text{sub} - \varphi_\text{probe}}{I_\text{src}}.

The probe is a pure sampling point (an ideal voltmeter draws no current),
so it adds no electrode to the world.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from groundfield.generators.electrode_specs import RodElectrodeSpec
from groundfield.generators.grounding import GroundingSystemSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from groundfield.generators.layout.model import CabinetPlacement

__all__ = [
    "AuxiliaryElectrodePlacement",
    "VoltageProbePlacement",
    "triangle_rod_grounding",
]


def triangle_rod_grounding(
    rod_length_m: float = 0.5,
    triangle_side_m: float = 0.5,
) -> GroundingSystemSpec:
    r"""Default Hilfserder geometry: three parallel vertical rods in
    an equilateral triangle.

    Real-world fall-of-potential test sets use a *bundle* of rods
    rather than a single Tiefenerder so the auxiliary's own
    spreading resistance stays well below the substation's. Three
    0.5 m rods at the corners of a 0.5 m equilateral triangle give
    roughly a third of the single-rod impedance.
    """
    # Equilateral triangle, centroid at the origin: corners are at
    # angles 90 deg, 210 deg, 330 deg from the centre at a radius
    # ``triangle_side_m / sqrt(3)``.
    r = triangle_side_m / math.sqrt(3.0)
    offsets = [
        (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        for a in (90.0, 210.0, 330.0)
    ]
    return GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(
                length_m=rod_length_m,
                depth_m=0.0,
                offset_xy_m=off,
            )
            for off in offsets
        ],
    )


def _default_aux_grounding() -> GroundingSystemSpec:
    """Project default for the Hilfserder: 3 × 0.5 m rods in a
    0.5 m equilateral triangle (parallel-bonded)."""
    return triangle_rod_grounding(rod_length_m=0.5, triangle_side_m=0.5)


class AuxiliaryElectrodePlacement(BaseModel):
    r"""Auxiliary current electrode (Hilfserder) for a fall-of-
    potential measurement of the substation grounding.

    The auxiliary electrode closes the test-current loop. It is
    materialised in :meth:`OrtsnetzLayout.to_world` via
    :meth:`GroundingSystemSpec.build_at`, giving the user full
    control over its geometry (single rod, ring, mesh, foundation,
    or any combination). The default is a *bundle of three 0.5 m
    rods in a 0.5 m equilateral triangle, parallel-bonded* —
    matching real-world Hilfserder practice and yielding a
    spreading resistance roughly a third of a single rod.

    During :meth:`OrtsnetzLayout.to_world` the layout adds a
    second :class:`CurrentSource` with ``phase_deg=180`` at the
    anchor electrode so the engine's potential field develops the
    expected dipole pattern (positive trumpet at the substation,
    negative trumpet at the Hilfserder).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "aux"
    position_xy: tuple[float, float]
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_aux_grounding,
        description=(
            "Per-Hilfserder grounding system spec. Default is a "
            "3 × 0.5 m rod triangle (0.5 m side); override for "
            "single-rod or other custom geometries."
        ),
    )


class VoltageProbePlacement(BaseModel):
    r"""Voltage probe (Spannungssonde) used during the simulated
    fall-of-potential measurement.

    Unlike the :class:`AuxiliaryElectrodePlacement`, the probe is
    modelled as a pure *sampling point* in 3-D space: an ideal
    voltmeter has infinite input impedance, so a physical probe
    rod would carry zero current and therefore must not perturb
    the engine's potential field. In :meth:`OrtsnetzLayout.to_world`
    no rod electrode is created for the probe; in
    :meth:`OrtsnetzLayout.measured_grounding_impedance` the field
    is sampled at :attr:`position_xy` via
    :meth:`FieldResult.potential`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "probe"
    position_xy: tuple[float, float]
    depth_m: float = Field(
        default=0.0,
        description=(
            "Depth in m at which the field is sampled. 0 = ground "
            "surface (typical: a probe rod tip just below grass)."
        ),
    )


class _MeasurementMixin:
    """Measurement methods of :class:`~groundfield.generators.layout.LvNetworkLayout`."""

    if TYPE_CHECKING:  # pragma: no cover - attributes of the layout model
        substation_name: str
        substation_xy: tuple[float, float]
        kvs_placements: list["CabinetPlacement"]
        auxiliary_electrode: Optional[AuxiliaryElectrodePlacement]
        voltage_probe: Optional[VoltageProbePlacement]

        def lat_lon_to_xy(
            self, lat_deg: float, lon_deg: float
        ) -> tuple[float, float]: ...

    def add_auxiliary_electrode(
        self,
        *,
        distance_m: Optional[float] = None,
        direction_deg: float = 0.0,
        position_xy: Optional[tuple[float, float]] = None,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: str = "aux",
        grounding: Optional[GroundingSystemSpec] = None,
        length_m: Optional[float] = None,
        depth_m: float = 0.0,
    ) -> str:
        r"""Place the auxiliary current electrode (Hilfserder).

        Exactly one of the three position kinds must be set:

        * ``distance_m`` (with optional ``direction_deg``) places
          the electrode at
          :math:`\mathbf{r}_\text{aux} = \mathbf{r}_\text{sub} +
          d\,(\cos\theta,\,\sin\theta)`,
          with :math:`\theta` measured CCW from the local +x axis.
          ``direction_deg = 0`` corresponds to east, ``90`` to
          north. This is the convenient form for a controlled
          sensitivity sweep over distance, mimicking the
          fall-of-potential test geometry.
        * ``position_xy`` sets the electrode at an explicit
          ENU coordinate.
        * ``position_lat_lon`` projects through the layout's
          :attr:`frame_origin_lat_lon` (raises if no frame origin
          is set).

        Parameters
        ----------
        distance_m
            Manhattan-radius from the substation in metres. Must
            be positive.
        direction_deg
            Direction in degrees CCW from +x (east). Default 0.
        position_xy, position_lat_lon
            See above; mutually exclusive with ``distance_m``.
        name
            Anchor id of the auxiliary electrode (defaults to
            ``"aux"``). Must be unique across the layout. The
            materialised electrodes inside :meth:`to_world` are
            named ``f"{name}_{kind}_{i}"`` -- e.g. for the default
            triangle: ``aux_rod_0`` / ``aux_rod_1`` / ``aux_rod_2``.
        grounding
            :class:`GroundingSystemSpec` to use for the
            Hilfserder. ``None`` (default) selects the project
            default (3 × 0.5 m rods in a 0.5 m equilateral
            triangle, parallel-bonded), which matches typical
            field-deployment practice.
        length_m
            Convenience override: when ``grounding`` is ``None`` and
            ``length_m`` is given, the Hilfserder collapses to a
            *single* rod of that length (legacy v0.7 behaviour).
            Ignored when ``grounding`` is set explicitly.
        depth_m
            Depth of the rod head when ``length_m`` is used to
            build a single-rod spec. Default 0.

        Returns
        -------
        str
            The auxiliary anchor's name, identical to ``name``.

        Raises
        ------
        ValueError
            If no positioning kind is given, if more than one is
            given, or if ``name`` clashes with the substation or
            an existing KVS.
        RuntimeError
            From :meth:`lat_lon_to_xy` when
            ``position_lat_lon`` is set but the layout has no
            frame origin.

        Notes
        -----
        The placement *replaces* any previously configured
        auxiliary electrode (there is only one per layout in v1).
        Call this method again with different parameters to
        sweep the auxiliary-electrode distance without rebuilding
        the entire layout.
        """
        provided = sum(
            arg is not None for arg in (distance_m, position_xy, position_lat_lon)
        )
        if provided != 1:
            raise ValueError(
                "OrtsnetzLayout.add_auxiliary_electrode: exactly one "
                "of ``distance_m``, ``position_xy`` or "
                "``position_lat_lon`` must be set."
            )

        if distance_m is not None:
            if distance_m <= 0.0:
                raise ValueError(
                    "OrtsnetzLayout.add_auxiliary_electrode: "
                    f"distance_m must be > 0, got {distance_m}."
                )
            theta = math.radians(direction_deg)
            sx, sy = self.substation_xy
            xy = (sx + distance_m * math.cos(theta), sy + distance_m * math.sin(theta))
        elif position_lat_lon is not None:
            xy = self.lat_lon_to_xy(*position_lat_lon)
        else:
            assert position_xy is not None
            xy = tuple(position_xy)

        if name == self.substation_name:
            raise ValueError(
                f"OrtsnetzLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with the substation name."
            )
        if any(k.name == name for k in self.kvs_placements):
            raise ValueError(
                f"OrtsnetzLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with an existing KVS."
            )

        # Resolve the grounding spec. Priority:
        #   1. Explicit ``grounding`` argument wins.
        #   2. ``length_m`` collapses to a single-rod spec (legacy).
        #   3. Otherwise use the project default (3 x 0.5 m triangle).
        if grounding is not None:
            grounding_spec = grounding
        elif length_m is not None:
            grounding_spec = GroundingSystemSpec(
                electrodes=[
                    RodElectrodeSpec(
                        length_m=float(length_m),
                        depth_m=depth_m,
                    ),
                ],
            )
        else:
            grounding_spec = _default_aux_grounding()

        self.auxiliary_electrode = AuxiliaryElectrodePlacement(
            name=name,
            position_xy=xy,
            grounding=grounding_spec,
        )
        return name

    def add_voltage_probe(
        self,
        *,
        inline_fraction: Optional[float] = None,
        perpendicular_fraction: Optional[float] = None,
        perpendicular_side: str = "left",
        position_xy: Optional[tuple[float, float]] = None,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: str = "probe",
        depth_m: float = 0.0,
    ) -> str:
        r"""Place the voltage probe (Spannungssonde).

        The voltage probe replaces the "remote earth" reference of
        a simple ``U_sub / I`` measurement with the *actual*
        ground potential at a chosen point — exactly what a
        field-deployed grounding-impedance meter does.

        Four positioning kinds are accepted:

        * ``inline_fraction`` — along the substation → aux axis,
          at a fraction :math:`f` of the aux distance
          (:math:`f = 0` is the substation, :math:`f = 1` is the
          Hilfserder, :math:`f = 0.5` is the midpoint, which is
          the standard "50 % rule"). Requires
          :meth:`add_auxiliary_electrode` to have been called.
        * ``perpendicular_fraction`` (with ``perpendicular_side``
          ``"left"`` or ``"right"``) — 90° rotated from the
          substation → aux axis (left = CCW = north when aux runs
          east, right = CW = south). The probe distance from the
          substation is :math:`f \cdot D_\text{aux}`. This matches
          the *senkrecht zur Hilfserdertrasse* setup commonly
          used to suppress inductive coupling between the
          current-feed loop and the voltage-probe lead.
        * ``position_xy`` — explicit ENU metres.
        * ``position_lat_lon`` — WGS84 (projects through
          :attr:`frame_origin_lat_lon`).

        Examples
        --------
        Classic 50 % inline probe between substation and a 200 m
        east Hilfserder:

        >>> layout.add_auxiliary_electrode(distance_m=200.0,
        ...                                direction_deg=0.0)
        >>> layout.add_voltage_probe(inline_fraction=0.5)
        # probe ends up at (100, 0) relative to the substation

        90° perpendicular probe at the same fraction, on the
        "left" (= north) side:

        >>> layout.add_voltage_probe(perpendicular_fraction=0.5,
        ...                          perpendicular_side="left")
        # probe ends up at (0, 100) relative to the substation
        """
        provided = sum(
            arg is not None
            for arg in (
                inline_fraction,
                perpendicular_fraction,
                position_xy,
                position_lat_lon,
            )
        )
        if provided != 1:
            raise ValueError(
                "OrtsnetzLayout.add_voltage_probe: exactly one of "
                "``inline_fraction``, ``perpendicular_fraction``, "
                "``position_xy`` or ``position_lat_lon`` must be set."
            )
        if perpendicular_side not in ("left", "right"):
            raise ValueError(
                "OrtsnetzLayout.add_voltage_probe: perpendicular_side "
                f"must be 'left' or 'right', got {perpendicular_side!r}."
            )

        if inline_fraction is not None or perpendicular_fraction is not None:
            if self.auxiliary_electrode is None:
                raise RuntimeError(
                    "OrtsnetzLayout.add_voltage_probe: the inline / "
                    "perpendicular fractions need an auxiliary "
                    "electrode -- call ``add_auxiliary_electrode`` "
                    "first."
                )
            sx, sy = self.substation_xy
            ax, ay = self.auxiliary_electrode.position_xy
            d_vec = (ax - sx, ay - sy)
            d_aux = math.hypot(*d_vec)
            if d_aux <= 0.0:
                raise RuntimeError(
                    "OrtsnetzLayout.add_voltage_probe: the auxiliary "
                    "electrode coincides with the substation -- the "
                    "fractional placement is undefined."
                )
            u_inline = (d_vec[0] / d_aux, d_vec[1] / d_aux)
            if inline_fraction is not None:
                f = float(inline_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "OrtsnetzLayout.add_voltage_probe: "
                        "inline_fraction must be in [0, 1], got "
                        f"{f}."
                    )
                xy = (sx + f * d_aux * u_inline[0], sy + f * d_aux * u_inline[1])
            else:
                f = float(perpendicular_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "OrtsnetzLayout.add_voltage_probe: "
                        "perpendicular_fraction must be in [0, 1], "
                        f"got {f}."
                    )
                # 90° rotation: left = CCW, right = CW.
                if perpendicular_side == "left":
                    u_perp = (-u_inline[1], u_inline[0])
                else:
                    u_perp = (u_inline[1], -u_inline[0])
                xy = (sx + f * d_aux * u_perp[0], sy + f * d_aux * u_perp[1])
        elif position_lat_lon is not None:
            xy = self.lat_lon_to_xy(*position_lat_lon)
        else:
            assert position_xy is not None
            xy = tuple(position_xy)

        if name == self.substation_name:
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with the substation name."
            )
        if any(k.name == name for k in self.kvs_placements):
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with an existing KVS."
            )
        if (
            self.auxiliary_electrode is not None
            and name == self.auxiliary_electrode.name
        ):
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with the auxiliary electrode."
            )

        self.voltage_probe = VoltageProbePlacement(
            name=name,
            position_xy=xy,
            depth_m=depth_m,
        )
        return name

    def measured_grounding_impedance(
        self,
        result: object,
        *,
        frequency_index: int = 0,
        source_magnitude_A: float = 1.0,
        probe_xy: Optional[tuple[float, float]] = None,
        probe_depth_m: float = 0.0,
    ) -> complex:
        r"""Compute the simulated fall-of-potential measurement reading.

        Returns
        -------
        complex
            :math:`Z_\text{meas} =
            (\varphi_\text{sub} - \varphi_\text{probe}) / I_\text{src}`
            at the requested frequency.

        When neither ``probe_xy`` nor :attr:`voltage_probe` is set,
        the reference falls back to remote earth, i.e. the method
        returns :math:`\varphi_\text{sub} / I_\text{src}`.

        Parameters
        ----------
        result
            :class:`FieldResult` from :meth:`Engine.solve`.
        frequency_index, source_magnitude_A
            Pick the frequency and the test-current normalisation.
        probe_xy
            Optional explicit probe position in local ENU metres
            (overrides :attr:`voltage_probe` for this call only).
            Lets the caller compare several probe geometries from
            a single solve -- useful for the
            "0° vs 90° probe at the same distance" comparison
            without having to call ``add_voltage_probe`` /
            ``to_world`` again.
        probe_depth_m
            Sampling depth for the override (default 0 = surface).
            Ignored when ``probe_xy`` is ``None``.

        Raises
        ------
        RuntimeError
            If no auxiliary electrode is configured (the
            measurement loop is then physically open).
        """
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "OrtsnetzLayout.measured_grounding_impedance: no "
                "auxiliary electrode is configured. Call "
                "``add_auxiliary_electrode`` first."
            )

        # Find the substation cluster's anchor in the result.
        electrode_potentials = getattr(result, "electrode_potentials", None)
        if electrode_potentials is None:
            raise TypeError(
                "OrtsnetzLayout.measured_grounding_impedance: "
                "``result`` does not look like a FieldResult."
            )
        sub_anchor: Optional[str] = None
        prefix = f"{self.substation_name}_"
        for ename in electrode_potentials:
            if ename.startswith(prefix):
                sub_anchor = ename
                break
        if sub_anchor is None:
            raise RuntimeError(
                "OrtsnetzLayout.measured_grounding_impedance: no "
                "electrode with prefix "
                f"{self.substation_name!r} found in the result. "
                "Did you re-run ``to_world`` before solving?"
            )

        phi_sub = electrode_potentials[sub_anchor][frequency_index]

        # Pick the probe location: explicit override > layout's
        # voltage_probe > remote earth fall-back.
        if probe_xy is not None:
            sample_xy = tuple(probe_xy)
            sample_depth = probe_depth_m
        elif self.voltage_probe is not None:
            sample_xy = self.voltage_probe.position_xy
            sample_depth = self.voltage_probe.depth_m
        else:
            sample_xy = None

        if sample_xy is None:
            phi_ref: complex = 0.0 + 0.0j
        else:
            point = np.asarray(
                [[sample_xy[0], sample_xy[1], sample_depth]],
                dtype=float,
            )
            phi_ref = complex(
                result.potential(
                    point,
                    frequency_index=frequency_index,
                )[0]
            )

        return (phi_sub - phi_ref) / source_magnitude_A

    def verify_current_balance(
        self,
        result: object,
        *,
        frequency_index: int = 0,
        source_magnitude_A: float = 1.0,
        tolerance: float = 0.02,
    ) -> dict:
        r"""Plausibility check: sum of leakage currents must close the
        measurement loop.

        With the v0.7 closed-loop wiring the engine sees:

        * a positive ``+I`` injection at the substation cluster, which
          distributes itself across the substation grounding *and* the
          PEN-bonded foundations (because the PEN cables are
          finite-impedance branches that share the substation
          potential);
        * a negative ``-I`` injection at the Hilfserder cluster.

        Per Kirchhoff the *total* leakage current at every other
        electrode must vanish:

        .. math::
            \sum_e I_e^\text{(positive side)} \approx +I_\text{src}
            ,\qquad
            \sum_e I_e^\text{(aux side)} \approx -I_\text{src}

        Returns a dict with the two sums and a boolean ``ok`` flag
        indicating that both sums match :math:`\pm I_\text{src}`
        within ``tolerance`` (default 2 % of the test current).

        The "positive side" is every electrode whose name does *not*
        start with the auxiliary's name prefix; the "aux side" is
        every electrode whose name *does* start with that prefix.
        """
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "OrtsnetzLayout.verify_current_balance: no auxiliary "
                "electrode is configured. The measurement loop is "
                "physically open; current balance is undefined."
            )
        aux_prefix = f"{self.auxiliary_electrode.name}_"
        currents = result.electrode_currents
        pos_sum: complex = 0j
        neg_sum: complex = 0j
        for ename, i_list in currents.items():
            i = i_list[frequency_index]
            if ename.startswith(aux_prefix) or ename == self.auxiliary_electrode.name:
                neg_sum += i
            else:
                pos_sum += i
        expected_pos = +source_magnitude_A
        expected_neg = -source_magnitude_A
        err_pos = abs(pos_sum - expected_pos)
        err_neg = abs(neg_sum - expected_neg)
        tol_abs = tolerance * source_magnitude_A
        ok = (err_pos <= tol_abs) and (err_neg <= tol_abs)
        return {
            "positive_side_A": pos_sum,
            "auxiliary_side_A": neg_sum,
            "expected_positive_A": expected_pos,
            "expected_auxiliary_A": expected_neg,
            "abs_error_positive_A": err_pos,
            "abs_error_auxiliary_A": err_neg,
            "tolerance_A": tol_abs,
            "ok": ok,
        }
