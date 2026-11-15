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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from groundfield.generators._compat import DeprecatedKeyDict, take_renamed_kwargs
from groundfield.generators.electrode_specs import RodElectrodeSpec
from groundfield.generators.grounding import GroundingSystemSpec

#: Source kinds whose (impressed or solved) current is the test current of
#: the reading (a voltage source since 2026.11, audit W2A-01).
_SOURCE_KINDS = ("current", "voltage")


def _preferred_record(result: object, match: Callable[[Any], bool]) -> Any:
    """First current-source record of ``result`` that ``match`` accepts, else
    the first such voltage-source record, else ``None`` (audit W2R2-05)."""
    records = getattr(result, "sources", None) or []
    for kind in _SOURCE_KINDS:  # current first
        for record in records:
            if record.kind == kind and match(record):
                return record
    return None


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
    r"""Default auxiliary-electrode geometry: three parallel vertical rods in
    an equilateral triangle.

    Real-world fall-of-potential test sets use a *bundle* of rods
    rather than a single driven rod so the auxiliary's own
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
    """Project default for the auxiliary electrode: 3 × 0.5 m rods in a
    0.5 m equilateral triangle (parallel-bonded)."""
    return triangle_rod_grounding(rod_length_m=0.5, triangle_side_m=0.5)


class AuxiliaryElectrodePlacement(BaseModel):
    r"""Auxiliary current electrode (Hilfserder) for a fall-of-
    potential measurement of the substation grounding.

    The auxiliary electrode closes the test-current loop. It is
    materialised in :meth:`LvNetworkLayout.to_world` via
    :meth:`GroundingSystemSpec.build_at`, giving the user full
    control over its geometry (single rod, ring, mesh, foundation,
    or any combination). The default is a *bundle of three 0.5 m
    rods in a 0.5 m equilateral triangle, parallel-bonded* —
    matching real-world practice and yielding a
    spreading resistance roughly a third of a single rod.

    In :meth:`LvNetworkLayout.to_world` the substation's current
    source returns its current through the auxiliary electrode's anchor
    (``Source.return_to``), so the potential field develops the
    expected dipole pattern (positive trumpet at the substation,
    negative trumpet at the auxiliary electrode). Up to 0.15.0 the
    solver ignored ``return_to`` and a second source with
    ``phase_deg=180`` on the auxiliary electrode stood in for it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "aux"
    position_xy: tuple[float, float]
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_aux_grounding,
        description=(
            "Auxiliary-electrode grounding system spec. Default is a "
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
    the engine's potential field. In :meth:`LvNetworkLayout.to_world`
    no rod electrode is created for the probe; in
    :meth:`LvNetworkLayout.measured_grounding_impedance` the field
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
        cabinet_placements: list["CabinetPlacement"]
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
            auxiliary electrode. ``None`` (default) selects the project
            default (3 × 0.5 m rods in a 0.5 m equilateral
            triangle, parallel-bonded), which matches typical
            field-deployment practice.
        length_m
            Convenience override: when ``grounding`` is ``None`` and
            ``length_m`` is given, the auxiliary electrode collapses to a
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
            an existing cabinet.
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
                "LvNetworkLayout.add_auxiliary_electrode: exactly one "
                "of ``distance_m``, ``position_xy`` or "
                "``position_lat_lon`` must be set."
            )

        if distance_m is not None:
            if distance_m <= 0.0:
                raise ValueError(
                    "LvNetworkLayout.add_auxiliary_electrode: "
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
                f"LvNetworkLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with the substation name."
            )
        if any(k.name == name for k in self.cabinet_placements):
            raise ValueError(
                f"LvNetworkLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with an existing cabinet."
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
          auxiliary electrode, :math:`f = 0.5` is the midpoint, which is
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
        east auxiliary electrode:

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
                "LvNetworkLayout.add_voltage_probe: exactly one of "
                "``inline_fraction``, ``perpendicular_fraction``, "
                "``position_xy`` or ``position_lat_lon`` must be set."
            )
        if perpendicular_side not in ("left", "right"):
            raise ValueError(
                "LvNetworkLayout.add_voltage_probe: perpendicular_side "
                f"must be 'left' or 'right', got {perpendicular_side!r}."
            )

        if inline_fraction is not None or perpendicular_fraction is not None:
            if self.auxiliary_electrode is None:
                raise RuntimeError(
                    "LvNetworkLayout.add_voltage_probe: the inline / "
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
                    "LvNetworkLayout.add_voltage_probe: the auxiliary "
                    "electrode coincides with the substation -- the "
                    "fractional placement is undefined."
                )
            u_inline = (d_vec[0] / d_aux, d_vec[1] / d_aux)
            if inline_fraction is not None:
                f = float(inline_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "LvNetworkLayout.add_voltage_probe: "
                        "inline_fraction must be in [0, 1], got "
                        f"{f}."
                    )
                xy = (sx + f * d_aux * u_inline[0], sy + f * d_aux * u_inline[1])
            else:
                f = float(perpendicular_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "LvNetworkLayout.add_voltage_probe: "
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
                f"LvNetworkLayout.add_voltage_probe: name {name!r} "
                "clashes with the substation name."
            )
        if any(k.name == name for k in self.cabinet_placements):
            raise ValueError(
                f"LvNetworkLayout.add_voltage_probe: name {name!r} "
                "clashes with an existing cabinet."
            )
        if (
            self.auxiliary_electrode is not None
            and name == self.auxiliary_electrode.name
        ):
            raise ValueError(
                f"LvNetworkLayout.add_voltage_probe: name {name!r} "
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
        source_magnitude_a: Optional[float] = None,
        probe_xy: Optional[tuple[float, float]] = None,
        probe_depth_m: float = 0.0,
        **deprecated: float,
    ) -> complex:
        r"""Compute the simulated fall-of-potential measurement reading.

        Returns
        -------
        complex
            :math:`Z_\text{meas} =
            (\varphi_\text{sub} - \varphi_\text{probe}) / I_\text{src}`
            in Ω at the requested frequency.

        When neither ``probe_xy`` nor :attr:`voltage_probe` is set,
        the reference falls back to remote earth, i.e. the method
        returns :math:`\varphi_\text{sub} / I_\text{src}`.

        Parameters
        ----------
        result
            :class:`FieldResult` from :meth:`Engine.solve`.
        frequency_index
            Index of the frequency to read.
        source_magnitude_a
            Test current :math:`I_\text{src}` in A. ``None`` (default)
            takes the current of the solved source that feeds the
            substation from the result — the impressed current of a
            current source, the solved current of a voltage source
            (since 2026.11, audit W2A-01); results without a source
            snapshot fall back to 1 A (the default before 2026.11,
            which made the reading of a world built with another
            ``source_magnitude_a`` wrong unless it was repeated here).
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
        ValueError
            If ``source_magnitude_a`` is ``None`` and the source that
            feeds the substation carries no current at
            ``frequency_index`` (a voltage record without its solved
            current).

        Notes
        -----
        The probe is an ideal voltmeter (a sampling point of the field,
        no electrode), and the substation source returns its current
        through the auxiliary electrode (``Source.return_to``), so the
        reading follows the fall-of-potential curve of the real
        measurement: it approaches the impedance of the network against
        remote earth for a probe at 61.8 % of a remote auxiliary
        electrode.
        """
        renamed = take_renamed_kwargs(
            "LvNetworkLayout.measured_grounding_impedance",
            deprecated,
            {"source_magnitude_A": "source_magnitude_a"},
        )
        source_magnitude_a = renamed.get("source_magnitude_a", source_magnitude_a)
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "LvNetworkLayout.measured_grounding_impedance: no "
                "auxiliary electrode is configured. Call "
                "``add_auxiliary_electrode`` first."
            )

        # Find the substation cluster's anchor in the result.
        electrode_potentials = getattr(result, "electrode_potentials", None)
        if electrode_potentials is None:
            raise TypeError(
                "LvNetworkLayout.measured_grounding_impedance: "
                "``result`` does not look like a FieldResult."
            )
        sub_anchor = self._substation_anchor(result)
        if sub_anchor is None:
            raise RuntimeError(
                "LvNetworkLayout.measured_grounding_impedance: no "
                "electrode with prefix "
                f"{self.substation_name!r} found in the result. "
                "Did you re-run ``to_world`` before solving?"
            )

        phi_sub = electrode_potentials[sub_anchor][frequency_index]
        current = self._source_current(
            result, sub_anchor, frequency_index, source_magnitude_a
        )

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

        return (phi_sub - phi_ref) / current

    def _substation_anchor(self, result: object) -> Optional[str]:
        """Electrode the substation source feeds, else the first ``substation_*``.

        The source is the first current source at the substation or,
        without one, the first voltage source there (since 2026.11, audits
        W2A-01, W2R2-05).
        """
        prefix = f"{self.substation_name}_"
        record = _preferred_record(result, lambda r: r.attached_to.startswith(prefix))
        if record is not None:
            return record.attached_to
        for ename in getattr(result, "electrode_potentials", {}):
            if ename.startswith(prefix):
                return ename
        return None

    @staticmethod
    def _source_current(
        result: object,
        anchor: str,
        frequency_index: int,
        source_magnitude_a: Optional[float],
    ) -> complex:
        """Test current in A: explicit, else the solved source feeding ``anchor``.

        The first current source at ``anchor``, else the first voltage
        source there (audit W2R2-05: a current source wins, as before
        2026.11). The current of a voltage source is the solved one (audit
        W2A-01; up to then a voltage-driven world fell back to 1 A); a
        voltage record without it raises :class:`ValueError`. Results
        without a source snapshot fall back to 1 A.
        """
        if source_magnitude_a is not None:
            return complex(source_magnitude_a)
        record = _preferred_record(result, lambda r: r.attached_to == anchor)
        if record is None:
            return 1.0 + 0.0j
        try:
            return complex(record.current[frequency_index])
        except IndexError:
            raise ValueError(
                f"{record.kind.capitalize()} source {record.name!r} feeds "
                f"{anchor!r}, but the result carries no solved source "
                f"current at frequency index {frequency_index}; pass "
                "source_magnitude_a explicitly."
            ) from None

    def verify_current_balance(
        self,
        result: object,
        *,
        frequency_index: int = 0,
        source_magnitude_a: Optional[float] = None,
        tolerance: float = 0.02,
        **deprecated: float,
    ) -> dict:
        r"""Plausibility check: sum of leakage currents must close the
        measurement loop.

        The substation source is a two-terminal source
        (``Source.return_to``), so the engine sees

        * a positive ``+I`` injection at the substation cluster, which
          distributes itself across the substation grounding *and* the
          PEN-bonded foundations (because the PEN cables are
          finite-impedance branches that share the substation
          potential);
        * a negative ``-I`` injection at the auxiliary-electrode cluster.

        Per Kirchhoff the *total* leakage current at every other
        electrode must vanish:

        .. math::
            \sum_e I_e^\text{(positive side)} \approx +I_\text{src}
            ,\qquad
            \sum_e I_e^\text{(aux side)} \approx -I_\text{src}

        Returns a dict with the two sums (``positive_side_a``,
        ``auxiliary_side_a``, in A) and a boolean ``ok`` flag
        indicating that both sums match :math:`\pm I_\text{src}`
        within ``tolerance`` (default 2 % of the test current). The
        former upper-case keys (``positive_side_A``, ...) stay readable
        with a :class:`DeprecationWarning`.

        ``source_magnitude_a`` is the test current in A; ``None``
        (default) takes it from the solved source that feeds the
        substation — a current source, or the solved current of a
        voltage source (since 2026.11, audit W2A-01; a voltage record
        without it raises :class:`ValueError`) — and 1 A for results
        without a source snapshot.

        The "positive side" is every electrode whose name does *not*
        start with the auxiliary's name prefix; the "aux side" is
        every electrode whose name *does* start with that prefix.
        """
        renamed = take_renamed_kwargs(
            "LvNetworkLayout.verify_current_balance",
            deprecated,
            {"source_magnitude_A": "source_magnitude_a"},
        )
        source_magnitude_a = renamed.get("source_magnitude_a", source_magnitude_a)
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "LvNetworkLayout.verify_current_balance: no auxiliary "
                "electrode is configured. The measurement loop is "
                "physically open; current balance is undefined."
            )
        anchor = self._substation_anchor(result) or ""
        current = self._source_current(
            result, anchor, frequency_index, source_magnitude_a
        )
        i_src = float(current.real) if current.imag == 0.0 else current
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
        expected_pos = +i_src
        expected_neg = -i_src
        err_pos = abs(pos_sum - expected_pos)
        err_neg = abs(neg_sum - expected_neg)
        tol_abs = tolerance * abs(i_src)
        ok = (err_pos <= tol_abs) and (err_neg <= tol_abs)
        values = {
            "positive_side_a": pos_sum,
            "auxiliary_side_a": neg_sum,
            "expected_positive_a": expected_pos,
            "expected_auxiliary_a": expected_neg,
            "abs_error_positive_a": err_pos,
            "abs_error_auxiliary_a": err_neg,
            "tolerance_a": tol_abs,
            "ok": ok,
        }
        old_keys = {k[:-2] + "_A": k for k in values if k.endswith("_a")}
        return DeprecatedKeyDict(values, old_keys, "verify_current_balance()")
