"""Connection conductors between electrodes (or arbitrary points).

A :class:`Conductor` models a single wire segment with a start point,
an end point, a wire radius, and a conductor type. It can be located
above ground (PEN, low-voltage line) or inside the soil (connection
between electrodes, cable shield) — the $z$-coordinate decides.

Conductor impedance model
-------------------------
A conductor between two electrodes carries the **series resistance**

$$
R_\\text{ser} \\;=\\; \\rho_\\text{mat}\\, L \\,/\\, A,
$$

where $\\rho_\\text{mat}$ is the material resistivity, $L$ the
geometric length, and $A = \\pi r_\\text{wire}^2$ the wire's
cross section (or an explicit ``cross_section`` if set). When
$R_\\text{ser}$ is below the threshold returned by
:meth:`Conductor.is_ideal` the solver collapses the two end
electrodes into a single cluster (galvanic short, classical
``[image_2layer]`` behaviour). Otherwise the solver inserts the
conductor as a branch with admittance $1/R_\\text{ser}$ in the
nodal-analysis system that augments the multi-port grounding
matrix (see ADR-0003).

Coupling models (inductive Neumann coupling, Carson and Sommerfeld
earth-return corrections) live in :mod:`groundfield.coupling`; this
module holds the conductor's own series impedance. With
``resistance_model="skin"`` (opt-in, review pass 10, finding 10.37) the
internal impedance per unit length follows the exact solid-round Bessel
solution (:func:`groundfield.conductors.skin.solid_round_internal_impedance`);
the default ``"dc"`` keeps $\\rho L / A$ at every frequency.

Input validation (review pass 10, findings 10.43 / 10.67): non-finite
or non-positive physical values raise with the parameter name; a cross
section above $10^{-2}\\,\\mathrm{m^2}$ warns (a value in mm² such as
``cross_section=50`` would otherwise be a silent ideal short of
6.7 nΩ); a distributed conductor must have a finite series resistance.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from groundfield.conductors.skin import solid_round_internal_impedance

__all__ = [
    "Conductor",
    "ConductorType",
    "SoilCoupling",
    "InductanceModel",
    "ResistanceModel",
    "CrossSectionUnitWarning",
    "MAX_PLAUSIBLE_CROSS_SECTION",
]

Point3D = tuple[float, float, float]

ConductorType = Literal[
    "pen",  # PEN conductor in low-voltage networks
    "cable_shield",  # cable shield: thin-walled tube around its core (Q32 / Q33)
    "bare_copper",  # bare copper wire
    "overhead",  # overhead line / earth wire
    "generic",  # fallback
]

SoilCoupling = Literal[
    "isolated",  # cable / PEN inside an insulating jacket — no leakage
    "galvanic",  # bare copper / exposed shield — leakage along the wire
]

InductanceModel = Literal[
    "neumann",  # Neumann double-line integrals + the engine's earth-return model
]

ResistanceModel = Literal[
    "dc",  # rho * L / A at every frequency (default)
    "skin",  # exact solid-round internal impedance (Bessel solution)
]

#: Largest cross section in m² accepted without a warning (100 cm²).
#: Real conductors stay far below it; a value in mm² does not.
MAX_PLAUSIBLE_CROSS_SECTION = 1e-2

_MU_0 = 4.0e-7 * math.pi


class CrossSectionUnitWarning(UserWarning):
    """A conductor cross section is implausibly large for m².

    ``Conductor.cross_section`` is in m². A value above
    :data:`MAX_PLAUSIBLE_CROSS_SECTION` almost always means the value
    was given in mm² (``50`` instead of ``50e-6``), which turns the
    conductor into an ideal short (series resistance in the nΩ range).
    Silence with ``warnings.simplefilter("ignore", CrossSectionUnitWarning)``
    when a huge cross section is intended (e.g. a deliberate near-ideal
    bond).
    """


# A conductor whose series resistance is below this threshold (in Ω)
# is treated as an *ideal* galvanic short by the solver. The threshold
# is well below typical low-voltage cable resistances (a 30 m run of
# 50 mm² Al has R ≈ 16 mΩ) and well above floating-point noise.
_IDEAL_RESISTANCE_THRESHOLD = 1e-6


class Conductor(BaseModel):
    """Wire segment between two points.

    Attributes
    ----------
    name
        Unique name within the ``World``.
    start, end
        Start and end point ``(x, y, z)`` in metres. ``z > 0`` puts
        the conductor inside the soil, ``z < 0`` above ground.
    start_electrode, end_electrode
        Optional name of the electrode anchored at start / end. If
        both are set, the solver inserts this conductor as a branch
        in the nodal analysis: ideal (``cross_section=None``) → hard
        cluster constraint, finite section → branch with
        $R_\\text{ser} = \\rho L / A$.
    conductor_type
        Type tag from :data:`ConductorType`. ``"cable_shield"`` marks the
        shield of a concentric cable, a thin-walled tube (user decisions
        Q32 and Q33 of 2026.11): with ``inductance_model="neumann"`` a
        thinner conductor drawn on its axis, inside it, is its core and
        couples to it at the shield radius (the geometric mean distance of
        a point inside a tube), and its self inductance carries no internal
        inductance of a solid wire. Two conductors of other types on one
        axis are separate conductors that touch. The other tags only
        select plot styles.
    wire_radius
        Wire radius in metres. Default 5 mm. Used for the geometric
        thin-wire approximation in the segment self-action and as
        a fallback for ``cross_section`` when the latter is given as
        ``"from_radius"``.
    resistivity
        Material resistivity in $\\Omega\\,\\mathrm{m}$. Default:
        copper (1.68e-8).
    cross_section
        Conductor cross section in $\\mathrm{m}^2$. ``None``
        (default) means *ideal galvanic short* — the conductor is a
        zero-impedance bridge. A finite value enables the
        nodal-analysis branch model with
        $R_\\text{ser} = \\rho_\\text{mat}\\, L / A$. The string
        ``"from_radius"`` is a convenience shortcut for
        $A = \\pi r_\\text{wire}^2$.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(...)
    start: Point3D = Field(..., description="Start point (x, y, z) in m.")
    end: Point3D = Field(..., description="End point (x, y, z) in m.")
    start_electrode: str | None = Field(
        default=None,
        description=(
            "Name of the electrode anchored at the start point, or ``None`` "
            "for a purely geometric conductor."
        ),
    )
    end_electrode: str | None = Field(
        default=None,
        description="Name of the electrode anchored at the end point.",
    )
    conductor_type: ConductorType = Field(default="generic")
    wire_radius: float = Field(default=0.005, gt=0.0, allow_inf_nan=False)
    resistivity: float = Field(default=1.68e-8, gt=0.0, allow_inf_nan=False)
    cross_section: float | Literal["from_radius"] | None = Field(
        default=None,
        description=(
            "Conductor cross section in m². ``None`` keeps the historic "
            "ideal-galvanic short. Set a finite value to enable the "
            'finite-impedance branch model. ``"from_radius"`` resolves '
            "to π · wire_radius²."
        ),
    )
    discretize_segment_length: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Maximum segment length used to *discretise* the conductor "
            "for the distributed-conductor model (ADR-0003). ``None`` "
            "keeps the conductor lumped — a single segment that exchanges "
            "current with the soil only at its end electrodes. A finite "
            "value splits the conductor into ``n = ceil(length / "
            "discretize_segment_length)`` sub-segments, each with its own "
            "longitudinal current and (if ``coupling_to_soil == "
            "'galvanic'``) its own midpoint leakage."
        ),
    )
    coupling_to_soil: SoilCoupling = Field(
        default="isolated",
        description=(
            "How the conductor exchanges current with the soil along its "
            'length. ``"isolated"`` (default) — no leakage along the '
            "wire; the only earth contact is via the end electrodes. "
            '``"galvanic"`` — every segment leaks current into the soil '
            "through the same Green's-function kernel as the electrode "
            "segments. For a buried bare-copper conductor or an exposed "
            'cable shield, set ``"galvanic"``; for an insulated cable '
            '(PEN inside NAYY) keep ``"isolated"``.'
        ),
    )
    inductance_model: InductanceModel | None = Field(
        default=None,
        description=(
            "Model used for the longitudinal-segment inductance "
            "(ADR-0004). ``None`` (default) keeps the system purely "
            "resistive — DC behaviour, frequency-independent. "
            '``"neumann"`` activates the Neumann double-line '
            "integral for self- and mutual-inductance between every "
            "pair of inductive branches — the segments of a distributed "
            "conductor or, for a lumped or single-segment conductor, one "
            "branch from start to end (review pass 10, 10.24) — with the "
            "thin-wire self-formula (DC internal inductance included) "
            "and the earth-return model of the engine "
            "(``Engine.earth_inductive_model``: Sommerfeld, Carson or the "
            "perfect mirror). The longitudinal-branch impedance becomes "
            "$Z_b = R + j\\omega L + \\Delta Z(\\omega)$, evaluated per "
            "frequency."
        ),
    )
    lumped_series_resistance_ohm: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Optional override for the conductor's total series "
            "resistance. ``None`` (default) keeps the geometric "
            "formula $R = \\rho_\\text{mat}\\, L / A$. When set, "
            ":attr:`series_resistance` returns this value verbatim, "
            "ignoring resistivity and cross section. Introduced for "
            "the V1 concrete-shell path of ADR-0012, where each "
            "foundation electrode's lumped Sunde-shell impedance is "
            "injected on the PEN service drop without having to fake "
            "geometric properties. Has no effect when the conductor "
            "is in the ideal-galvanic-short branch (i.e. "
            ":attr:`cross_section` is ``None``); pass a finite "
            "``cross_section`` together with this field to put the "
            "conductor in the finite-impedance branch model."
        ),
    )
    resistance_model: ResistanceModel = Field(
        default="dc",
        description=(
            "Frequency dependence of the conductor's own series impedance "
            '(review pass 10, finding 10.37). ``"dc"`` (default) keeps '
            "$R = \\rho L/A$ and the uniform-current internal inductance "
            "$\\mu_0/(8\\pi)$ per metre at every frequency. "
            '``"skin"`` uses the exact internal impedance of a solid round '
            "conductor with the same cross section "
            "(:meth:`internal_impedance_per_length`): 150 mm² Al has "
            "$R_\\text{ac}/R_\\text{dc}$ = 1.23 at 550 Hz and 1.55 at "
            "1 kHz. Needs a finite ``cross_section`` and no "
            "``lumped_series_resistance_ohm``. Stranded conductors are "
            "approximated by the solid conductor of equal cross section; "
            "tubular screens are not modelled."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_default_resistance_model(self, handler):
        """Serialise ``resistance_model`` only when it is not ``"dc"``.

        Keeps the JSON of every existing world byte-identical and loadable
        by earlier releases (``extra="forbid"``); a ``"skin"`` conductor
        still round-trips.
        """
        data = handler(self)
        if isinstance(data, dict) and data.get("resistance_model") == "dc":
            del data["resistance_model"]
        return data

    # ------------------------------------------------------------------
    # validation (review pass 10, findings 10.43 / 10.67)
    # ------------------------------------------------------------------

    @field_validator("start", "end")
    @classmethod
    def _finite_point(cls, value: Point3D, info) -> Point3D:
        if not all(math.isfinite(float(c)) for c in value):
            raise ValueError(
                f"{info.field_name} must have finite coordinates (m), got {value!r}"
            )
        return value

    @field_validator("cross_section")
    @classmethod
    def _plausible_cross_section(cls, value):
        if value is None or value == "from_radius":
            return value
        area = float(value)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(
                "cross_section must be a positive, finite area in m² "
                f"(e.g. 50e-6 for 50 mm²), got {value!r}"
            )
        if area > MAX_PLAUSIBLE_CROSS_SECTION:
            warnings.warn(
                f"cross_section = {area:g} m² is larger than "
                f"{MAX_PLAUSIBLE_CROSS_SECTION:g} m² — was it given in mm²? "
                f"{area:g} mm² is {area * 1e-6:g} m². As given, the conductor "
                "is practically an ideal short.",
                CrossSectionUnitWarning,
                stacklevel=2,
            )
        return value

    @model_validator(mode="after")
    def _consistent_impedance_model(self) -> Conductor:
        if self.discretize_segment_length is not None and self.is_ideal():
            raise ValueError(
                f"Conductor {self.name!r}: a distributed conductor "
                f"(discretize_segment_length={self.discretize_segment_length:g} m) "
                "needs a finite series resistance, but cross_section is "
                f"{self.cross_section!r} (series resistance "
                f"{self.series_resistance:.3g} ohm, at or below the ideal "
                "threshold of 1e-6 ohm). Its segment chain would consist of "
                "zero-resistance branches inside one fused cluster (a "
                "singular system). Set cross_section in m² (e.g. 50e-6), or "
                "drop discretize_segment_length to keep the conductor an "
                "ideal bond."
            )
        if self.resistance_model == "skin":
            if self.effective_cross_section is None:
                raise ValueError(
                    f"Conductor {self.name!r}: resistance_model='skin' needs a "
                    "finite cross_section (m²)."
                )
            if self.lumped_series_resistance_ohm is not None:
                raise ValueError(
                    f"Conductor {self.name!r}: resistance_model='skin' cannot be "
                    "combined with lumped_series_resistance_ohm (a lumped "
                    "override has no conductor geometry)."
                )
            if self.conductor_type == "cable_shield":
                # user decision Q33: a shield is a thin-walled tube, which the
                # solid-round Bessel solution does not describe.
                raise ValueError(
                    f"Conductor {self.name!r}: resistance_model='skin' models "
                    "a solid round conductor; a 'cable_shield' is a thin-walled "
                    "tube without the internal inductance of a solid wire, whose "
                    "skin effect is not modelled. Use resistance_model='dc'."
                )
        return self

    @property
    def is_shield(self) -> bool:
        """Whether the conductor is a cable shield (``"cable_shield"``).

        A shield is a thin-walled tube: no internal inductance of a solid
        wire, and a thinner conductor on its axis is its core (user
        decisions Q32 and Q33 of 2026.11).
        """
        return self.conductor_type == "cable_shield"

    @property
    def length(self) -> float:
        """Euclidean length of the conductor in m."""
        sx, sy, sz = self.start
        ex, ey, ez = self.end
        return ((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5

    @property
    def effective_cross_section(self) -> float | None:
        """Resolved cross section in $\\mathrm{m}^2$ or ``None``.

        Returns
        -------
        float or None
            ``None`` when :attr:`cross_section` is ``None``
            (ideal-galvanic mode). Otherwise the explicit cross section
            in m² (``"from_radius"`` resolved to
            $\\pi\\, r_\\text{wire}^2$).
        """
        if self.cross_section is None:
            return None
        if self.cross_section == "from_radius":
            return math.pi * self.wire_radius**2
        return float(self.cross_section)

    @property
    def series_resistance(self) -> float:
        """Series DC resistance $R_\\text{ser}$ in Ω.

        Resolution order:

        1. If :attr:`lumped_series_resistance_ohm` is set (ADR-0012 V1
           concrete-shell path or any user-supplied lumped override),
           return that value verbatim.
        2. Otherwise compute the geometric formula
           $R = \\rho_\\text{mat}\\, L / A$ from
           :attr:`resistivity`, :attr:`length` and
           :attr:`effective_cross_section`.
        3. If :attr:`cross_section` is ``None``, the conductor is in
           the ideal-galvanic-short branch and the series resistance
           is ``0.0``.

        Returns
        -------
        float
            Total series resistance in Ω consumed by the solver's
            distributed-conductor model (ADR-0003).
        """
        if self.lumped_series_resistance_ohm is not None:
            return float(self.lumped_series_resistance_ohm)
        A = self.effective_cross_section
        if A is None:
            return 0.0
        return float(self.resistivity * self.length / A)

    def is_ideal(self, threshold: float = _IDEAL_RESISTANCE_THRESHOLD) -> bool:
        """Return ``True`` when the conductor acts as a galvanic short.

        Parameters
        ----------
        threshold
            Resistance threshold in Ω. Conductors at or below this
            value are treated as ideal by the solver. Default
            ``1e-6 Ω``.
        """
        if self.cross_section is None:
            return True
        return self.series_resistance <= threshold

    def internal_impedance_per_length(
        self, frequency: float | np.ndarray
    ) -> complex | np.ndarray:
        r"""Internal series impedance per unit length in Ω/m.

        The conductor's own longitudinal impedance $Z'_\text{int} =
        R'(f) + j\omega L'_\text{int}(f)$, excluding the external
        (thin-wire / earth-return) inductance of
        :mod:`groundfield.coupling`.

        - ``resistance_model="dc"``: $R' = R_\text{ser}/L$ and the
          uniform-current internal inductance $\mu_0/(8\pi)$ — the
          convention of
          :func:`groundfield.coupling.inductance.thin_wire_self_inductance`
          with ``include_internal=True``; a ``"cable_shield"`` (a
          thin-walled tube) has none (user decision Q33). An ideal
          conductor has $R' = 0$.
        - ``resistance_model="skin"``: the exact solid-round Bessel
          solution for the radius $\sqrt{A/\pi}$ of the conducting
          cross section
          (:func:`~groundfield.conductors.skin.solid_round_internal_impedance`).

        Parameters
        ----------
        frequency
            Frequency in Hz (scalar or array, ``>= 0``).

        Returns
        -------
        complex or np.ndarray
            Internal impedance per unit length in Ω/m.
        """
        f = np.asarray(frequency, dtype=float)
        if np.any(f < 0.0) or not np.all(np.isfinite(f)):
            raise ValueError("frequency must be finite and >= 0")
        if self.resistance_model == "skin":
            area = self.effective_cross_section
            assert area is not None  # guaranteed by the validator
            return solid_round_internal_impedance(
                frequency,
                radius=math.sqrt(area / math.pi),
                resistivity=self.resistivity,
            )
        length = self.length
        r_per_m = self.series_resistance / length if length > 0.0 else 0.0
        l_int = 0.0 if self.is_shield else _MU_0 / (8.0 * math.pi)
        z = r_per_m + 1j * 2.0 * math.pi * f * l_int
        if np.ndim(frequency) == 0:
            return complex(z)
        return z

    def skin_correction_per_length(
        self, frequency: float | np.ndarray
    ) -> complex | np.ndarray:
        r"""Skin-effect correction to the DC convention, in Ω/m.

        $\Delta Z' = Z'_\text{int}(f) - [R'_\text{dc} + j\omega\mu_0/(8\pi)]$:
        what a solver that assembles $R_\text{dc}$ and the DC internal
        inductance (``include_internal=True``) has to add per metre of
        branch length. Zero for ``resistance_model="dc"``.

        Parameters
        ----------
        frequency
            Frequency in Hz (scalar or array, ``>= 0``).

        Returns
        -------
        complex or np.ndarray
            Correction per unit length in Ω/m.
        """
        f = np.asarray(frequency, dtype=float)
        if np.any(f < 0.0) or not np.all(np.isfinite(f)):
            raise ValueError("frequency must be finite and >= 0")
        if self.resistance_model == "dc":
            return 0j if np.ndim(frequency) == 0 else np.zeros(f.shape, dtype=complex)
        area = self.effective_cross_section
        assert area is not None
        z = self.internal_impedance_per_length(frequency)
        z_dc = self.resistivity / area + 1j * 2.0 * math.pi * f * _MU_0 / (
            8.0 * math.pi
        )
        if np.ndim(frequency) == 0:
            return complex(z - z_dc)
        return z - z_dc

    @property
    def is_distributed(self) -> bool:
        """``True`` iff the conductor is split into sub-segments.

        A distributed conductor has a finite
        :attr:`discretize_segment_length` — the solver builds one
        longitudinal segment per sub-piece. ``False`` keeps the
        conductor lumped (single branch between the two end electrodes).
        """
        return self.discretize_segment_length is not None

    @property
    def n_segments(self) -> int:
        """Number of segments produced by the discretiser.

        Returns ``1`` for a lumped conductor. Otherwise
        $n = \\lceil L / \\Delta s \\rceil$ with $L$ the
        Euclidean length and $\\Delta s =
        $ :attr:`discretize_segment_length`.
        """
        if self.discretize_segment_length is None:
            return 1
        n = int(math.ceil(self.length / self.discretize_segment_length))
        return max(1, n)
