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

Coupling models (inductive Neumann coupling, Carson correction,
capacitive coupling) live in :mod:`groundfield.coupling` and are
**not yet** included here — only the resistive series impedance is.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Conductor", "ConductorType", "SoilCoupling", "InductanceModel"]

Point3D = tuple[float, float, float]

ConductorType = Literal[
    "pen",            # PEN conductor in low-voltage networks
    "cable_shield",   # cable shield
    "bare_copper",    # bare copper wire
    "overhead",       # overhead line / earth wire
    "generic",        # fallback
]

SoilCoupling = Literal[
    "isolated",   # cable / PEN inside an insulating jacket — no leakage
    "galvanic",   # bare copper / exposed shield — leakage along the wire
]

InductanceModel = Literal[
    "neumann",    # Neumann double-line integral, perfect-mirror earth
]

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
        Type tag from :data:`ConductorType`. Drives later choice of
        coupling and impedance model.
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
    wire_radius: float = Field(default=0.005, gt=0.0)
    resistivity: float = Field(default=1.68e-8, gt=0.0)
    cross_section: float | Literal["from_radius"] | None = Field(
        default=None,
        description=(
            "Conductor cross section in m². ``None`` keeps the historic "
            "ideal-galvanic short. Set a finite value to enable the "
            "finite-impedance branch model. ``\"from_radius\"`` resolves "
            "to π · wire_radius²."
        ),
    )
    discretize_segment_length: float | None = Field(
        default=None,
        gt=0.0,
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
            "length. ``\"isolated\"`` (default) — no leakage along the "
            "wire; the only earth contact is via the end electrodes. "
            "``\"galvanic\"`` — every segment leaks current into the soil "
            "through the same Green's-function kernel as the electrode "
            "segments. For a buried bare-copper conductor or an exposed "
            "cable shield, set ``\"galvanic\"``; for an insulated cable "
            "(PEN inside NAYY) keep ``\"isolated\"``."
        ),
    )
    inductance_model: InductanceModel | None = Field(
        default=None,
        description=(
            "Model used for the longitudinal-segment inductance "
            "(ADR-0004). ``None`` (default) keeps the system purely "
            "resistive — DC behaviour, frequency-independent. "
            "``\"neumann\"`` activates the Neumann double-line "
            "integral for self- and mutual-inductance between every "
            "pair of distributed-conductor segments, using the "
            "thin-wire self-formula and a perfect-mirror earth for "
            "the magnetic image. The longitudinal-branch impedance "
            "becomes $Z_b = R + j\\omega L$, evaluated per "
            "frequency."
        ),
    )
    lumped_series_resistance_ohm: float | None = Field(
        default=None,
        ge=0.0,
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
            return math.pi * self.wire_radius ** 2
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

    def is_ideal(
        self, threshold: float = _IDEAL_RESISTANCE_THRESHOLD
    ) -> bool:
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
