"""Tolerance table of the engine benchmark suite (B1–B13).

One entry per benchmark and backend: the bound the benchmark test
asserts, how the deviation is measured, the value measured when the
entry was written (review pass 10, 2026-11-07, default engine settings
with the point-source kernel) and a note (reference, target, finding).
The tests in ``test_engine_benchmarks.py`` read the bounds from here, so
this table is the single source the documentation quotes::

    python -m tests.benchmarks.tolerances     # prints the Markdown table

A backend that is not a valid model for a benchmark is listed with
``bound=None`` and the reason. An entry with ``xfail`` names the finding
whose fix is expected to meet the bound; the test is a strict xfail, so
the fix turns it into a failure that forces this table to be updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BENCHMARKS", "Benchmark", "Bound", "bound", "markdown"]


@dataclass(frozen=True)
class Bound:
    """One row of the tolerance table.

    Attributes
    ----------
    bound
        Asserted upper bound of the measure; ``None`` = backend excluded.
    measure
        What is bounded, e.g. ``"abs(Z/Z_ref - 1)"``.
    measured
        Value(s) measured when the entry was written.
    note
        Reference details, target value or the finding that changes it.
    xfail
        Finding id whose fix is expected to meet ``bound`` (the test is
        a strict xfail until then); empty when the bound holds.
    """

    bound: float | None
    measure: str
    measured: str
    note: str = ""
    xfail: str = ""


@dataclass(frozen=True)
class Benchmark:
    """A benchmark world, its reference and the bounds per backend."""

    title: str
    case: str
    reference: str
    bounds: dict[str, Bound] = field(default_factory=dict)


_REL = "abs(Z/Z_ref - 1)"

BENCHMARKS: dict[str, Benchmark] = {
    "B1": Benchmark(
        title="Surface disc",
        case="disc b = 1 m on the surface (a coin 10 µm thick), homogeneous 100 Ω·m",
        reference="ρ/(4b) = 25 Ω, Weber's disc, analytic",
        bounds={
            "fem": Bound(
                5e-4,
                _REL,
                "-3.1e-4 at levels 0/1 extrapolated; observed order 1.87",
                "discretisation of the conforming tensor mesh alone (user decision "
                "Q37; up to 2026.11 the hemisphere of the equivalent-hemisphere mesh)",
            ),
        },
    ),
    "B2": Benchmark(
        title="Rod, homogeneous soil",
        case="rod L = 3 m, a = 8 mm, top at the surface, 100 Ω·m, ds ∈ {0.5, 0.25, 0.125} m",
        reference=(
            "thin-wire limit 33.1957 Ω (independent analytic Galerkin solver, "
            "Richardson on 96/192 segments; reduced-kernel end effect "
            "leaves ~0.2 % uncertainty); Sunde 33.4927 Ω for orientation"
        ),
        bounds={
            "image": Bound(1.5e-2, _REL, "-0.30 / -0.63 / -0.79 %", "point kernel"),
            "mom": Bound(
                2.0e-2,
                _REL,
                "-0.63 / -1.10 / -1.39 %",
                "point-source near field drifts under refinement (10.31); "
                "target 0.5 % with an exact near field",
            ),
            "fem": Bound(
                1e-3,
                _REL,
                "-7.3e-5",
                "solid rod with flat ends (user decision Q37); the thin-wire limit "
                "itself is uncertain at a few 1e-4 (33.185 Ω at 384 segments)",
            ),
        },
    ),
    "B3": Benchmark(
        title="Ring, homogeneous soil",
        case="ring r = 2 m, a = 8 mm, z = 0.8 m, 100 Ω·m, ds ∈ {0.5, 0.25, 0.125} m",
        reference=(
            "uniformly loaded thin ring + air image, closed form with "
            "complete elliptic integrals: 12.4816 Ω (exact for the thin-wire "
            "model by symmetry; Dwight 12.5429 Ω)"
        ),
        bounds={
            "image": Bound(1.5e-2, _REL, "-1.18 %", "point kernel (10.28)"),
            "mom": Bound(1.5e-2, _REL, "-1.18 %", "uniform current by symmetry"),
            "fem": Bound(5e-4, _REL, "-5.7e-5", "solid torus (user decision Q37)"),
        },
    ),
    "B4": Benchmark(
        title="Two-layer Sommerfeld kernel",
        case=(
            "1050 points: h_1 ∈ {0.05 … 30} m, ρ₂/ρ₁ ∈ {1/2000 … 2000}, "
            "s ∈ {0 … 5000} m, five depth pairs"
        ),
        reference="exact Tagg/Sunde image series until |K|^m < 1e-17",
        bounds={
            "mom_sommerfeld": Bound(
                1e-9,
                "rel. kernel error, or SommerfeldConvergenceWarning",
                "max silent error 2.0e-10; 30 warnings, all at s = 5 km over h_1 = 5 cm",
                "never silent (10.47); no warning for h_1 >= 0.2 m",
            ),
        },
    ),
    "B5": Benchmark(
        title="Three-layer Sommerfeld kernel",
        case=(
            "duplicated stacks [ρ₁, ρ₂, ρ₂] and [ρ₁, ρ₁, ρ₂] (405 points each) "
            "and the genuine [100, 1000, 50] Ω·m, h = [2, 1] m"
        ),
        reference="n = 2 series; mpmath.quadosc of the remainder (30 digits)",
        bounds={
            "mom_sommerfeld": Bound(
                1e-10, "rel. kernel error", "2.7e-13 (duplicated) / 4e-16 (genuine)"
            ),
        },
    ),
    "B6": Benchmark(
        title="Top-layer electrodes, two-layer soil",
        case=(
            "ring r = 2 m at 0.8 m and rod 0–1.5 m; 2L(1000, 30, h₁ ∈ {5, 10, 30}), "
            "2L(30, 200, 5), 2L(100, 400, 2); ds = 0.25 m"
        ),
        reference="mom on the same discretisation",
        bounds={
            "image_2layer": Bound(
                1e-2,
                _REL,
                "ring 0 (uniform by symmetry), rod 0.25 … 0.37 %",
                "uniform-current ansatz",
            ),
            "mom_sommerfeld": Bound(1e-7, _REL, "<= 1.2e-8", "kernel check"),
            "fem": Bound(
                3e-2,
                _REL,
                "+0.57 … +1.34 %",
                "conforming model (user decision Q37): the deficit of mom's point "
                "kernel (10.28)",
            ),
        },
    ),
    "B7": Benchmark(
        title="Rod through the layer interface",
        case="rod 0–3 m through h₁ = 2 m in 2L(1000, 30) and 2L(100, 1000); ds = 0.25 m",
        reference="mom: 25.6913 Ω and 64.0749 Ω",
        bounds={
            "mom_sommerfeld": Bound(1e-7, _REL, "4.4e-11 / 1.4e-10", "kernel check"),
            "image_2layer": Bound(
                2e-2,
                _REL,
                "+648 % / +89 %",
                "one uniform current across the interface (F02)",
                xfail="10.15",
            ),
            "fem": Bound(
                3e-2,
                _REL,
                "+0.35 / +1.12 %",
                "conforming model (user decision Q37; up to 2026.11 it raised, "
                "v0.15.0: +1092 %, silent): the deficit of mom's point kernel",
            ),
        },
    ),
    "B8": Benchmark(
        title="Two-port reciprocity",
        case=(
            "rod A 0–3 m and ring B r = 1.5 m at 0.8 m, 4 m apart; homogeneous, "
            "2L(1000, 30, 5) and the cross-layer 2L(1000, 30, 2); ds = 0.5 m"
        ),
        reference=(
            "Z_AB = Z_BA; floating-port mom 17.28197 Ω (2L h₁ = 5) and "
            "1.43970 Ω (h₁ = 2)"
        ),
        bounds={
            "image": Bound(1e-10, "abs(Z_AB/Z_BA - 1)", "3.6e-16"),
            "image_2layer": Bound(1e-7, "abs(Z_AB/Z_BA - 1)", "1.0e-15 / 1.05e-9"),
            "cim": Bound(1e-7, "abs(Z_AB/Z_BA - 1)", "as image_2layer (10.30)"),
            "mom": Bound(1e-7, "abs(Z_AB/Z_BA - 1)", "2e-16 / 7.2e-9 (10.29)"),
            "bem": Bound(1e-7, "abs(Z_AB/Z_BA - 1)", "6e-16 / 7.2e-9 (10.29)"),
            "mom_sommerfeld": Bound(1e-7, "abs(Z_AB/Z_BA - 1)", "4e-16 / 0 (10.29)"),
            "fem": Bound(
                None,
                "abs(Z_AB/Z_BA - 1)",
                "refused",
                "excluded: rod and ring on different axes, which the axisymmetric "
                "model refuses (user decision Q37); the port matrix of coaxial "
                "electrodes is symmetric to round-off",
            ),
        },
    ),
    "B9": Benchmark(
        title="Limits of the layered kernels",
        case=(
            "ring r = 2 m at 0.8 m: TwoLayer(ρ, ρ, h) and MultiLayer([ρ, ρ, ρ]) "
            "against homogeneous; 2L(1000, 30) with h₁ = 4 mm and 4 km against "
            "homogeneous 30 / 1000 Ω·m; ds = 0.5 m"
        ),
        reference="the homogeneous result of the same backend",
        bounds={
            "equal_rho_integral": Bound(
                1e-10,
                _REL,
                "0 (canonical stack, 10.48)",
                "image_2layer, mom, mom_sommerfeld, cim, bem",
            ),
            "equal_rho_fem": Bound(
                1e-12,
                _REL,
                "0 (canonical stack, 10.48)",
                "the conforming model meshes the canonical stack (user decision Q37)",
            ),
            "layer_limit_integral": Bound(
                1e-3, _REL, "5.1e-4 (4 mm) / 2.0e-4 (4 km)", "physical limit"
            ),
            "layer_limit_fem": Bound(
                1e-3,
                _REL,
                "5.0e-4 / 2.0e-4",
                "physical limit (user decision Q37)",
            ),
        },
    ),
    "B10": Benchmark(
        title="Ring with three bonded rods",
        case="ring r = 2 m at 0.5 m + three 4 m rods; homogeneous 100 and 2L(1000, 30, 5); ds = 0.25 m",
        reference="mom on the same discretisation",
        bounds={
            "image": Bound(
                3e-2,
                _REL,
                "+1.43 % / +2.83 %",
                "uniform current per electrode (10.22); target 2 %",
            ),
            "fem": Bound(
                None,
                _REL,
                "refused",
                "excluded: rods off the ring's axis, which the axisymmetric model "
                "refuses (user decision Q37; up to 2026.11 -35 % / -45 %, radii summed)",
            ),
        },
    ),
    "B11": Benchmark(
        title="Grid refinement",
        case="grid 40 × 40 m, 4 × 4 meshes, z = 0.8 m, 2L(1000, 30, 5), ds ∈ {5, 2.5, 1.25} m",
        reference="the backend itself at ds / 2 (convergence contract)",
        bounds={
            # user decision Q30: one owner per wire line moves the coarse
            # step of the point kernel from 0.43 % to 0.54 % (evidence w1sol
            # b11_refinement.out; the default line kernel 0.12 % / 0.09 %).
            "image_2layer": Bound(
                6e-3,
                "abs(Z(ds)/Z(ds/2) - 1)",
                "0.54 % / 0.35 %",
                "not monotone (point kernel); 0.43 % / 0.39 % with one owner "
                "for the grid before user decision Q30",
            ),
            "mom": Bound(1e-2, "abs(Z(ds)/Z(ds/2) - 1)", "0.86 % / 0.14 %"),
            "mom_sommerfeld": Bound(
                1e-7, "abs(Z/Z_mom - 1) at ds = 5 m", "7.5e-10", "kernel check"
            ),
        },
    ),
    "B12": Benchmark(
        title="Three-layer grid",
        case="grid of B11 in 3L[1000, 30, 200], h = [5, 10] m, ds ∈ {2.5, 1.25} m",
        reference="mom_sommerfeld with every distinct pair by quadrature (tabulate=False)",
        bounds={
            "mom_sommerfeld": Bound(
                1e-6,
                "max abs(Z_ij,tab / Z_ij - 1) and abs(Z_g1,tab / Z_g1 - 1)",
                "0 / 0 (ds = 2.5 m: no group large enough); 6.9e-9 / 1.5e-10 "
                "(ds = 1.25 m)",
                "depth-group tabulation of the reflected kernel (10.61)",
            ),
        },
    ),
    "B13": Benchmark(
        title="Ring with an inductive 400 m line",
        case=(
            "ring r = 2 m at 0.8 m, isolated 400 m line (95 mm², Neumann, "
            "Sommerfeld earth return) to a 3 m rod; f ∈ {50, 1000} Hz; ds = 1 m"
        ),
        reference=(
            "compare_engines(image, mom), complex Z at every frequency; the "
            "line's per-metre series impedance against the Pollaczek "
            "conductor-earth loop impedance R'_dc + Z'_P (references.pollaczek, "
            "GMR = a e^{-1/4})"
        ),
        bounds={
            "image_vs_mom": Bound(
                5e-2,
                "compare_engines row deviation (Re and Im)",
                "0.09 % (worst of 8 rows)",
                "inductive branch shared; galvanic part differs by the ansatz",
            ),
            "pollaczek_x": Bound(
                5e-3,
                "abs(X'/X'_P - 1), X' = Im(Z_1600 m - Z_800 m) / 800 m at 1 kHz",
                "4.1e-6",
                "the two-length difference cancels the end effects; the 400 m "
                "line itself is 3.7 % (1 kHz) and 10.8 % (50 Hz) below "
                "l X'_P (finite length, l / delta = 2.5 and 0.56)",
            ),
            "pollaczek_r": Bound(
                2e-2,
                "abs(R'/R'_P - 1), same difference",
                "-1.04 %",
                "converges from below with the length (-4.5 % for 400/800 m); "
                "the 400 m line is 34 % (1 kHz) and 17 % (50 Hz) below l R'_P",
            ),
        },
    ),
}


def bound(benchmark: str, backend: str) -> Bound:
    """Look up one entry of the table (raises ``KeyError`` if absent)."""
    return BENCHMARKS[benchmark].bounds[backend]


def markdown() -> str:
    """The table as Markdown, for the validation page of the docs."""
    lines = [
        "| ID | Case | Reference | Backend | Bound | Measure | Measured | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, bench in BENCHMARKS.items():
        for backend, b in bench.bounds.items():
            value = "excluded" if b.bound is None else f"{b.bound:.2g}"
            note = b.note + (f" (xfail until {b.xfail})" if b.xfail else "")
            lines.append(
                f"| {key} | {bench.case} | {bench.reference} | `{backend}` | "
                f"{value} | {b.measure} | {b.measured} | {note} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(markdown())
