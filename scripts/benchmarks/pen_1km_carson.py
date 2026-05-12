"""1 km PEN benchmark — self and mutual impedance with Carson correction.

Stand-alone tool that answers two AP1 questions directly:

1. *What is the self-impedance per unit length of a 1 km bare-copper PEN
   conductor at depth 0.6 m above a homogeneous earth?*
2. *What is the mutual impedance between that PEN and a parallel
   measurement lead at separation* $d$, *across frequency* $f$?

The script runs every distributed-capable backend
(``image``, ``mom``, ``cim``, ``bem``) with
``earth_inductive_model = "carson_series"`` and prints a Markdown
table with cluster impedance per backend at four frequencies, plus
a side-by-side comparison against the closed-form Oeding+Carson
textbook expression.

Usage
-----

::

    poetry run python -m scripts.benchmarks.pen_1km_carson
    poetry run python -m scripts.benchmarks.pen_1km_carson --rho 200 --separation 100

Run flags
---------
``--rho``     Earth resistivity in $\\Omega\\,\\mathrm{m}$ (default 100).
``--length``  PEN conductor length in m (default 1000).
``--separation``  Horizontal separation between source and measurement
              lead in m (default 50). Set to 0 to skip the mutual
              section.
``--depth``   Depth of the conductors in m (default 0.6).
``--seg``     Sub-segment length for the discretiser (default 50 m).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import groundfield as gf
from groundfield.coupling.carson import (
    MU_0,
    carson_mutual_correction,
    carson_self_correction,
    skin_depth,
)


# ---------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------


def _build_world(
    *,
    length: float,
    separation: float,
    depth: float,
    rho_earth: float,
    seg: float,
) -> gf.World:
    """Build a benchmark world: source PEN ± measurement PEN + earth rods.

    Geometry (z > 0 is into the soil):
    - Two earth rods ``g1, g2`` anchor the source PEN's ends.
    - Two earth rods ``m1, m2`` anchor the measurement PEN's ends.
    - The measurement leg is omitted when ``separation == 0``.
    """
    soil = gf.HomogeneousSoil(resistivity=rho_earth)
    w = gf.create_world(soil=soil)
    rod_len = 2.0
    rod_r = 0.0075
    # Source PEN
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5),
                        length=rod_len, wire_radius=rod_r)
    gf.create_electrode(w, "rod", name="g2", position=(length, 0.0, 0.5),
                        length=rod_len, wire_radius=rod_r)
    gf.create_conductor(
        w, name="src_pen", start="g1", end="g2",
        conductor_type="bare_copper",
        wire_radius=0.004,
        resistivity=2.82e-8,         # copper
        cross_section=50.0e-6,       # 50 mm² (typical NS PEN)
        discretize_segment_length=seg,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    if separation > 0.0:
        gf.create_electrode(w, "rod", name="m1",
                            position=(0.0, separation, 0.5),
                            length=rod_len, wire_radius=rod_r)
        gf.create_electrode(w, "rod", name="m2",
                            position=(length, separation, 0.5),
                            length=rod_len, wire_radius=rod_r)
        gf.create_conductor(
            w, name="meas_pen", start="m1", end="m2",
            conductor_type="bare_copper",
            wire_radius=0.004,
            resistivity=2.82e-8,
            cross_section=50.0e-6,
            discretize_segment_length=seg,
            coupling_to_soil="galvanic",
            inductance_model="neumann",
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


# ---------------------------------------------------------------------
# Closed-form per-unit-length impedance (Oeding + Carson)
# ---------------------------------------------------------------------


def textbook_self_impedance_per_m(
    *, omega: float, depth: float, wire_radius: float,
    rho_material: float, cross_section: float, sigma_earth: float,
) -> complex:
    """Per-m self-impedance of a horizontal buried PEN.

    Decomposition (ADR-0005, Tleis 2008 §3.4):

    .. code-block:: text

        Z' = R_dc           + jωL_perfect_mirror + (R_g + jX_g)
            ohmic            buried-wire image      Carson correction

    The buried-wire image term uses the classical thin-wire formula
    $L = (\\mu_0 / 2\\pi) [\\ln(2 \\ell / a) - 1]$ with $\\ell$ taken
    as 1 m (per-unit-length expression). The Carson correction comes
    from :func:`groundfield.coupling.carson.carson_self_correction`.
    """
    R_dc = rho_material / cross_section
    # Per-m loop self-inductance (Oeding 9.13c with d = 2*depth as
    # equivalent loop spacing): L' = (mu_0 / 2pi) * ln(2*depth / r')
    # with r' = wire_radius * exp(-1/4) (internal-field correction).
    r_eff = wire_radius * math.exp(-0.25)
    L_per_m_perfect = (MU_0 / (2.0 * math.pi)) * math.log(2.0 * depth / r_eff)
    # Carson correction.
    Z_carson = carson_self_correction(omega, depth, sigma_earth)
    return complex(R_dc + Z_carson.real, omega * L_per_m_perfect + Z_carson.imag)


def textbook_mutual_impedance_per_m(
    *, omega: float, depth_i: float, depth_j: float,
    horizontal_distance: float, sigma_earth: float,
) -> complex:
    """Per-m mutual impedance between two parallel buried wires.

    .. code-block:: text

        Z'_12 = jωL_perfect_mirror + (R_g + jX_g)
              = (jω·μ_0 / 2π) · ln(D'/D) + Carson(omega, hi, hj, d, sigma)

    with $D = $ direct distance and $D' = $ image distance.
    """
    h_sum = depth_i + depth_j
    D_prime = math.hypot(h_sum, horizontal_distance)
    D = math.hypot(depth_i - depth_j, horizontal_distance)
    if D == 0.0:
        D = 1e-9
    L_per_m_perfect = (MU_0 / (2.0 * math.pi)) * math.log(D_prime / D)
    Z_carson = carson_mutual_correction(
        omega, depth_i, depth_j, horizontal_distance, sigma_earth,
    )
    return complex(Z_carson.real, omega * L_per_m_perfect + Z_carson.imag)


# ---------------------------------------------------------------------
# Cross-engine sweep
# ---------------------------------------------------------------------


_BACKENDS = ("image", "mom", "cim", "bem")


def run_cross_engine_table(
    *, length: float, separation: float, depth: float,
    rho_earth: float, seg: float, frequencies: list[float],
) -> str:
    """Build the Markdown table comparing engines and theory."""
    sigma = 1.0 / rho_earth
    out = []
    out.append(f"# 1 km PEN benchmark (Carson on)")
    out.append("")
    out.append(
        f"- Length: **{length:.0f} m**, depth: **{depth:.2f} m**, "
        f"separation: **{separation:.0f} m**"
    )
    out.append(
        f"- Earth: ρ = {rho_earth:.0f} Ω·m  →  σ = {sigma:.4f} S/m"
    )
    out.append("")
    out.append("Skin depth δ(ω) = √(2/(ω·μ₀·σ)):")
    out.append("")
    out.append("| f [Hz] | δ [m] | a (self, h=depth) |")
    out.append("|---:|---:|---:|")
    for f in frequencies:
        omega = 2.0 * math.pi * f
        delta = skin_depth(omega, sigma)
        a_self = 2.0 * depth * math.sqrt(omega * MU_0 * sigma)
        out.append(f"| {f:.1f} | {delta:.1f} | {a_self:.4f} |")
    out.append("")

    # Cross-engine self impedance table (per-conductor cluster impedance).
    out.append("## Source-rod cluster impedance Z = U_g1 / I_g1")
    out.append("")
    out.append("Backend rows; frequency columns.")
    out.append("")
    header = "| Backend | " + " | ".join(f"{f:g} Hz" for f in frequencies) + " |"
    sep = "|---|" + "---|" * len(frequencies)
    out.append(header)
    out.append(sep)

    for backend in _BACKENDS:
        cells = [backend]
        for f in frequencies:
            try:
                w = _build_world(length=length, separation=separation,
                                 depth=depth, rho_earth=rho_earth, seg=seg)
                eng = gf.create_engine(
                    backend=backend, segment_length=seg, frequencies=[f],
                    earth_inductive_model="carson_series",
                )
                t0 = time.perf_counter()
                res = eng.solve(w)
                dt = time.perf_counter() - t0
                U = res.electrode_potentials["g1"][0]
                I = res.electrode_currents["g1"][0]
                if abs(I) == 0.0:
                    cells.append("(I = 0)")
                else:
                    Z = U / I
                    cells.append(
                        f"{Z.real:.4f} + j {Z.imag:.4f} Ω ({dt:.2f} s)"
                    )
            except Exception as exc:  # noqa: BLE001 — informational
                cells.append(f"err: {exc.__class__.__name__}")
        out.append("| " + " | ".join(cells) + " |")

    out.append("")
    out.append("## Per-unit-length closed form (Oeding + Carson)")
    out.append("")
    out.append("| f [Hz] | Z'_self [Ω/km] | Z'_12 [Ω/km] (sep ≠ 0) |")
    out.append("|---:|---|---|")
    for f in frequencies:
        omega = 2.0 * math.pi * f
        Z_self = textbook_self_impedance_per_m(
            omega=omega, depth=depth, wire_radius=0.004,
            rho_material=2.82e-8, cross_section=50.0e-6, sigma_earth=sigma,
        ) * 1000.0
        if separation > 0.0:
            Z_12 = textbook_mutual_impedance_per_m(
                omega=omega, depth_i=depth, depth_j=depth,
                horizontal_distance=separation, sigma_earth=sigma,
            ) * 1000.0
            mut_cell = f"{Z_12.real:.4f} + j {Z_12.imag:.4f}"
        else:
            mut_cell = "—"
        out.append(
            f"| {f:.1f} | {Z_self.real:.4f} + j {Z_self.imag:.4f} | {mut_cell} |"
        )
    out.append("")
    out.append("> Closed-form decomposition: "
               "$Z' = R_\\text{dc} + j\\omega L_\\text{perfect mirror} + "
               "\\Delta Z_\\text{Carson}(\\omega, h_i, h_j, d, \\sigma)$.")
    out.append("> See ADR-0005 for the derivation and limitations.")

    return "\n".join(out)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="1 km PEN benchmark with Carson earth-return correction"
    )
    p.add_argument("--length", type=float, default=1000.0)
    p.add_argument("--separation", type=float, default=50.0)
    p.add_argument("--depth", type=float, default=0.6)
    p.add_argument("--rho", type=float, default=100.0)
    p.add_argument("--seg", type=float, default=50.0)
    p.add_argument(
        "--frequencies", type=float, nargs="+",
        default=[50.0, 150.0, 500.0, 1000.0],
    )
    args = p.parse_args(argv)

    md = run_cross_engine_table(
        length=args.length, separation=args.separation,
        depth=args.depth, rho_earth=args.rho, seg=args.seg,
        frequencies=list(args.frequencies),
    )
    print(md)


if __name__ == "__main__":
    main()
