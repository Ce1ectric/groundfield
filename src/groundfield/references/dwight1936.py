"""Grounding resistances of simple geometries after Dwight (1936).

Source
------
Dwight, H. B.: *Calculation of Resistances to Ground*. Electrical
Engineering / AIEE Transactions, Vol. 55, No. 12, December 1936,
pp. 1319-1328 (Paper 36-129). In particular the summary Table I on
p. 1325.

Conventions
-----------
- All lengths are given in **metres** (Dwight uses cm; the SI
  conversion is lossless because only dimensionless ratios and the
  prefactor $\\rho/(\\dots\\pi L)$ appear).
- $\\rho$ is the resistivity of the homogeneous soil in
  $\\Omega\\,\\mathrm{m}$.
- For buried electrodes Dwight's Table I uses $s$ for the *image
  distance*, i.e. **twice the burial depth**: $s = 2\\,t$, where
  $t$ is the depth of the centre of the electrode below the
  surface. The functions defined here take ``depth`` (= $t$) and
  convert internally to $s = 2\\,t$.
- Wire parameters: ``a`` is the **wire radius** (not the diameter);
  for geometries that use the wire diameter $d$ in Table I (the
  ring), the conversion ``d = 2*a`` applies.

Range of validity
-----------------
All formulas assume **homogeneous soil** in the low-frequency
(quasi-static) regime. Dwight quotes a few-percent accuracy for most
expressions (Howe's average-potential method). The ``rod`` formula is
within < 1 %; the star geometries within a few percent
(Dwight 1936, p. 1324, "equations 23 to 26").

Notes
-----
The functions are pure Python scalars (NumPy scalars also work). They
are intentionally not vectorised — comparison tests need clarity, not
performance.
"""

from __future__ import annotations

import math

__all__ = [
    "rod",
    "two_rods_far",
    "two_rods_close",
    "horizontal_wire",
    "right_angle_wire",
    "n_point_star",
    "buried_ring",
    "horizontal_strip",
    "horizontal_round_plate",
    "vertical_round_plate",
    "hemisphere",
]


# ---------------------------------------------------------------------
# Rod and rod pairs
# ---------------------------------------------------------------------


def rod(rho: float, length: float, radius: float) -> float:
    """Vertical driven rod (single ground rod), Eq. (13) / Table I.
    $$
    R = \\frac{\\rho}{2\\pi L}\\,\\bigl(\\ln\\tfrac{4L}{a} - 1\\bigr)
    $$
    Assumptions: rod starts at the soil surface and extends $L$
    downwards; the rod is treated together with its image in the
    half-space. Wire radius $a \\ll L$.
    """
    if length <= 0.0 or radius <= 0.0:
        raise ValueError("length and radius must be positive.")
    return rho / (2.0 * math.pi * length) * (math.log(4.0 * length / radius) - 1.0)


def two_rods_far(rho: float, length: float, radius: float, spacing: float) -> float:
    """Two parallel rods, axis spacing $s > L$. Table I, Eq. (20).
    $$
    R = \\frac{\\rho}{4\\pi L}\\bigl(\\ln\\tfrac{4L}{a} - 1\\bigr)
        + \\frac{\\rho}{4\\pi s}\\,
          \\Bigl(1 - \\tfrac{L^2}{3 s^2}
                    + \\tfrac{2 L^4}{5 s^4}\\Bigr)
    $$
    Assumption: ``spacing`` is the horizontal axis-to-axis distance.
    The series is an expansion for large ``s/L`` and is usable for
    $s/L \\gtrsim 1$.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    s = spacing
    R_self = rho / (4.0 * math.pi * length) * (math.log(4.0 * length / radius) - 1.0)
    series = 1.0 - (length ** 2) / (3.0 * s ** 2) + 2.0 * (length ** 4) / (5.0 * s ** 4)
    R_mutual = rho / (4.0 * math.pi * s) * series
    return R_self + R_mutual


def two_rods_close(rho: float, length: float, radius: float, spacing: float) -> float:
    """Two parallel rods, axis spacing $s < L$. Table I, Eq. (21).
    $$
    R = \\frac{\\rho}{4\\pi L}\\Bigl(\\ln\\tfrac{4L}{a}
        + \\ln\\tfrac{4L}{s} - 2 + \\tfrac{s}{2L}
        - \\tfrac{s^2}{16 L^2} + \\tfrac{s^4}{512 L^4}\\Bigr)
    $$
    Suitable for closely-spaced rods, $s \\ll L$.
    """
    s = spacing
    if s <= 0.0:
        raise ValueError("spacing must be positive.")
    inner = (
        math.log(4.0 * length / radius)
        + math.log(4.0 * length / s)
        - 2.0
        + s / (2.0 * length)
        - s ** 2 / (16.0 * length ** 2)
        + s ** 4 / (512.0 * length ** 4)
    )
    return rho / (4.0 * math.pi * length) * inner


# ---------------------------------------------------------------------
# Horizontal wires and right-angle bend
# ---------------------------------------------------------------------


def horizontal_wire(
    rho: float, length: float, radius: float, depth: float
) -> float:
    """Buried horizontal wire of total length $2L$, Table I.
    $$
    R = \\frac{\\rho}{4\\pi L}
        \\Bigl(\\ln\\tfrac{4L}{a} + \\ln\\tfrac{4L}{s}
               - 2 + \\tfrac{s}{2L}
               - \\tfrac{s^2}{16 L^2}
               + \\tfrac{s^4}{512 L^4}\\Bigr)
    $$
    Parameters
    ----------
    length
        ``L`` = half the wire length in m (total length is ``2*L``).
    radius
        Wire radius $a$ in m.
    depth
        Burial depth $t$ in m. The image distance is
        $s = 2\\,t$.
    """
    s = 2.0 * depth
    inner = (
        math.log(4.0 * length / radius)
        + math.log(4.0 * length / s)
        - 2.0
        + s / (2.0 * length)
        - s ** 2 / (16.0 * length ** 2)
        + s ** 4 / (512.0 * length ** 4)
    )
    return rho / (4.0 * math.pi * length) * inner


def right_angle_wire(
    rho: float, arm_length: float, radius: float, depth: float
) -> float:
    """Right-angle wire: two arms of length $L$, Table I, Eq. (22)."""
    L = arm_length
    s = 2.0 * depth
    inner = (
        math.log(2.0 * L / radius)
        + math.log(2.0 * L / s)
        - 0.2373
        + 0.2146 * (s / L)
        + 0.1035 * (s / L) ** 2
        - 0.0424 * (s / L) ** 4
    )
    return rho / (4.0 * math.pi * L) * inner


# ---------------------------------------------------------------------
# n-point stars
# ---------------------------------------------------------------------

# Coefficients of the series in Table I, Eqs. (23) to (26).
# Entry: (constant, c1·s/L, c2·(s/L)^2, c3·(s/L)^4)
# Prefactor: rho / (2 n pi L), where n is the number of arms.
_STAR_COEFFS: dict[int, tuple[float, float, float, float]] = {
    3: (1.071, -0.209, 0.238, -0.054),
    4: (2.912, -1.071, 0.645, -0.145),
    6: (6.851, -3.128, 1.758, -0.490),
    8: (10.98, -5.51, 3.26, -1.17),
}


def n_point_star(
    rho: float,
    arm_length: float,
    radius: float,
    depth: float,
    n_arms: int,
) -> float:
    """n-point star (star of $n$ equal-length arms). Table I.
    $$
    R = \\frac{\\rho}{2 n \\pi L}\\,
        \\Bigl(\\ln\\tfrac{2L}{a} + \\ln\\tfrac{2L}{s}
               + c_0 + c_1\\tfrac{s}{L}
               + c_2\\tfrac{s^2}{L^2}
               + c_3\\tfrac{s^4}{L^4}\\Bigr)
    $$
    Parameters
    ----------
    n_arms
        Number of arms. Supported values: 3, 4, 6, 8 (Dwight,
        Eqs. 23-26).
    arm_length
        Length of a single arm $L$ in m.
    depth
        Burial depth $t$; internally $s = 2\\,t$.
    """
    if n_arms not in _STAR_COEFFS:
        raise ValueError(
            f"Dwight 1936 only covers n_arms in {{3, 4, 6, 8}}, got {n_arms}."
        )
    L = arm_length
    s = 2.0 * depth
    c0, c1, c2, c3 = _STAR_COEFFS[n_arms]
    inner = (
        math.log(2.0 * L / radius)
        + math.log(2.0 * L / s)
        + c0
        + c1 * (s / L)
        + c2 * (s / L) ** 2
        + c3 * (s / L) ** 4
    )
    return rho / (2.0 * n_arms * math.pi * L) * inner


# ---------------------------------------------------------------------
# Ring
# ---------------------------------------------------------------------


def buried_ring(
    rho: float, ring_diameter: float, wire_diameter: float, depth: float
) -> float:
    """Buried wire ring, Table I, Eq. (29).
    $$
    R = \\frac{\\rho}{2\\pi^2\\,D}\\,
        \\Bigl(\\ln\\tfrac{8D}{d} + \\ln\\tfrac{4D}{s}\\Bigr)
    $$
    Parameters
    ----------
    ring_diameter
        Ring diameter $D$ in m.
    wire_diameter
        Wire diameter $d$ in m (i.e. ``2*radius``).
    depth
        Burial depth $t$; $s = 2\\,t$.

    Notes
    -----
    Validity: $d \\ll s \\ll D$. Outside this regime the error
    grows quickly.
    """
    D = ring_diameter
    d = wire_diameter
    s = 2.0 * depth
    return rho / (2.0 * math.pi ** 2 * D) * (
        math.log(8.0 * D / d) + math.log(4.0 * D / s)
    )


# ---------------------------------------------------------------------
# Strip
# ---------------------------------------------------------------------


def horizontal_strip(
    rho: float,
    length: float,
    width: float,
    thickness: float,
    depth: float,
) -> float:
    """Buried strip electrode (rectangular cross-section), Table I, Eq. (31).
    $$
    R = \\frac{\\rho}{4\\pi L}\\,
        \\Bigl(\\ln\\tfrac{4L}{a} + \\tfrac{a^2 - \\pi a b}{2(a+b)^2}
               + \\ln\\tfrac{4L}{s} - 2 + \\tfrac{s}{2L}
               - \\tfrac{s^2}{16 L^2}
               + \\tfrac{s^4}{512 L^4}\\Bigr)
    $$
    Parameters
    ----------
    length
        Half the strip length $L$ in m (total length ``2*L``).
    width
        Strip width $a$ in m.
    thickness
        Strip thickness $b$ in m. Validity: $b < a/8$.
    depth
        Burial depth; $s = 2\\,t$.
    """
    L, a, b = length, width, thickness
    if b >= a / 8.0:
        raise ValueError(
            "horizontal_strip: precondition b < a/8 violated "
            f"(a={a}, b={b}). See Dwight 1936, Eq. 30."
        )
    s = 2.0 * depth
    inner = (
        math.log(4.0 * L / a)
        + (a ** 2 - math.pi * a * b) / (2.0 * (a + b) ** 2)
        + math.log(4.0 * L / s)
        - 2.0
        + s / (2.0 * L)
        - s ** 2 / (16.0 * L ** 2)
        + s ** 4 / (512.0 * L ** 4)
    )
    return rho / (4.0 * math.pi * L) * inner


# ---------------------------------------------------------------------
# Plate electrodes
# ---------------------------------------------------------------------


def horizontal_round_plate(rho: float, radius: float, depth: float) -> float:
    """Horizontal round plate, Table I (Eqs. 32 + 36).
    $$
    R = \\frac{\\rho}{8 a} +
        \\frac{\\rho}{4\\pi s}\\,
          \\Bigl(1 - \\tfrac{7 a^2}{12 s^2}
                    + \\tfrac{33 a^4}{40 s^4}\\Bigr)
    $$
    Assumption: plate plane parallel to the soil surface; image
    distance $s = 2\\,t$ ≫ plate radius $a$ for the series
    to converge.
    """
    s = 2.0 * depth
    a = radius
    R_self = rho / (8.0 * a)
    R_image = rho / (4.0 * math.pi * s) * (
        1.0 - 7.0 * a ** 2 / (12.0 * s ** 2) + 33.0 * a ** 4 / (40.0 * s ** 4)
    )
    return R_self + R_image


def vertical_round_plate(rho: float, radius: float, depth: float) -> float:
    """Vertical round plate, Table I (Eqs. 32 + 38).

    In the original the plate-to-image distance is
    $s_2 \\approx 2 s$ (twice the depth of the plate centre to
    the surface). The series for the image contribution flips the sign
    of the $a^2$ term.
    """
    s = 2.0 * depth
    a = radius
    R_self = rho / (8.0 * a)
    # Series Eq. (38): (1 + 7 a^2 / (24 s^2) + 99 a^4 / (320 s^4))
    R_image = rho / (4.0 * math.pi * 2.0 * s) * (
        1.0 + 7.0 * a ** 2 / (24.0 * s ** 2) + 99.0 * a ** 4 / (320.0 * s ** 4)
    )
    return R_self + R_image


# ---------------------------------------------------------------------
# Hemisphere
# ---------------------------------------------------------------------


def hemisphere(rho: float, radius: float) -> float:
    """Hemispherical electrode of radius $A$ at the soil surface.
    $$
    R = \\frac{\\rho}{2\\pi A}
    $$
    (Dwight, p. 1320: hemisphere + image = full sphere,
    $C_{\\text{full sphere}} = A$, hence
    $R = \\rho/(2\\pi C) = \\rho/(2\\pi A)$.)
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    return rho / (2.0 * math.pi * radius)
