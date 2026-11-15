"""Independent reference values for the engine benchmark suite.

Nothing here imports a groundfield solver: every value is either a
closed form or the result of a small, independent solver of the same
physical problem, so a benchmark compares a backend against a number it
did not help to compute.

Conventions: SI units (Ω·m, m, A, Ω), soil surface at ``z = 0``,
``z`` positive into the soil, quasi-static current field.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ellipk, j0

__all__ = [
    "hemisphere_resistance",
    "ring_layered_hankel",
    "ring_thin_wire_resistance",
    "rod_thin_wire_galerkin",
    "rod_thin_wire_limit",
    "solid_rod_finite_volume",
    "three_layer_kernel_mpmath",
    "two_layer_series_kernel",
    "wire_proximity_correction",
]


def hemisphere_resistance(rho: float, radius: float) -> float:
    """Resistance of a hemisphere at the soil surface, ``rho / (2 pi a)``.

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m.
    radius
        Hemisphere radius in m.

    Returns
    -------
    float
        Resistance in Ω (exact).
    """
    return rho / (2.0 * math.pi * radius)


def _h(t: np.ndarray, a: float) -> np.ndarray:
    """Antiderivative of ``asinh(t / a)`` (double integral of the reduced kernel)."""
    return t * np.arcsinh(t / a) - np.sqrt(t * t + a * a)


def rod_thin_wire_galerkin(length: float, radius: float, rho: float, n: int) -> float:
    """Galerkin thin-wire resistance of a vertical rod with its top at ``z = 0``.

    Pulse basis on ``n`` equal segments, Galerkin testing (test = basis),
    the reduced kernel ``1 / sqrt(t^2 + a^2)`` and the air image; every
    matrix entry is the analytic double integral. The Galerkin value is
    an upper bound of the thin-wire resistance that decreases
    monotonically under refinement (energy principle); see
    :func:`rod_thin_wire_limit` for the extrapolated limit.

    Parameters
    ----------
    length
        Rod length in m.
    radius
        Wire radius in m.
    rho
        Soil resistivity in Ω·m.
    n
        Number of equal segments.

    Returns
    -------
    float
        Resistance in Ω.
    """
    e = np.linspace(0.0, length, n + 1)
    z1, z2 = e[:-1], e[1:]
    zi, zf = z1[:, None], z2[:, None]
    w1, w2 = z1[None, :], z2[None, :]
    direct = (
        _h(zf - w1, radius)
        - _h(zi - w1, radius)
        - _h(zf - w2, radius)
        + _h(zi - w2, radius)
    )
    image = (
        _h(zf + w2, radius)
        - _h(zi + w2, radius)
        - _h(zf + w1, radius)
        + _h(zi + w1, radius)
    )
    ds = z2 - z1
    z_mat = (direct + image) / ds[:, None] / ds[None, :] * rho / (4.0 * math.pi)
    a_mat = np.block([[z_mat, -np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return float(np.linalg.solve(a_mat, b)[-1])


def rod_thin_wire_limit(length: float, radius: float, rho: float) -> float:
    """Thin-wire limit of :func:`rod_thin_wire_galerkin` (Richardson).

    Galerkin values on ``n = 96`` and ``n = 192`` segments, extrapolated
    with an ``O(h^2)``-type step. For ``L = 3 m, a = 8 mm, rho = 100``
    this gives 33.1957 Ω, against 33.4927 Ω from Sunde's closed form
    ``rho / (2 pi L) [ln(4 L / a) - 1]`` (which assumes a uniform line
    current) and 33.3254 Ω for the Galerkin solution at ``ds = 0.5 m``.

    Returns
    -------
    float
        Resistance in Ω.
    """
    coarse = rod_thin_wire_galerkin(length, radius, rho, 96)
    fine = rod_thin_wire_galerkin(length, radius, rho, 192)
    return fine + (fine - coarse) / 3.0


def ring_thin_wire_resistance(
    rho: float, ring_radius: float, wire_radius: float, depth: float
) -> float:
    """Thin-wire resistance of a horizontal ring buried at ``depth``.

    By rotational symmetry the leakage current along a ring is uniform,
    so the thin-wire problem has the closed-form solution

    ``R = rho / (2 pi^2) [K(m) / sqrt(a^2 + 4 R0^2) + K(m') / sqrt(4 z^2 + 4 R0^2)]``

    with ``m = 4 R0^2 / (a^2 + 4 R0^2)`` (the ring seen from its own wire
    surface) and ``m' = 4 R0^2 / (4 z^2 + 4 R0^2)`` (its air image), and
    ``K`` the complete elliptic integral of the first kind (parameter
    convention of :func:`scipy.special.ellipk`).

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m.
    ring_radius
        Ring radius ``R0`` in m.
    wire_radius
        Wire radius ``a`` in m.
    depth
        Burial depth ``z`` in m.

    Returns
    -------
    float
        Resistance in Ω.
    """
    r0, a, z = ring_radius, wire_radius, depth
    m_self = 4.0 * r0**2 / (a**2 + 4.0 * r0**2)
    m_img = 4.0 * r0**2 / (4.0 * z**2 + 4.0 * r0**2)
    self_term = ellipk(m_self) / math.sqrt(a**2 + 4.0 * r0**2)
    image_term = ellipk(m_img) / math.sqrt(4.0 * z**2 + 4.0 * r0**2)
    return rho / (2.0 * math.pi**2) * float(self_term + image_term)


def three_layer_kernel_mpmath(
    rhos: tuple[float, float, float],
    h: tuple[float, float],
    s: float,
    z: float,
    zs: float,
) -> float:
    """Top-layer three-layer Green's function by mpmath quadrature.

    The direct pair and its air mirror are taken in closed form; the
    reflected remainder

    ``int_0^inf [xi(lambda) - e^{-lambda|dz|} - e^{-lambda p}] J0(lambda s) dlambda``

    with ``xi`` the full top-layer spectral kernel (both image families
    and the multiple-reflection denominator, ``Gamma_1`` from the
    two-interface recursion) is integrated with ``mpmath.quadosc`` at
    30 digits (``mpmath.quad`` for ``s = 0``). Shares no code with
    :mod:`groundfield.solver.mom_sommerfeld`.

    Parameters
    ----------
    rhos
        Layer resistivities ``(rho_1, rho_2, rho_3)`` in Ω·m.
    h
        Thicknesses ``(h_1, h_2)`` in m.
    s
        Horizontal distance in m.
    z, zs
        Depths of the field point and the source in m (both ``<= h_1``).

    Returns
    -------
    float
        ``G`` in 1/m (potential ``rho_1 I G / (4 pi)``).
    """
    import mpmath as mp

    with mp.workdps(30):
        r1, r2, r3 = (mp.mpf(r) for r in rhos)
        h1, h2 = mp.mpf(h[0]), mp.mpf(h[1])
        k1 = (r2 - r1) / (r2 + r1)
        k2 = (r3 - r2) / (r3 + r2)
        adz, p = abs(mp.mpf(z) - mp.mpf(zs)), mp.mpf(z) + mp.mpf(zs)

        def remainder(lam):
            e = mp.e ** (-2 * lam * h2)
            g = (k1 + k2 * e) / (1 + k1 * k2 * e)
            e2h = mp.e ** (-2 * lam * h1)
            xi = (
                mp.e ** (-lam * adz)
                + g * mp.e ** (-lam * (2 * h1 - adz))
                + mp.e ** (-lam * p)
                + g * mp.e ** (-lam * (2 * h1 - p))
            ) / (1 - g * e2h)
            out = xi - mp.e ** (-lam * adz) - mp.e ** (-lam * p)
            return out * mp.besselj(0, lam * s) if s > 0 else out

        if s > 0:
            refl = mp.quadosc(remainder, [0, mp.inf], period=2 * mp.pi / s)
        else:
            refl = mp.quad(remainder, [0, 1, 10, mp.inf])
        direct = 1 / max(math.hypot(s, float(adz)), 1e-3) + 1 / math.hypot(s, float(p))
        return direct + float(refl)


def two_layer_series_kernel(k: float, h: float, s: float, z: float, zs: float) -> float:
    """Top-layer two-layer Green's function as the Tagg/Sunde image series.

    ``G = sum_m K^m [1/R(2mh + |dz|) + K/R(2(m+1)h - |dz|) + 1/R(2mh + p)
    + K/R(2(m+1)h - p)]`` with ``R(d) = sqrt(s^2 + d^2)`` floored at 1 mm
    and ``p = z + zs``, summed (small terms first) until ``|K|^m < 1e-17``.
    The potential of a point source ``I`` is ``rho_1 I G / (4 pi)``.

    Parameters
    ----------
    k
        Reflection coefficient ``(rho_2 - rho_1) / (rho_2 + rho_1)``.
    h
        Top-layer thickness in m.
    s
        Horizontal distance in m.
    z, zs
        Depths of the field point and the source in m (both ``<= h``).

    Returns
    -------
    float
        ``G`` in 1/m.
    """
    adz, p = abs(z - zs), z + zs

    def r(d):
        return np.maximum(np.hypot(s, d), 1e-3)

    if k == 0.0:
        return float(1.0 / r(adz) + 1.0 / r(p))
    m_max = int(math.ceil(math.log(1e-17) / math.log(abs(k)))) + 10
    m = np.arange(m_max + 1, dtype=float)
    terms = k**m * (
        1.0 / r(2 * m * h + adz)
        + k / r(2 * (m + 1) * h - adz)
        + 1.0 / r(2 * m * h + p)
        + k / r(2 * (m + 1) * h - p)
    )
    return float(np.sum(terms[::-1]))


# ---------------------------------------------------------------------
# References of the conforming fem model (stream w4axi, user decision Q37)
# ---------------------------------------------------------------------


def _ring_average_inverse_distance(s: float, dz: float, r0: float) -> float:
    """Average of ``1/R`` over a ring of radius ``r0``: ``(2/pi) K(m) / sqrt(q)``."""
    q = (s + r0) ** 2 + dz**2
    return (2.0 / math.pi) * float(ellipk(4.0 * s * r0 / q)) / math.sqrt(q)


def ring_layered_hankel(
    rhos,
    thicknesses,
    ring_radius: float,
    depth: float,
    wire_radius: float,
    *,
    n_points: int = 16,
) -> float:
    """Thin-ring resistance in horizontally layered soil (Hankel transform).

    A uniform current on the circle of radius ``ring_radius`` at ``depth``
    (the wire axis) in a stack of any number of layers over a half-space,
    insulating soil surface; the potential is averaged over ``n_points``
    points of the wire surface (radius ``wire_radius`` around the axis).
    The ring potential is
    ``phi(s, z) = int_0^inf Phi(lambda, z) J0(lambda s) J0(lambda r0) lambda dlambda``
    with ``Phi`` the transformed field of a point source, computed from the
    admittances ``y = Phi' / (lambda Phi)`` of the solutions regular at the
    surface and at depth (``tanh`` recursions, stable for every
    ``lambda``). The direct term (with the mean conductivity of the two
    sides for a ring on an interface) and the first images in the
    boundaries of the ring's layer are taken in closed form (elliptic
    ``K``); the remainder is integrated on Gauss panels, geometric towards
    ``lambda = 0`` (a resistive bottom carries the current far out).
    Shares no code with groundfield.

    Parameters
    ----------
    rhos
        Layer resistivities in Ω·m, top first (the last is the half-space).
    thicknesses
        Thicknesses of all but the last layer in m.
    ring_radius, depth, wire_radius
        Ring radius, depth of the wire axis and wire radius in m.
    n_points
        Points on the wire surface for the average.

    Returns
    -------
    float
        Resistance in Ω (potential for 1 A).
    """
    rhos = np.asarray(rhos, dtype=float)
    sig = 1.0 / rhos
    bounds = np.concatenate([[0.0], np.cumsum(np.asarray(thicknesses, dtype=float))])
    n_layers = rhos.size
    r0, t, a = float(ring_radius), float(depth), float(wire_radius)

    def layer_of(z: float, below: bool = True) -> int:
        return int(np.searchsorted(bounds[1:], z, side="right" if below else "left"))

    on_interface = bool(np.any(np.abs(bounds[1:] - t) <= 1e-12 * max(1.0, t)))
    ia, ib = layer_of(t, below=False), layer_of(t, below=True)
    sig_a, sig_b = sig[ia], sig[ib]
    sig_ref = 0.5 * (sig_a + sig_b)

    def y_up_tops(lam):
        y = np.zeros_like(lam)
        tops = [y]
        for k in range(n_layers - 1):
            th = np.tanh(lam * (bounds[k + 1] - bounds[k]))
            y = (sig[k] / sig[k + 1]) * (th + y) / (1.0 + y * th)
            tops.append(y)
        return tops

    def y_dn_bottoms(lam):
        bottoms = [None] * n_layers
        y = -np.ones_like(lam)
        bottoms[-1] = y
        for k in range(n_layers - 2, -1, -1):
            if k + 1 < n_layers - 1:
                th = np.tanh(lam * (bounds[k + 2] - bounds[k + 1]))
                y = (y - th) / (1.0 - y * th)
            y = (sig[k + 1] / sig[k]) * y
            bottoms[k] = y
        return bottoms

    def y_up(lam, tops, z, k):
        th = np.tanh(lam * (z - bounds[k]))
        return (th + tops[k]) / (1.0 + tops[k] * th)

    def y_dn(lam, bottoms, z, k):
        if k == n_layers - 1:
            return bottoms[k]
        th = np.tanh(lam * (bounds[k + 1] - z))
        return (bottoms[k] - th) / (1.0 - bottoms[k] * th)

    def ratio(lam, tops, bottoms, z):
        """``Phi(z) / Phi(t)``, layer by layer."""
        out = np.ones_like(lam)
        zc = t
        if z < t:
            k = ia
            while True:
                z_end = max(bounds[k], z)
                d = zc - z_end
                y = y_up(lam, tops, z_end, k)
                ch = np.cosh(np.minimum(lam * d, 700.0))
                out = out / (ch * (1.0 + y * np.tanh(lam * d)))
                if z_end == z:
                    return out
                zc, k = bounds[k], k - 1
        if z > t:
            k = ib
            while True:
                bot = bounds[k + 1] if k + 1 < bounds.size else math.inf
                z_end = min(bot, z)
                d = z_end - zc
                y = y_dn(lam, bottoms, z_end, k)
                ch = np.cosh(np.minimum(lam * d, 700.0))
                out = out / (ch * (1.0 - y * np.tanh(lam * d)))
                if z_end == z:
                    return out
                zc, k = bot, k + 1
        return out

    angles = 2.0 * math.pi * (np.arange(n_points) + 0.5) / n_points
    points = [(r0 + a * math.cos(q), t + a * math.sin(q)) for q in angles]
    images: list[tuple[float, float]] = []  # (image depth, strength)
    if not on_interface:
        k = ia
        top = bounds[k]
        bot = bounds[k + 1] if k + 1 < bounds.size else math.inf
        k_top = 1.0 if k == 0 else (sig[k] - sig[k - 1]) / (sig[k] + sig[k - 1])
        images.append((2.0 * top - t, k_top))
        decay = [2.0 * t]
        if k > 0:
            decay.append(2.0 * (t - bounds[k - 1]))
        if math.isfinite(bot):
            images.append(
                (2.0 * bot - t, (sig[k] - sig[k + 1]) / (sig[k] + sig[k + 1]))
            )
            decay.append(2.0 * (bot - top))
            decay.append(
                2.0 * (bounds[k + 2] - t)
                if k + 2 < bounds.size
                else 4.0 * (bot - t) + 2.0 * (t - top)
            )
        d_rem = min(decay)
    else:
        others = [abs(t - b) for b in bounds if abs(t - b) > 1e-12 * max(1.0, t)]
        d_rem = 2.0 * min(others + [t])

    closed = 0.0
    for s, z in points:
        closed += _ring_average_inverse_distance(s, z - t, r0)
        for zi, c in images:
            closed += c * _ring_average_inverse_distance(s, z - zi, r0)
    closed /= n_points * 4.0 * math.pi * sig_ref

    def integrand(lam):
        tops, bottoms = y_up_tops(lam), y_dn_bottoms(lam)
        phi_t = 1.0 / (
            2.0
            * math.pi
            * lam
            * (sig_a * y_up(lam, tops, t, ia) - sig_b * y_dn(lam, bottoms, t, ib))
        )
        total = np.zeros_like(lam)
        for s, z in points:
            sub = np.exp(-lam * abs(z - t))
            for zi, c in images:
                sub = sub + c * np.exp(-lam * abs(z - zi))
            sub = sub / (4.0 * math.pi * sig_ref * lam)
            total += (phi_t * ratio(lam, tops, bottoms, z) - sub) * j0(lam * s)
        return total / n_points * j0(lam * r0) * lam

    period = math.pi / (r0 + a)
    lam_max = 60.0 / d_rem
    edges = np.concatenate(
        [
            [0.0],
            period * 2.0 ** -np.arange(60, 0, -1),
            np.arange(period, lam_max + period, period),
        ]
    )
    xg, wg = np.polynomial.legendre.leggauss(16)
    half = 0.5 * np.diff(edges)
    lam = (half[:, None] * (xg[None, :] + 1.0) + edges[:-1, None]).ravel()
    w = (half[:, None] * wg[None, :]).ravel()
    return float(closed + np.sum(w * integrand(lam)))


def wire_proximity_correction(
    rho: float, ring_radius: float, wire_radius: float, distance: float, k: float
) -> float:
    """Proximity correction in Ω of a thin ring near a plane interface (2-D).

    The exact potential of an equipotential cylinder of radius ``a`` whose
    axis lies at ``distance`` ``d > a`` from a plane interface, minus the
    thin-wire value ``ln(1/a) + k ln(1/(2 d))`` (a line source on the axis,
    its image ``k`` in the interface, averaged over the wire surface), in
    units of ``rho / (2 pi)`` per unit current per length, converted to a
    ring of radius ``ring_radius`` (current per length ``I / (2 pi r0)``).
    The exact potential comes from the classical image iteration: the image
    of every line source inside the cylinder in the interface (strength
    ``k``) is compensated by its inverse point in the circle and a source at
    the centre, until the strengths fall below 1e-15; sources at equal
    positions are merged. For ``k = -1`` it reproduces
    ``arccosh(d/a) - ln(2 d / a)`` (checked to 2e-12). Valid when ``d`` is
    small against the ring radius and the distance to every other boundary.

    Parameters
    ----------
    rho
        Resistivity in Ω·m of the wire's side of the interface.
    ring_radius, wire_radius, distance
        Ring radius, wire radius and distance of the wire axis from the
        interface in m.
    k
        Reflection coefficient ``(rho_other - rho) / (rho_other + rho)``.

    Returns
    -------
    float
        Correction in Ω to add to a thin-ring resistance.
    """
    a, d = float(wire_radius), float(distance)
    internal: dict[float, float] = {d: 1.0}
    images: dict[float, float] = {-d: k}
    pending = {-d: k}
    while pending:
        inner: dict[float, float] = {}
        for y_img, q in pending.items():
            y_inv = d - a * a / (d - y_img)
            inner[y_inv] = inner.get(y_inv, 0.0) - q
            inner[d] = inner.get(d, 0.0) + q
        pending = {}
        for y_src, q in inner.items():
            internal[y_src] = internal.get(y_src, 0.0) + q
            images[-y_src] = images.get(-y_src, 0.0) + k * q
            if abs(k * q) > 1e-15:
                pending[-y_src] = pending.get(-y_src, 0.0) + k * q
    sources = list(internal.items()) + list(images.items())
    surface = [
        -sum(
            q * math.log(math.hypot(a * math.cos(th), d + a * math.sin(th) - y))
            for y, q in sources
        )
        for th in np.linspace(0.0, 2.0 * math.pi, 9)[:-1]
    ]
    delta = float(np.mean(surface)) - (-math.log(a) - k * math.log(2.0 * d))
    return rho / (2.0 * math.pi) * delta / (2.0 * math.pi * ring_radius)


def _graded_faces(points, h_min, growth, h_max, lo, hi):
    """Monotone faces on [lo, hi] refined to ``h_min`` at every point."""
    pts = sorted({lo, hi, *(p for p in points if lo <= p <= hi)})
    faces = [lo]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        length = b - a
        left, h, used = [], h_min, 0.0
        while used + h < length / 2:
            left.append(h)
            used += h
            h = min(h * growth, h_max)
        rest = length - 2 * used
        mid = [rest] if rest > 0 else []
        if rest > h_max:
            n = int(np.ceil(rest / h_max))
            mid = [rest / n] * n
        steps = np.array(left + mid + left[::-1]) * (length / (2 * used + rest))
        faces.extend(list(a + np.cumsum(steps)))
    faces = np.array(faces)
    faces[-1] = hi
    return faces


def solid_rod_finite_volume(
    rho_1: float,
    rho_2: float,
    h_1: float,
    z_top: float,
    z_tip: float,
    radius: float,
    *,
    r_far: float = 2.0e6,
    h_min: float = 5.0e-5,
    growth: float = 1.05,
) -> float:
    """Resistance of a solid rod in two-layer soil, cell-centred finite volumes.

    A perfectly conducting cylinder (``radius``, flat end faces, from
    ``z_top`` to ``z_tip``) on the axis of an axisymmetric two-layer
    half-space (``rho_1`` down to ``h_1``, then ``rho_2``), insulating
    surface, ``phi = 0`` on ``r = R`` and ``z = R`` (``r_far``; the bias
    ``rho_2 / (2 pi R)`` is 5e-4 Ω for 1000 Ω·m at the default 2000 km).
    Cell-centred finite volumes on a graded ``(r, z)`` grid: logarithmic
    radial and harmonic axial face resistances; the rod faces, the
    interface and the soil surface are cell faces, refined to ``h_min``
    with the ratio ``growth``, the rod's end faces from inside its radius
    as well. Adapted from the independent reference of audit A3
    (``p20_axisymmetric_reference.py``, which resolved the end faces with
    four cells only and converged to a value 6e-4 too high for a tip in a
    conductive layer). First order in ``h_min``; shares no code with
    groundfield.

    Returns
    -------
    float
        Resistance in Ω.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    a = float(radius)
    inner = [a - h_min * growth**k for k in range(2000) if a - h_min * growth**k > 0.0]
    rf_in = np.array(sorted({0.0, a, *inner}))
    rf_out = [a]
    h = h_min
    while rf_out[-1] < r_far:
        rf_out.append(min(rf_out[-1] + h, r_far))
        h *= growth
    rf = np.concatenate([rf_in[:-1], np.array(rf_out)])
    key = [0.0, z_top, z_tip] + ([h_1] if 0.0 < h_1 < r_far else [])
    zf = list(_graded_faces(key + [z_tip + 1.0], h_min, growth, 0.1, 0.0, z_tip + 1.0))
    h = zf[-1] - zf[-2]
    while zf[-1] < r_far:
        h *= growth
        zf.append(min(zf[-1] + h, r_far))
    zf = np.array(zf)
    nr, nz = rf.size - 1, zf.size - 1
    rc, zc = 0.5 * (rf[:-1] + rf[1:]), 0.5 * (zf[:-1] + zf[1:])
    dz = np.diff(zf)
    sig = np.where(zc < h_1, 1.0 / rho_1, 1.0 / rho_2)
    rod = (rc[:, None] < a) & (zc[None, :] > z_top) & (zc[None, :] < z_tip)
    idx = np.arange(nr * nz).reshape(nr, nz)
    rows, cols, vals = [], [], []

    def couple(p, q, g):
        rows.extend([p, q, p, q])
        cols.extend([q, p, p, q])
        vals.extend([-g, -g, g, g])

    with np.errstate(divide="ignore"):
        for i in range(nr - 1):
            face = rf[i + 1]
            r_i = np.log(face / max(rc[i], 1e-12)) / (2 * np.pi * sig * dz)
            r_j = np.log(rc[i + 1] / face) / (2 * np.pi * sig * dz)
            r_i = np.where(rod[i, :], 0.0, r_i)
            r_j = np.where(rod[i + 1, :], 0.0, r_j)
            couple(
                idx[i, :],
                idx[i + 1, :],
                np.where(rod[i] & rod[i + 1], 0.0, 1 / (r_i + r_j)),
            )
        area = np.pi * (rf[1:] ** 2 - rf[:-1] ** 2)
        for k in range(nz - 1):
            r_i = np.where(rod[:, k], 0.0, 0.5 * dz[k] / (sig[k] * area))
            r_j = np.where(rod[:, k + 1], 0.0, 0.5 * dz[k + 1] / (sig[k + 1] * area))
            couple(
                idx[:, k],
                idx[:, k + 1],
                np.where(rod[:, k] & rod[:, k + 1], 0.0, 1 / (r_i + r_j)),
            )
    g_r = 2 * np.pi * sig * dz / np.log(rf[-1] / rc[-1])
    rows.append(idx[-1, :])
    cols.append(idx[-1, :])
    vals.append(g_r)  # noqa: E702
    g_z = sig[-1] * area / (0.5 * dz[-1])
    rows.append(idx[:, -1])
    cols.append(idx[:, -1])
    vals.append(g_z)  # noqa: E702
    mat = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nr * nz, nr * nz),
    )
    fixed = rod.ravel()
    free = ~fixed
    phi = np.zeros(nr * nz)
    phi[fixed] = 1.0
    phi[free] = spsolve(mat[free][:, free].tocsc(), -mat[free][:, fixed] @ phi[fixed])
    return 1.0 / float((mat[fixed] @ phi).sum())
