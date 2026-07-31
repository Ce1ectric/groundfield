r"""Rigorous layered (Pollaczek/Sunde) earth-return line impedance reference.

The per-unit-length **self** impedance of a conductor buried in a
two-layer earth, in the rigorous Pollaczek/Sunde integral form. This is
the layered, buried-conductor companion to
:func:`groundfield.references.earth_return.carson_self_impedance`: where
Carson gives the homogeneous equivalent-depth closed form, this gives the
exact spectral integral over a stratified earth, and reduces to Carson in
the homogeneous limit (``rho_1 == rho_2``).

It is the analytic per-unit-length quantity that the assembled
Sommerfeld/Pollaczek stack (``earth_inductive_model="sommerfeld"``) does
**not** expose directly — the assembled stack builds a PEEC segment
inductance matrix, from which a single per-unit-length line impedance
cannot be read back without an ill-defined return-path assumption. This
closed integral form therefore both (a) serves callers that need a
frequency- and layer-dependent series impedance belag ``z'(f)`` (e.g. a
reduced nodal network of a distribution earthing system) and (b) gives
groundfield an independent per-unit-length reference against which the
assembled stack's earth-return reactance can be benchmarked.

Mathematical background
-----------------------
Integrating the Sommerfeld point Green's function along the conductor
axis yields the Pollaczek self impedance of a conductor buried at depth
:math:`h` with geometric mean radius :math:`\mathrm{GMR}`:

.. math::

    Z'_\text{self} = j\,\frac{\omega\mu_0}{2\pi}\left[
        \ln\!\frac{2h}{\mathrm{GMR}}
        + 2\int_0^\infty
          \frac{e^{-2h\,u_\text{eff}(\lambda)}}{\lambda + u_\text{eff}(\lambda)}
          \,d\lambda \right],

where the effective vertical wavenumber of the two-layer earth follows
the Tagg/Sunde/Wait recursion

.. math::

    u_\text{eff}(\lambda) = u_1\,
        \frac{u_2 + u_1\tanh(u_1 h_1)}{u_1 + u_2\tanh(u_1 h_1)},
    \qquad
    u_i = \sqrt{\lambda^2 + j\omega\mu_0/\rho_i}.

The :math:`\ln(2h/\mathrm{GMR})` term is the small-argument form of the
perfect-mirror image :math:`K_0(\gamma\,\mathrm{GMR}) - K_0(2\gamma h)`;
the integral is the finite-conductivity earth-return correction.

Note the exponent: for a conductor **buried in the earth** the vertical
decay of the reflected field is governed by :math:`u_\text{eff}(\lambda)`,
not by :math:`\lambda`. The Laplace kernel :math:`e^{-2h\lambda}` is
Carson's *air* kernel, valid only for a conductor above the interface
(:math:`\gamma \to 0` in the upper half-space); using it below the
interface systematically under-counts :math:`R'` and inverts the sign of
its depth dependence (fixed in 0.15.0). Because the head term uses the
small-argument form of the :math:`K_0` pair, the validity envelope is
:math:`|\gamma|\,2h \ll 1` — comfortably satisfied over the
50 Hz – 1 kHz band this package targets (< 0.05 % in :math:`R'` at
:math:`h \le 1` m, ~0.4 % at :math:`h = 3` m against the full
:math:`K_0` form), but degrading above ~10 kHz.

Like :mod:`groundfield.references.earth_return`, this returns the
**earth-return** series impedance only (external inductance, no internal
conductor resistance or internal inductance): pass ``gmr`` equal to the
physical wire radius for a pure external model, and add the internal
conductor impedance ``R'_\text{int} + jX'_\text{int}`` separately if
required.

Limiting cases (all covered by the regression tests):

* ``rho_1 == rho_2``  →  ``u_eff = u_1``  →  homogeneous Pollaczek, which
  matches :func:`carson_self_impedance` to well under 1 % in the reactance
  across 50 Hz – 1 kHz; the earth-return resistance sits slightly
  **above** :math:`\omega\mu_0/8` (Carson's low-frequency, surface-return
  limit) and **rises** with burial depth and with frequency, because a
  deeper conductor forces the return current through more soil (e.g.
  ``rho = 30`` Ω·m, 1 kHz: 1.012·:math:`\omega\mu_0/8` at 0.7 m,
  1.040·:math:`\omega\mu_0/8` at 3 m).
* ``h_1 → ∞``  →  ``u_eff → u_1``  (upper layer only).
* ``h_1 → 0``  →  ``u_eff → u_2``  (lower layer only).
* two-layer  →  the reactance lies within the
  ``[carson(rho_1), carson(rho_2)]`` bracket, weighted toward the deeper,
  more conductive layer at low frequency (the inductive return is
  deep-layer dominated — a thin resistive topsoil is "seen through").

References
----------
- Pollaczek, F. (1926). Über das Feld einer unendlich langen
  wechselstromdurchflossenen Einfachleitung. *Elektr. Nachr.-Techn.*
  **3**, 339–359.
- Sunde, E. D. (1949). *Earth Conduction Effects in Transmission
  Systems*, Van Nostrand, §§ 3–4 (buried-conductor earth-return, layered
  surface impedance).
- Wait, J. R. (1970). *Electromagnetic Waves in Stratified Media*, §2.
- Tsiamitros, D. A. et al. (2005). Earth return path impedances of
  underground cables for the two-layer earth case.
  *IEEE Trans. Power Delivery* **20**(3), 2174–2181 (Eq. 8).
"""

from __future__ import annotations

import cmath
import math

from scipy.integrate import quad

from .earth_return import MU_0, _check_positive

__all__ = [
    "layered_effective_wavenumber",
    "pollaczek_self_impedance",
    "pollaczek_mutual_impedance",
    "loop_impedance_conductor_earth",
]


def layered_effective_wavenumber(
    lam: float, rho_1: float, rho_2: float, h_1: float, omega: float
) -> complex:
    r"""Effective vertical wavenumber :math:`u_\text{eff}(\lambda)` of a two-layer earth.

    The Tagg/Sunde/Wait tanh-recursion surface term seen by a buried
    conductor. Reduces to the homogeneous half-space value :math:`u_1` when
    ``rho_1 == rho_2``, to :math:`u_1` as ``h_1 → ∞`` and to :math:`u_2`
    as ``h_1 → 0``. Units: 1/m.

    Parameters
    ----------
    lam
        Spectral (Hankel) variable, 1/m.
    rho_1, rho_2
        Upper / lower layer resistivity in Ω·m.
    h_1
        Upper-layer thickness in m.
    omega
        Angular frequency in rad/s.
    """
    u1 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_1)
    if rho_1 == rho_2:
        return u1
    u2 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_2)
    t = cmath.tanh(u1 * h_1)
    return u1 * (u2 + u1 * t) / (u1 + u2 * t)


def pollaczek_self_impedance(
    rho_1: float,
    rho_2: float,
    h_1: float,
    frequency: float,
    *,
    depth: float,
    gmr: float,
) -> complex:
    r"""Layered Pollaczek earth-return **self** impedance per unit length (Ω/m).

    Rigorous two-layer, buried-conductor series impedance
    :math:`Z' = R' + jX'`; earth-return only (no internal-conductor
    resistance or inductance — see the module docstring). Reduces to
    :func:`carson_self_impedance` in the homogeneous limit.

    Parameters
    ----------
    rho_1, rho_2
        Upper / lower layer resistivity in Ω·m. Must be positive. Set
        equal for a homogeneous earth.
    h_1
        Upper-layer thickness in m. Must be positive (ignored when
        ``rho_1 == rho_2``).
    frequency
        Frequency in Hz. Must be positive.
    depth
        Burial depth of the conductor in m. Must be positive.
    gmr
        Geometric mean radius of the conductor in m (the physical wire
        radius for a thin external-inductance model). Must be positive.

    Returns
    -------
    complex
        Earth-return series impedance :math:`R' + jX'` in Ω/m.
    """
    _check_positive(
        rho_1=rho_1, rho_2=rho_2, h_1=h_1, frequency=frequency,
        depth=depth, gmr=gmr,
    )
    omega = 2.0 * math.pi * frequency

    def integrand(lam: float, part: int) -> float:
        ue = layered_effective_wavenumber(lam, rho_1, rho_2, h_1, omega)
        # Buried-conductor (Pollaczek/Sunde) kernel: the vertical decay of
        # the reflected field is governed by u_eff, not by lam. lam would be
        # Carson's AIR kernel (gamma -> 0), which under-counts R'.
        v = cmath.exp(-2.0 * depth * ue) / (lam + ue)
        return v.real if part == 0 else v.imag

    re, _ = quad(integrand, 0.0, math.inf, args=(0,), limit=200)
    im, _ = quad(integrand, 0.0, math.inf, args=(1,), limit=200)
    return (
        1j * omega * MU_0 / (2.0 * math.pi)
        * (math.log(2.0 * depth / gmr) + 2.0 * (re + 1j * im))
    )


def pollaczek_mutual_impedance(
    rho_1: float,
    rho_2: float,
    h_1: float,
    frequency: float,
    *,
    depth_i: float,
    depth_j: float,
    separation: float,
) -> complex:
    r"""Layered Pollaczek earth-return **mutual** impedance per unit length (Ω/m).

    The per-unit-length coupling impedance between two parallel buried
    conductors — each closing its current through the earth — over a
    two-layer earth, in the same Pollaczek/Sunde spectral-integral form as
    :func:`pollaczek_self_impedance`:

    .. math::

        Z'_\text{M} = j\,\frac{\omega\mu_0}{2\pi}\left[
            \ln\!\frac{D}{d}
            + 2\int_0^\infty
              \frac{e^{-(h_i+h_j)\,u_\text{eff}(\lambda)}}
                   {\lambda + u_\text{eff}(\lambda)}
              \cos(\lambda x)\,d\lambda \right],

    with the horizontal separation :math:`x`, the direct and mirror
    distances

    .. math::

        d = \sqrt{x^2 + (h_i - h_j)^2}, \qquad
        D = \sqrt{x^2 + (h_i + h_j)^2},

    and the two-layer effective vertical wavenumber
    :math:`u_\text{eff}(\lambda)` from
    :func:`layered_effective_wavenumber`. The :math:`\ln(D/d)` term is the
    perfect-mirror image pair (the small-argument form of
    :math:`K_0(\gamma d) - K_0(\gamma D)`); the integral is the
    finite-conductivity earth-return correction.

    This is the mutual (off-diagonal) counterpart of
    :func:`pollaczek_self_impedance`: the coupling that induces a voltage on
    a parallel earthing conductor / cable through the shared earth-return
    path — the term a self impedance alone omits. It carries **no
    internal-conductor term** (the two conductors are galvanically separate;
    the coupling is purely inductive through the soil), so with ideal
    (:math:`\approx 0\,\Omega`) earthing at both ends the coupling impedance
    of a parallel run of length :math:`\ell` is :math:`Z'_\text{M}\,\ell`.
    Together with two self impedances it forms the :math:`2\times2` loop
    impedance matrix of the coupled pair.

    Limiting behaviour (covered by the tests): it reduces to the homogeneous
    Carson mutual impedance
    (:func:`~groundfield.references.earth_return.carson_mutual_impedance`)
    when :math:`\rho_1 = \rho_2` (reactance to well under 1 % at low
    frequency), its magnitude is below the self impedance and decays
    monotonically with the separation :math:`x`, and its two-layer reactance
    lies within the two homogeneous Carson bounds, deep-layer dominated at
    low frequency.

    Parameters
    ----------
    rho_1, rho_2, h_1, frequency
        Two-layer earth resistivities (Ω·m), upper-layer thickness (m) and
        frequency (Hz), as in :func:`pollaczek_self_impedance`. Set
        ``rho_1 == rho_2`` for a homogeneous earth.
    depth_i, depth_j
        Burial depths of the two conductors in m. Must be positive.
    separation
        Horizontal centre-to-centre distance :math:`x` between the
        conductors in m. Must be non-negative; ``separation == 0`` is
        allowed only when ``depth_i != depth_j`` (one conductor directly
        above the other). Coincident conductors (``separation == 0`` and
        ``depth_i == depth_j``) are the self case — use
        :func:`pollaczek_self_impedance`.

    Returns
    -------
    complex
        Mutual series impedance :math:`R'_\text{M} + jX'_\text{M}` in Ω/m.

    Notes
    -----
    Earth-return coupling only: quasi-static, no displacement current, and
    (like the self) the layering enters solely through
    :math:`u_\text{eff}`, which also governs the vertical decay in the
    exponent — the buried-conductor kernel
    :math:`e^{-(h_i+h_j)u_\text{eff}}`, not Carson's air kernel
    :math:`e^{-(h_i+h_j)\lambda}`. As :math:`x \to \mathrm{GMR}` with
    :math:`depth_i = depth_j` it recovers :func:`pollaczek_self_impedance`.
    """
    _check_positive(
        rho_1=rho_1, rho_2=rho_2, h_1=h_1, frequency=frequency,
        depth_i=depth_i, depth_j=depth_j,
    )
    if separation < 0.0:
        raise ValueError(f"separation must be non-negative, got {separation}")
    omega = 2.0 * math.pi * frequency
    x = separation
    d = math.hypot(x, depth_i - depth_j)
    if d <= 0.0:
        raise ValueError(
            "coincident conductors (separation == 0 and depth_i == depth_j); "
            "use pollaczek_self_impedance for the self case"
        )
    big_d = math.hypot(x, depth_i + depth_j)

    def integrand(lam: float, part: int) -> float:
        ue = layered_effective_wavenumber(lam, rho_1, rho_2, h_1, omega)
        # Buried-conductor (Pollaczek/Sunde) kernel: exp(-(h_i+h_j) u_eff),
        # not Carson's air kernel exp(-(h_i+h_j) lam) — see the self impedance.
        v = cmath.exp(-(depth_i + depth_j) * ue) / (lam + ue)
        return v.real if part == 0 else v.imag

    # Oscillatory cos(lam*x) tail. Use the weighted quadrature when the
    # oscillation is resolved over the exponential decay length
    # L = depth_i + depth_j (x >~ 0.3 L); for small x (cos ~ 1 over the
    # decay, where the weighted rule is ill-conditioned) fold the cosine
    # into the integrand and integrate plainly.
    decay_len = depth_i + depth_j
    if x > 0.3 * decay_len:
        re, _ = quad(integrand, 0.0, math.inf, args=(0,), weight="cos",
                     wvar=x, limit=200)
        im, _ = quad(integrand, 0.0, math.inf, args=(1,), weight="cos",
                     wvar=x, limit=200)
    else:
        def integrand_cos(lam: float, part: int) -> float:
            return integrand(lam, part) * math.cos(lam * x)

        re, _ = quad(integrand_cos, 0.0, math.inf, args=(0,), limit=200)
        im, _ = quad(integrand_cos, 0.0, math.inf, args=(1,), limit=200)
    return (
        1j * omega * MU_0 / (2.0 * math.pi)
        * (math.log(big_d / d) + 2.0 * (re + 1j * im))
    )


def loop_impedance_conductor_earth(
    rho_1: float,
    rho_2: float,
    h_1: float,
    frequency: float,
    *,
    depth: float,
    gmr: float,
    r_internal: float,
) -> complex:
    r"""Conductor–earth loop impedance per unit length (Ω/m), ideal termination.

    The series impedance seen by a current that flows out along a buried
    earthing conductor — a cable PEN, a cable screen — and returns through
    the earth, with the earthing electrodes at both ends idealised as
    :math:`\approx 0\,\Omega`:

    .. math::

        z'_\text{loop}
        = \underbrace{R'_\text{internal}}_{\text{conductor}}
          + \underbrace{Z'_\text{earth-return}}_{\text{Pollaczek self}}
        = R'_\text{internal}
          + Z'_\text{self}(\rho_1, \rho_2, h_1, f;\, \text{depth}, \text{gmr}).

    This is the "go along the conductor, return through the soil" loop that
    governs earth-fault current sharing and the reach of an earthing system
    connected through cable PENs / screens — the quantity a bare
    earth-return impedance (:func:`pollaczek_self_impedance`) does not give
    on its own, because it omits the conductor's own series resistance.

    Convention. The earth-return term carries the external inductance and,
    **through** ``gmr``, the conductor's internal inductance: for a solid
    round conductor pass ``gmr = 0.7788·a = a·e^{-1/4}`` (``a`` the physical
    radius), which already includes the internal inductance
    :math:`\mu_0/(8\pi)`; for a thin tubular / wire screen pass ``gmr`` equal
    to the screen radius (negligible internal inductance). ``r_internal`` is
    the conductor's series resistance per unit length — its DC / IEC 60228
    value, or a skin-corrected AC value near the top of the band. With the
    ideal (:math:`\approx 0\,\Omega`) earthing at both ends the loop
    impedance of a line of length :math:`\ell` is simply
    :math:`z'_\text{loop}\,\ell`; a real earthing resistance :math:`R_E` at
    the terminations adds in series on top of that.

    Parameters
    ----------
    rho_1, rho_2, h_1, frequency
        Two-layer earth resistivities (Ω·m), upper-layer thickness (m) and
        frequency (Hz), exactly as in :func:`pollaczek_self_impedance`.
        Set ``rho_1 == rho_2`` for a homogeneous earth.
    depth
        Burial depth of the conductor in m. Must be positive.
    gmr
        Geometric mean radius in m — ``0.7788·a`` for a solid round
        conductor, the screen radius for a tubular / wire screen. Must be
        positive.
    r_internal
        Conductor series resistance per unit length in Ω/m. Must be
        non-negative.

    Returns
    -------
    complex
        Loop series impedance :math:`R'_\text{loop} + jX'_\text{loop}` in
        Ω/m. Multiply by the line length for the terminated loop impedance.

    Notes
    -----
    Earth-return path only: no mutual coupling to a parallel conductor, and
    no internal skin-effect model (pass a skin-corrected ``r_internal`` if
    that matters at the top of the band). A conductor–conductor loop (e.g.
    line–PEN at a fixed spacing) has its return-path reactance set by that
    spacing, not by the earth-return depth, and is much smaller — this
    function is specifically the conductor–*earth* loop.

    Examples
    --------
    >>> import math
    >>> a = math.sqrt(150e-6 / math.pi)          # 150 mm^2 solid-round radius
    >>> z = loop_impedance_conductor_earth(       # NAYY PEN, 100 Ohm.m, 50 Hz
    ...     100.0, 100.0, 10.0, 50.0,
    ...     depth=0.7, gmr=0.7788 * a, r_internal=0.206e-3)
    >>> round(z.real * 1e3, 3), round(z.imag * 1e3, 3)  # mOhm/m -> Ohm/km
    (0.255, 0.758)
    """
    _check_positive(
        rho_1=rho_1, rho_2=rho_2, h_1=h_1, frequency=frequency,
        depth=depth, gmr=gmr,
    )
    if r_internal < 0.0:
        raise ValueError(f"r_internal must be non-negative, got {r_internal}")
    z_earth = pollaczek_self_impedance(
        rho_1, rho_2, h_1, frequency, depth=depth, gmr=gmr,
    )
    return complex(r_internal, 0.0) + z_earth
