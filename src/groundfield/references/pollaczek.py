r"""Layered (Pollaczek/Sunde) earth-return line impedance reference.

The per-unit-length **self** and **mutual** impedances of conductors
buried in a two-layer earth, as exact quasi-static spectral integrals.
This is the layered, buried-conductor companion to
:func:`groundfield.references.earth_return.carson_self_impedance`: where
Carson gives the homogeneous equivalent-depth closed form, this gives the
spectral integral over a stratified earth, and reduces to Pollaczek's
homogeneous form for ``rho_1 == rho_2``.

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
A line current at depth :math:`h` in a two-layer earth (layer 1 of
thickness :math:`h_1` and resistivity :math:`\rho_1` over a half-space
:math:`\rho_2`, air above) has the quasi-static vector potential

.. math::

    A(y, z) \propto K_0(\gamma_k R)
        + \int_0^\infty F(\lambda; z, h)\,\cos(\lambda y)\,d\lambda ,

with :math:`\gamma_k^2 = j\omega\mu_0/\rho_k` of the conductor's layer and
:math:`u_k = \sqrt{\lambda^2 + \gamma_k^2}`. The spectral amplitude
:math:`F` follows exactly from the continuity of :math:`A` and
:math:`\partial A/\partial z` at the soil surface and at the layer
interface (Tsiamitros et al. 2005, Eq. 8). For a conductor in layer 1,
with :math:`r_0 = (u_1-\lambda)/(u_1+\lambda)` and
:math:`K = (u_1-u_2)/(u_1+u_2)`,

.. math::

    F = \frac{r_0 e^{-u_1(h+h')} + K e^{-u_1(2h_1-h-h')}
              + r_0 K\bigl[e^{-u_1(2h_1-h+h')} + e^{-u_1(2h_1+h-h')}\bigr]}
             {u_1\,\bigl(1 - r_0 K e^{-2u_1 h_1}\bigr)} ,

and the per-unit-length self impedance is

.. math::

    Z'_\text{self} = j\,\frac{\omega\mu_0}{2\pi}\left[
        K_0(\gamma_1\,\mathrm{GMR}) + \int_0^\infty F(\lambda; h, h)\,d\lambda
    \right].

Conductors in layer 2 and pairs in different layers use the
corresponding generalised-reflection and transmission amplitudes (see
:func:`_two_layer_amplitude`). For :math:`\rho_1 = \rho_2` the form
reduces to the homogeneous Pollaczek integral
:math:`K_0(\gamma\,\mathrm{GMR}) - K_0(2\gamma h)
+ 2\int_0^\infty e^{-2hu}/(\lambda+u)\,d\lambda`.

Review pass 10 (finding 10.45) replaced two approximations of earlier
releases:

* the two-layer kernel used the *surface* wavenumber
  :math:`u_\text{eff}` of the layered earth (Tagg/Sunde/Wait recursion,
  :func:`layered_effective_wavenumber`) in the buried exponent,
  :math:`e^{-2h u_\text{eff}}/(\lambda + u_\text{eff})` — exact only for
  :math:`h \to 0` or :math:`\rho_1 = \rho_2`; it missed the lower-interface
  term :math:`K e^{-2u_1(h_1-h)}`: :math:`R'` −0.13…−0.5 % in band near
  an interface, −28.5 % at 1 MHz;
* the head term was the small-argument form :math:`\ln(2h/\mathrm{GMR})`
  of :math:`K_0(\gamma\,\mathrm{GMR}) - K_0(2\gamma h)`: :math:`R'`
  −0.15 % (1 m) and −0.99 % (3 m) at 30 Ω·m, 1 kHz — more than the
  documented "< 0.05 %".

Both are now exact at every frequency (quasi-static) for the same cost.

Note the exponent: for a conductor **buried in the earth** the vertical
decay of the reflected field is governed by :math:`u_1`, not by
:math:`\lambda`. The Laplace kernel :math:`e^{-2h\lambda}` is Carson's
*air* kernel, valid only for a conductor above the interface; using it
below the interface under-counts :math:`R'` and inverts the sign of its
depth dependence (fixed in 0.15.0, pass-9 F09).

Internal-inductance convention. Like
:mod:`groundfield.references.earth_return`, this returns the
**earth-return** series impedance, external to the conductor, for the
geometric mean radius ``gmr``: pass ``gmr`` equal to the physical wire
radius :math:`a` for a pure external model, or
:math:`a\,e^{-1/4} = 0.7788\,a` to include the DC internal inductance
:math:`\mu_0/(8\pi)` of a solid round conductor — the convention of the
assembled inductive stack (``include_internal=True``, the default of
:func:`groundfield.coupling.inductance.thin_wire_self_inductance`).

Limiting cases (all covered by the regression tests):

* ``rho_1 == rho_2`` — homogeneous Pollaczek, which matches
  :func:`carson_self_impedance` to well under 1 % in the reactance across
  50 Hz – 1 kHz; the earth-return resistance sits slightly **above**
  :math:`\omega\mu_0/8` (Carson's low-frequency, surface-return limit) and
  **rises** with burial depth and with frequency.
* ``h_1 → ∞`` — upper layer only; ``h_1 → 0`` (conductor in layer 2) —
  lower layer only.
* two-layer — the reactance lies within the
  ``[carson(rho_1), carson(rho_2)]`` bracket, weighted toward the deeper,
  more conductive layer at low frequency.

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
from scipy.special import kv

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

    The Tagg/Sunde/Wait tanh-recursion surface term: the input
    wavenumber of the layered earth seen from the **surface** (it sets the
    surface reflection coefficient :math:`(u_\text{eff}-\lambda)/
    (u_\text{eff}+\lambda)` of an overhead conductor). It is not the
    exponent of a buried conductor's kernel — the self and mutual
    impedances of this module use the exact buried amplitude since review
    pass 10 (finding 10.45). Reduces to the homogeneous half-space value :math:`u_1` when
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

    Returns
    -------
    complex
        Effective wavenumber :math:`u_\text{eff}` in 1/m.
    """
    u1 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_1)
    if rho_1 == rho_2:
        return u1
    u2 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_2)
    t = cmath.tanh(u1 * h_1)
    return u1 * (u2 + u1 * t) / (u1 + u2 * t)


def _wavenumbers(lam: float, rho_1: float, rho_2: float, omega: float) -> tuple:
    """``(u_1, u_2)`` at the spectral variable ``lam`` (1/m)."""
    u1 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_1)
    u2 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_2)
    return u1, u2


def _two_layer_amplitude(
    lam: float,
    rho_1: float,
    rho_2: float,
    h_1: float,
    omega: float,
    depth_obs: float,
    depth_src: float,
) -> complex:
    r"""Exact spectral amplitude $F(\lambda)$ of a buried line current.

    Two-layer earth (layer 1 of thickness ``h_1`` over a half-space),
    air above, quasi-static ($e^{+j\omega t}$). A line current at depth
    $h'$ produces the primary field
    $K_0(\gamma R) = \int_0^\infty \cos(\lambda y)\,e^{-u|z-h'|}/u\,d\lambda$
    in its own layer; $F$ is the rest of the field at the observer depth
    $h$, from the continuity of $A$ and $\partial A/\partial z$ at both
    interfaces (Tsiamitros et al. 2005, Eq. 8 structure). With
    $r_0 = (u_1-\lambda)/(u_1+\lambda)$ (air) and
    $K = (u_1-u_2)/(u_1+u_2)$ (layer interface):

    * both in layer 1,
      $F = [r_0 e^{-u_1(h+h')} + K e^{-u_1(2d-h-h')}
      + r_0 K(e^{-u_1(2d-h+h')} + e^{-u_1(2d+h-h')})]
      / [u_1(1 - r_0 K e^{-2u_1 d})]$;
    * both in layer 2, $F = R_\uparrow e^{-u_2(h+h'-2d)}/u_2$ with the
      generalised upward reflection
      $R_\uparrow = (-K + r_0 e^{-2u_1 d})/(1 - K r_0 e^{-2u_1 d})$;
    * source in layer 1, observer in layer 2 (and, by reciprocity, the
      reverse), $F = 2[e^{-u_1(d-h')} + r_0 e^{-u_1(d+h')}]
      e^{-u_2(h-d)} / [(u_1+u_2)(1 - r_0 K e^{-2u_1 d})]$ — the full
      field, there is no direct term.

    For $\rho_1 = \rho_2$ every case collapses to the homogeneous
    $r_0 e^{-u(h+h')}/u$.
    """
    u1, u2 = _wavenumbers(lam, rho_1, rho_2, omega)
    d = h_1
    r0 = (u1 - lam) / (u1 + lam)
    K = (u1 - u2) / (u1 + u2)
    E2 = cmath.exp(-2.0 * u1 * d)
    in1_obs = depth_obs < d
    in1_src = depth_src < d
    if in1_obs and in1_src:
        h, hp = depth_obs, depth_src
        num = (
            r0 * cmath.exp(-u1 * (h + hp))
            + K * cmath.exp(-u1 * (2.0 * d - h - hp))
            + r0
            * K
            * (
                cmath.exp(-u1 * (2.0 * d - h + hp))
                + cmath.exp(-u1 * (2.0 * d + h - hp))
            )
        )
        return num / (u1 * (1.0 - r0 * K * E2))
    if not in1_obs and not in1_src:
        R_up = (-K + r0 * E2) / (1.0 - K * r0 * E2)
        return R_up * cmath.exp(-u2 * (depth_obs + depth_src - 2.0 * d)) / u2
    h1_pt, h2_pt = (depth_src, depth_obs) if in1_src else (depth_obs, depth_src)
    num = cmath.exp(-u1 * (d - h1_pt)) + r0 * cmath.exp(-u1 * (d + h1_pt))
    return 2.0 * num * cmath.exp(-u2 * (h2_pt - d)) / ((u1 + u2) * (1.0 - r0 * K * E2))


def _spectral_integral(
    rho_1: float,
    rho_2: float,
    h_1: float,
    omega: float,
    depth_obs: float,
    depth_src: float,
    x: float,
) -> complex:
    r"""$\int_0^\infty F(\lambda)\cos(\lambda x)\,d\lambda$ (adaptive quadrature).

    The integrand is smooth and decays exponentially; for horizontal
    distances beyond the decay length the oscillatory tail is handled by
    QUADPACK's Fourier rule. Breakpoints at the skin and layer scales
    keep the adaptive rule on the transitions.
    """

    def part(lam: float, which: int) -> float:
        v = _two_layer_amplitude(lam, rho_1, rho_2, h_1, omega, depth_obs, depth_src)
        return v.real if which == 0 else v.imag

    p1 = math.sqrt(omega * MU_0 / rho_1)
    p2 = math.sqrt(omega * MU_0 / rho_2)
    in1_obs, in1_src = depth_obs < h_1, depth_src < h_1
    if in1_obs and in1_src:
        decay = min(depth_obs + depth_src, 2.0 * h_1 - depth_obs - depth_src)
    elif not in1_obs and not in1_src:
        decay = depth_obs + depth_src - 2.0 * h_1
    else:
        decay = abs(depth_obs - depth_src)
    decay = max(decay, 1e-3)
    if x > 0.3 * decay:
        re = quad(part, 0.0, math.inf, args=(0,), weight="cos", wvar=x, limit=400)[0]
        im = quad(part, 0.0, math.inf, args=(1,), weight="cos", wvar=x, limit=400)[0]
        return complex(re, im)

    def part_cos(lam: float, which: int) -> float:
        return part(lam, which) * math.cos(lam * x)

    lam_split = 40.0 / decay
    pts = sorted(
        {s for s in (p1, p2, 1.0 / decay, 1.0 / max(h_1, 1e-9)) if 0.0 < s < lam_split}
    )
    kw = dict(limit=800, epsabs=1e-14, epsrel=1e-12)
    re = quad(part_cos, 0.0, lam_split, args=(0,), points=pts or None, **kw)[0]
    im = quad(part_cos, 0.0, lam_split, args=(1,), points=pts or None, **kw)[0]
    re += quad(part_cos, lam_split, math.inf, args=(0,), **kw)[0]
    im += quad(part_cos, lam_split, math.inf, args=(1,), **kw)[0]
    return complex(re, im)


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

    Exact two-layer, buried-conductor series impedance
    :math:`Z' = R' + jX'` (quasi-static), earth-return only — no internal
    conductor resistance; the internal inductance is included only through
    ``gmr`` (see the module docstring). Reduces to Pollaczek's homogeneous
    form for ``rho_1 == rho_2`` and agrees with
    :func:`carson_self_impedance` in the reactance.

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
        Burial depth of the conductor in m. Must be positive; a depth
        ``>= h_1`` places the conductor in the lower layer.
    gmr
        Geometric mean radius of the conductor in m: the physical wire
        radius for an external-inductance model, ``0.7788·a`` to include
        the internal inductance of a solid round conductor. Must be
        positive.

    Returns
    -------
    complex
        Earth-return series impedance :math:`R' + jX'` in Ω/m.
    """
    _check_positive(
        rho_1=rho_1,
        rho_2=rho_2,
        h_1=h_1,
        frequency=frequency,
        depth=depth,
        gmr=gmr,
    )
    omega = 2.0 * math.pi * frequency
    if rho_1 == rho_2:
        rho_2, h_1 = rho_1, max(h_1, 1e3 * depth)  # one layer: no interface term
    rho_k = rho_1 if depth < h_1 else rho_2
    gamma = cmath.sqrt(1j * omega * MU_0 / rho_k)
    head = complex(kv(0, gamma * gmr))
    integral = _spectral_integral(rho_1, rho_2, h_1, omega, depth, depth, 0.0)
    return 1j * omega * MU_0 / (2.0 * math.pi) * (head + integral)


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
    two-layer earth, in the same exact spectral form as
    :func:`pollaczek_self_impedance`:

    .. math::

        Z'_\text{M} = j\,\frac{\omega\mu_0}{2\pi}\left[
            K_0(\gamma_k d)
            + \int_0^\infty F(\lambda; h_i, h_j)\cos(\lambda x)\,d\lambda
        \right],

    with the horizontal separation :math:`x`, the direct distance
    :math:`d = \sqrt{x^2 + (h_i - h_j)^2}`, :math:`\gamma_k` of the common
    layer and the exact two-layer amplitude :math:`F` (module docstring).
    For two conductors in different layers there is no direct
    :math:`K_0` term and :math:`F` is the transmitted amplitude. In the
    homogeneous case the bracket is
    :math:`K_0(\gamma d) - K_0(\gamma D) + 2\int_0^\infty
    e^{-(h_i+h_j)u}/(\lambda + u)\cos(\lambda x)\,d\lambda` with
    :math:`D = \sqrt{x^2 + (h_i + h_j)^2}`.

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
    Earth-return coupling only: quasi-static, no displacement current. The
    vertical decay in the soil is governed by :math:`u_1`, not by Carson's
    air kernel :math:`e^{-(h_i+h_j)\lambda}`. As :math:`x \to \mathrm{GMR}`
    with :math:`depth_i = depth_j` it recovers
    :func:`pollaczek_self_impedance`.

    Raises
    ------
    ValueError
        If ``rho_1``, ``rho_2``, ``h_1``, ``frequency``, ``depth_i`` or
        ``depth_j`` is not positive and finite, if ``separation`` is
        negative, or if the two conductors coincide (``separation == 0``
        and ``depth_i == depth_j``).
    """
    _check_positive(
        rho_1=rho_1,
        rho_2=rho_2,
        h_1=h_1,
        frequency=frequency,
        depth_i=depth_i,
        depth_j=depth_j,
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
    if rho_1 == rho_2:
        rho_2, h_1 = rho_1, max(h_1, 1e3 * max(depth_i, depth_j))
    same_layer = (depth_i < h_1) == (depth_j < h_1)
    head = 0.0j
    if same_layer:
        rho_k = rho_1 if depth_i < h_1 else rho_2
        gamma = cmath.sqrt(1j * omega * MU_0 / rho_k)
        head = complex(kv(0, gamma * d))
    integral = _spectral_integral(rho_1, rho_2, h_1, omega, depth_i, depth_j, x)
    return 1j * omega * MU_0 / (2.0 * math.pi) * (head + integral)


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

    Raises
    ------
    ValueError
        If ``rho_1``, ``rho_2``, ``h_1``, ``frequency``, ``depth`` or
        ``gmr`` is not positive and finite, or if ``r_internal`` is
        negative.

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
        rho_1=rho_1,
        rho_2=rho_2,
        h_1=h_1,
        frequency=frequency,
        depth=depth,
        gmr=gmr,
    )
    if r_internal < 0.0:
        raise ValueError(f"r_internal must be non-negative, got {r_internal}")
    z_earth = pollaczek_self_impedance(
        rho_1,
        rho_2,
        h_1,
        frequency,
        depth=depth,
        gmr=gmr,
    )
    return complex(r_internal, 0.0) + z_earth
