r"""Complex penetration depth of the earth return in layered soil.

The earth-return impedance of a cable or an overhead line over a
homogeneous earth is commonly written with a **complex penetration depth**
(Dubanton 1969; Deri, Tevan, Semlyen and Castanheira 1981): the earth is
replaced by a perfectly conducting plane at the complex depth

$$
p \;=\; \sqrt{\frac{\rho}{j\omega\mu_0}} \;=\; \frac{1}{\gamma},
\qquad \gamma = \sqrt{j\omega\mu_0/\rho},
$$

below the surface, and the per-metre mutual impedance of two wires at the
distances $h_i$, $h_j$ from the surface and the horizontal distance $x$ is

$$
Z'_{ij} \;=\; \frac{j\omega\mu_0}{2\pi}\,
\ln\frac{\sqrt{(h_i + h_j + 2p)^2 + x^2}}{\sqrt{(h_i - h_j)^2 + x^2}} ,
$$

the self impedance the same with $x$ the wire radius (or GMR) and
$h_i = h_j$. :func:`penetration_depth` returns, per frequency, the $p$ of a
**layered** soil for which these homogeneous formulas reproduce the layered
earth-return impedance, and the equivalent homogeneous resistivity
$\rho_\text{eq} = j\omega\mu_0\,p^2$ that gives the same $p$.

Matching
--------
The layered earth enters the impedance of horizontal wires only through
its surface reflection coefficient $\Gamma(\lambda)$ ($\lambda$ the
horizontal wavenumber), and the complex depth replaces it by
$e^{-2p\lambda}$. For a stack of layers $\Gamma(\lambda) = (1 - \lambda
q(\lambda))/(1 + \lambda q(\lambda))$ with an input admittance $q$ that
depends on $\lambda^2$ only, so with $p = q(0)$

$$
\Gamma(\lambda) \;=\; 1 - 2p\lambda + 2p^2\lambda^2 + O(\lambda^3)
\;=\; e^{-2p\lambda} + O(\lambda^3) ,
$$

exactly as for the homogeneous earth: $p$ is the plane-wave surface
impedance of the stack divided by $j\omega\mu_0$, computed by the
transmission-line recursion from the bottom layer up,

$$
p_n = \frac{1}{\gamma_n},\qquad
p_k = \frac{p_{k+1} + \tanh(\gamma_k h_k)/\gamma_k}
           {1 + \gamma_k\,p_{k+1}\tanh(\gamma_k h_k)},\qquad p = p_1 .
$$

Its modulus $|\rho_\text{eq}|$ is the magnetotelluric apparent resistivity
of the stack; its phase is zero for homogeneous soil.

Where it is exact, and its error
--------------------------------
- **Homogeneous soil:** $p = 1/\gamma$ and $\rho_\text{eq} = \rho$, i.e.
  the formulas of Dubanton and Deri themselves (which approximate Carson's
  integral to a few per cent at most, worst near $|\gamma| D \approx 1$).
- **Large separation and low frequency:** the term of $\Gamma$ that the
  formula misses is $O(\lambda^3)$, so the mutual impedance is reproduced
  asymptotically for $x \to \infty$ (the $1/x^2$ term depends on $p$ and
  $p^2$ only) and for $f \to 0$ (the field reaches the bottom half-space,
  $p \to 1/\gamma_n$).
- **Thin and thick top layers:** $p$ tends to the value of the lower layer
  for $h_1 \to 0$ and of the upper one for $h_1 \to \infty$.
- In between, the error is largest where the wavenumbers that dominate the
  impedance ($\lambda \sim 1/x$ and $1/(h_i + h_j)$) reach the layer
  scales ($1/h_k$, $|\gamma_k|$). Measured on the 39 soils of the AP1 grid
  (stream w3perf, two methods: the exact layered earth-return integral of
  :func:`groundfield.coupling.sommerfeld_inductance.per_metre_mutual_impedance`
  and this formula) the equivalent depth reproduces the earth-return
  correction of the layered soil to within the error that the same formula
  has in homogeneous soil plus a part of the layering (numbers on the Soil
  reference page of the documentation).

The function is a modelling aid for line-impedance formulas; the solver's
earth-return models (``carson_series``, ``sommerfeld``) do not use it.

References
----------
- Dubanton, C. (1969). Calcul approché des paramètres primaires et
  secondaires d'une ligne de transport, valeurs homopolaires. *EDF
  Bulletin de la Direction des Études et Recherches*, série B, 1, 53–62.
- Deri, A.; Tevan, G.; Semlyen, A.; Castanheira, A. (1981). The complex
  ground return plane: a simplified model for homogeneous and multi-layer
  earth return. *IEEE Trans. PAS* **100**(8), 3686–3693.
"""

from __future__ import annotations

import cmath
import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, TwoLayerSoil

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from groundfield.soil.models import SoilModel

__all__ = ["PenetrationDepth", "penetration_depth"]

_CALLER = "penetration_depth"
MU_0 = 4.0e-7 * math.pi


@dataclass(frozen=True, eq=False)
class PenetrationDepth:
    """Complex penetration depth of a soil, per frequency.

    A value: the three arrays are read-only copies of the given ones, two
    results compare equal when their arrays are equal (``==``), and a result
    is hashable (by its frequencies, so equal results have equal hashes), so
    it can key a dictionary or join a set (audit W3B-11).

    Attributes
    ----------
    frequencies : numpy.ndarray
        Frequencies in Hz, shape ``(n_f,)``.
    depth : numpy.ndarray
        Complex penetration depth $p$ in m, shape ``(n_f,)``:
        $\\sqrt{\\rho/(j\\omega\\mu_0)}$ for homogeneous soil, the surface
        impedance of the stack over $j\\omega\\mu_0$ for layered soil.
    resistivity : numpy.ndarray
        Equivalent homogeneous resistivity $\\rho_\\text{eq} = j\\omega\\mu_0
        p^2$ in Ω·m, shape ``(n_f,)``, complex: real and equal to $\\rho$ for
        homogeneous soil; its modulus is the magnetotelluric apparent
        resistivity of a layered soil.

    Raises
    ------
    ValueError
        If the three arrays are not one-dimensional of one length.
    """

    frequencies: np.ndarray
    depth: np.ndarray
    resistivity: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "frequencies": np.array(self.frequencies, dtype=float),
            "depth": np.array(self.depth, dtype=complex),
            "resistivity": np.array(self.resistivity, dtype=complex),
        }
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(
                "PenetrationDepth: frequencies, depth and resistivity must be "
                "one-dimensional arrays of one length (one entry per frequency), "
                f"got the shapes {[value.shape for value in arrays.values()]}."
            )
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PenetrationDepth):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("frequencies", "depth", "resistivity")
        )

    def __hash__(self) -> int:
        return hash(self.frequencies.tobytes())

    def mutual_impedance(self, z_i: float, z_j: float, distance: float) -> np.ndarray:
        """Per-metre impedance of the complex-depth formula, every frequency.

        $Z' = \\frac{j\\omega\\mu_0}{2\\pi}\\ln(D'/D)$ with
        $D = \\sqrt{(z_i - z_j)^2 + x^2}$ and
        $D' = \\sqrt{(|z_i| + |z_j| + 2p)^2 + x^2}$ (see the module
        docstring): the mutual impedance of two parallel wires, or the self
        impedance of one wire with ``z_i == z_j`` and ``distance`` its radius
        (or GMR). Earth return and external inductance only (no internal
        impedance of the wire).

        Parameters
        ----------
        z_i, z_j : float
            Depths of the wires in m, the groundfield convention (``> 0``
            buried, ``< 0`` overhead); both on one side of the surface.
        distance : float
            Horizontal distance in m (``> 0``; the radius for a self
            impedance).

        Returns
        -------
        numpy.ndarray
            Complex impedance in Ω/m, shape ``(n_f,)``.

        Raises
        ------
        ValueError
            For wires on opposite sides of the surface, a non-finite input
            or a distance that is not positive.
        """
        values = (float(z_i), float(z_j), float(distance))
        if not all(math.isfinite(v) for v in values) or not values[2] > 0.0:
            raise ValueError(
                f"{_CALLER}: z_i, z_j must be finite and distance positive, got "
                f"{z_i!r}, {z_j!r}, {distance!r}."
            )
        if values[0] * values[1] < 0.0:
            raise ValueError(
                f"{_CALLER}: the complex-depth formula needs both wires on one "
                f"side of the surface, got z_i = {z_i}, z_j = {z_j}."
            )
        a, b, x = abs(values[0]), abs(values[1]), values[2]
        direct = math.hypot(a - b, x)
        image = np.sqrt((a + b + 2.0 * self.depth) ** 2 + x * x)
        omega = 2.0 * math.pi * self.frequencies
        return 1j * omega * MU_0 / (2.0 * math.pi) * np.log(image / direct)


_TINY = float(np.finfo(float).tiny)
_SQRT_J = cmath.sqrt(1j)


def _frequencies(frequencies: object) -> np.ndarray:
    """Frequencies in Hz as a 1-D float array; finite, positive, real.

    ``bool`` and ``str`` are refused, not converted (audit W3B-11: ``"50"``
    and ``True`` were read as 50 Hz and 1 Hz), and so is a subnormal value,
    below ``numpy.finfo(float).tiny`` (``f = 1e-320`` raised
    :class:`ZeroDivisionError`).
    """
    try:
        values = np.asarray(frequencies, dtype=object)
        items = values.ravel().tolist()
    except (TypeError, ValueError):
        items = [None]
    if not all(
        isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
        for v in items
    ):
        raise ValueError(
            f"{_CALLER}: frequencies must be finite, positive real numbers "
            f"(Hz), not bool, str or complex, got {frequencies!r}."
        )
    f = np.atleast_1d(np.array([float(v) for v in items], dtype=float))
    if values.ndim > 1:
        f = f.reshape(values.shape)
    if f.ndim != 1 or f.size == 0 or not np.all(np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError(
            f"{_CALLER}: frequencies must be finite and positive (Hz), got "
            f"{frequencies!r}. At f = 0 the penetration depth is infinite."
        )
    if np.any(f < _TINY):
        raise ValueError(
            f"{_CALLER}: frequencies must be positive normal numbers, at least "
            f"numpy.finfo(float).tiny = {_TINY:.3g} Hz; got {frequencies!r}, "
            "which holds a subnormal value."
        )
    return f


def penetration_depth(
    soil: SoilModel, frequencies: float | np.ndarray
) -> PenetrationDepth:
    r"""Complex penetration depth and equivalent resistivity of a soil.

    Per frequency, the complex depth $p$ for which the homogeneous
    complex-depth formulas of Dubanton and Deri reproduce the earth-return
    impedance of the (layered) soil, and the equivalent homogeneous
    resistivity $\rho_\text{eq} = j\omega\mu_0 p^2$ with the same $p$ (see
    :mod:`groundfield.soil.penetration` for the matching, where it is exact
    and its accuracy). Homogeneous soil gives $p = \sqrt{\rho/(j\omega\mu_0)}$
    and $\rho_\text{eq} = \rho$; any number of layers is accepted.

    Parameters
    ----------
    soil : HomogeneousSoil, TwoLayerSoil or MultiLayerSoil
        The soil (reduced to its canonical stack: adjacent layers of equal
        resistivity merged, as in every layered backend).
    frequencies : float or array_like
        Frequencies in Hz: finite, positive real numbers, at least
        ``numpy.finfo(float).tiny`` (not ``bool``, ``str`` or complex).

    Returns
    -------
    PenetrationDepth
        ``frequencies`` (Hz), ``depth`` $p$ (m, complex) and ``resistivity``
        $\rho_\text{eq}$ (Ω·m, complex), one entry per frequency;
        :meth:`PenetrationDepth.mutual_impedance` evaluates the complex-depth
        formula.

    Raises
    ------
    TypeError
        If ``soil`` is not a soil model.
    ValueError
        If a frequency is not a finite, positive real number (``bool``,
        ``str`` and complex values are refused, not converted) or is
        subnormal (below ``numpy.finfo(float).tiny``).

    Notes
    -----
    The recursion is the transmission-line transformation of the plane-wave
    surface impedance from the bottom half-space up through the layers; it
    is the $\lambda \to 0$ value of the input admittance of the stack, which
    makes $e^{-2p\lambda}$ agree with the layered surface reflection
    coefficient to second order in the horizontal wavenumber.

    Examples
    --------
    >>> import groundfield as gf
    >>> pd = gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), [50.0])
    >>> complex(pd.resistivity[0])
    (100+0j)
    >>> pd = gf.penetration_depth(gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), [50.0])
    >>> round(float(abs(pd.resistivity[0])), 3)  # the lower layer dominates at 50 Hz
    30.756
    """
    if not isinstance(soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            f"{_CALLER}: soil must be a HomogeneousSoil, TwoLayerSoil or "
            f"MultiLayerSoil, got {type(soil).__name__}."
        )
    f = _frequencies(frequencies)
    from groundfield.solver._layered import as_layer_stack

    stack = as_layer_stack(soil)
    rhos = [float(r) for r in stack.rhos]
    thicknesses = [float(h) for h in stack.h]
    depth = np.empty(f.size, dtype=complex)
    resistivity = np.empty(f.size, dtype=complex)
    for k, freq in enumerate(f.tolist()):
        # sqrt(j w mu0) and gamma_k = sqrt(j w mu0 / rho_k) as products of
        # square roots: j w mu0 itself is subnormal near the smallest normal
        # frequency, and sqrt(rho / (j w mu0)) overflowed to inf below about
        # 1e-303 Hz (audit W3B-11, own attack)
        root = _SQRT_J * math.sqrt(2.0 * math.pi * freq) * math.sqrt(MU_0)
        gammas = [root / math.sqrt(rho) for rho in rhos]
        p = 1.0 / gammas[-1]
        if len(rhos) == 1:
            depth[k] = p
            resistivity[k] = rhos[0]
            continue
        for g, h in zip(reversed(gammas[:-1]), reversed(thicknesses), strict=True):
            e = cmath.exp(-2.0 * g * h)
            t = (1.0 - e) / (1.0 + e)  # tanh(g h), stable for Re(g h) > 0
            p = (p + t / g) / (1.0 + g * p * t)
        depth[k] = p
        resistivity[k] = (root * p) ** 2
    return PenetrationDepth(frequencies=f, depth=depth, resistivity=resistivity)
