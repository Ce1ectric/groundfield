r"""Effective resistivity of a homogeneous or two-layer soil.

A ``groundinsight`` bus carries one soil resistivity, its
``specific_earth_resistance``, and a BusType formula is evaluated at that
value. The AP1 catalogue of real LV networks describes a two-layer soil for
this purpose by its **effective resistivity**, a weighted geometric mean of
the two layer resistivities,

$$
\rho_\text{eff} \;=\; \rho_1^{\,1-w}\,\rho_2^{\,w},
\qquad
w \;=\; w_0 \left(\frac{h_1}{h_\text{ref}}\right)^{-\beta},
\qquad h_\text{ref} = 10\ \mathrm{m},
$$

that is, $\ln\rho_\text{eff} = (1 - w)\ln\rho_1 + w\ln\rho_2$: the
weight $w$ of the lower layer falls with the thickness $h_1$ of the upper
layer. The calibration on the AP1 catalogue is $w_0 = 0.13$,
$\beta = 0.8$; the AP2 hand-over carries group-specific values. AP2 sets the
``specific_earth_resistance`` of a bus to $\rho_\text{eff}$, so the ``rho``
of a BusType fitted in layered soil is $\rho_\text{eff}$ of the soil it was
fitted in (ADR-0008, amendment 2026.11).

Properties
----------
- **Homogeneous soil is exact:** a homogeneous soil, and a layer stack whose
  layers all have one resistivity, returns that resistivity unchanged (the
  plain formula with $\rho_1 = \rho_2$ can be one unit in the last place
  off).
- **Limited weight:** $w$ is limited to $[0, 1]$, so $\rho_\text{eff}$
  lies between $\rho_1$ and $\rho_2$. Without the limit $w > 1$ for
  $h_1 < h_\text{ref}\,w_0^{1/\beta}$ (0.78 m with the AP1 values) and
  $\rho_\text{eff}$ leaves that interval: 17.9 Ω·m for
  $\rho_1 = 100$, $\rho_2 = 30\ \Omega\,\mathrm{m}$, $h_1 = 0.5$ m.
  With the limit such a thin upper layer gives $\rho_2$.
- **Thick upper layer:** $w \to 0$ for $h_1 \to \infty$ ($\beta > 0$),
  so $\rho_\text{eff} \to \rho_1$; $w_0 = 0$ gives $\rho_1$ for every
  $h_1$.
- **Canonical stack:** a :class:`~groundfield.soil.MultiLayerSoil` is first
  reduced to its canonical stack, adjacent layers of equal resistivity
  merged, as every layered backend does (review pass 10, 10.48); three or
  more distinct layers raise :class:`ValueError`.

Validity
--------
An empirical one-parameter description of a two-layer soil, calibrated on
the LV networks of the AP1 catalogue: $h_1$ = 5 … 30 m,
$\rho_1$ = 30 … 1000 Ω·m, $\rho_2$ = 30 … 200 Ω·m. Outside that range it is
an extrapolation, and the weight depends on the extent of the earthing
system: a compact electrode in the upper layer sees less of the lower layer
than an LV network (see the Soil page of the documentation). Check the fit
quality (``max_relative``) of a reduced model fitted over
$\rho_\text{eff}$.

Not to be confused with :attr:`FieldResult.soil_resistivity
<groundfield.FieldResult.soil_resistivity>` (the upper-layer resistivity of
a layered soil) or with a frequency-dependent earth-return resistivity
(ADR-0005).
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING

from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, TwoLayerSoil

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from groundfield.soil.models import SoilModel

__all__ = ["effective_resistivity"]

_CALLER = "effective_resistivity"


def _number(value: object, name: str, *, positive: bool, unit: str) -> float:
    """``value`` as a finite float; positive or non-negative, else ValueError."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
    else:
        number = math.nan
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        bound = "positive" if positive else "non-negative (>= 0)"
        raise ValueError(
            f"{_CALLER}: {name} must be a finite, {bound} number{unit}, got {value!r}."
        )
    return number


def _two_layer_value(
    rho_1: float, rho_2: float, h_1: float, w0: float, beta: float, h_ref: float
) -> float:
    r"""$\rho_1^{1-w}\rho_2^{w}$ with the weight $w$ limited to $[0, 1]$.

    $w = w_0 (h_1/h_\text{ref})^{-\beta}$. The expression is the one of
    the AP1 scripts, so inside the calibration range the result is
    bit-identical to theirs.
    """
    if w0 == 0.0:
        w = 0.0
    else:
        try:
            w = min(w0 * (h_1 / h_ref) ** (-beta), 1.0)
        except OverflowError:  # (h_1 / h_ref)**(-beta) beyond the float range
            w = 1.0
    value = rho_1 ** (1.0 - w) * rho_2**w
    # Round-off must not take rho_eff out of [min, max] (w at 0 or 1 is exact).
    return min(max(value, min(rho_1, rho_2)), max(rho_1, rho_2))


def effective_resistivity(
    soil: SoilModel | None = None,
    *,
    rho_1: float | None = None,
    rho_2: float | None = None,
    h_1: float | None = None,
    w0: float = 0.13,
    beta: float = 0.8,
    h_ref: float = 10.0,
) -> float:
    r"""Effective resistivity of a homogeneous or two-layer soil in Ω·m.

    $\rho_\text{eff} = \rho_1^{\,1-w}\,\rho_2^{\,w}$ with
    $w = w_0\,(h_1/h_\text{ref})^{-\beta}$ limited to $[0, 1]$; the
    resistivity itself for homogeneous soil. This is the ``rho`` of a
    BusType fitted in layered soil and the ``specific_earth_resistance`` of
    the matching groundinsight bus (see :mod:`groundfield.soil.effective`
    for the derivation, the limit of $w$ and the calibration range).

    Parameters
    ----------
    soil : HomogeneousSoil, TwoLayerSoil or MultiLayerSoil, optional
        The soil. A :class:`~groundfield.soil.MultiLayerSoil` is reduced
        to its canonical stack (adjacent layers of equal resistivity
        merged) and must have at most two distinct layers. Give either
        ``soil`` or all of ``rho_1``, ``rho_2`` and ``h_1``.
    rho_1 : float, optional
        Resistivity of the upper layer in Ω·m, finite and positive
        (keyword, instead of ``soil``).
    rho_2 : float, optional
        Resistivity of the lower half-space in Ω·m, finite and positive
        (keyword, instead of ``soil``). ``rho_2 == rho_1`` is homogeneous
        soil.
    h_1 : float, optional
        Thickness of the upper layer in m, finite and positive (keyword,
        instead of ``soil``).
    w0 : float, optional
        Weight of the lower layer at $h_1 = h_\text{ref}$
        (dimensionless, finite, $\ge 0$). Default 0.13, the calibration
        of the AP1 catalogue. ``w0=0`` gives the upper-layer resistivity.
    beta : float, optional
        Exponent of the thickness dependence (dimensionless, finite,
        $\ge 0$, so that the weight does not grow with $h_1$). Default
        0.8 (AP1 catalogue).
    h_ref : float, optional
        Reference thickness in m, finite and positive. Default 10 m.

    Returns
    -------
    float
        $\rho_\text{eff}$ in Ω·m, between $\rho_1$ and $\rho_2$.

    Raises
    ------
    TypeError
        If both ``soil`` and numbers, or neither, are given, if a number
        is missing, or if ``soil`` is not a soil model.
    ValueError
        If a number is not finite or out of range, or if the soil has
        three or more distinct layers.

    Notes
    -----
    Inside the calibration range the value is bit-identical to the
    AP1 / AP2 scripts' ``r1 ** (1 - w) * r2 ** w``. For a thin upper layer
    ($h_1 < h_\text{ref}\,w_0^{1/\beta}$, 0.78 m with the defaults) the
    limit $w = 1$ gives $\rho_2$, where the unlimited formula would leave
    the interval $[\rho_1, \rho_2]$.

    Examples
    --------
    >>> import groundfield as gf
    >>> soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=5.0)
    >>> round(gf.effective_resistivity(soil), 4)  # w = 0.13 * 0.5**-0.8 = 0.226
    76.1465
    >>> gf.effective_resistivity(gf.HomogeneousSoil(rho=100.0))
    100.0
    >>> gf.effective_resistivity(rho_1=100.0, rho_2=30.0, h_1=0.5)  # w limited to 1
    30.0
    """
    w0 = _number(w0, "w0", positive=False, unit="")
    beta = _number(beta, "beta", positive=False, unit="")
    h_ref = _number(h_ref, "h_ref", positive=True, unit=" in m")
    numbers_given = {"rho_1": rho_1, "rho_2": rho_2, "h_1": h_1}
    if soil is None:
        missing = [name for name, value in numbers_given.items() if value is None]
        if missing:
            raise TypeError(
                f"{_CALLER}: pass either a soil model or rho_1, rho_2 and h_1; "
                f"missing {', '.join(missing)}."
            )
        soil = TwoLayerSoil(
            rho_1=_number(rho_1, "rho_1", positive=True, unit=" in Ohm m"),
            rho_2=_number(rho_2, "rho_2", positive=True, unit=" in Ohm m"),
            h_1=_number(h_1, "h_1", positive=True, unit=" in m"),
        )
    elif any(value is not None for value in numbers_given.values()):
        raise TypeError(
            f"{_CALLER}: pass either a soil model or rho_1, rho_2 and h_1, not both."
        )
    elif not isinstance(soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            f"{_CALLER}: soil must be a HomogeneousSoil, TwoLayerSoil or "
            f"MultiLayerSoil, got {type(soil).__name__}; for numbers pass "
            "rho_1=..., rho_2=..., h_1=... as keywords."
        )

    # The canonical stack of the layered backends (equal layers merged, 10.48).
    from groundfield.solver._layered import as_layer_stack

    stack = as_layer_stack(soil)
    if stack.n_layers == 1:
        return float(stack.rhos[0])
    if stack.n_layers > 2:
        rhos = ", ".join(f"{float(rho):g}" for rho in stack.rhos)
        raise ValueError(
            f"{_CALLER}: the soil has {stack.n_layers} distinct layers "
            f"(rho = {rhos} Ohm m); the effective resistivity is defined for "
            "homogeneous and two-layer soil (AP1 calibration). Describe the "
            "soil by an equivalent two-layer model, or pass rho_1, rho_2 and "
            "h_1 of one explicitly."
        )
    return _two_layer_value(
        float(stack.rhos[0]), float(stack.rhos[1]), float(stack.h[0]), w0, beta, h_ref
    )
