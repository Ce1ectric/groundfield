"""Tagg/Sunde image series of a two-layer soil (private).

One object, :class:`ImageSeries`, owns the coefficients, the truncation
rule and the convergence report of the two-layer image-charge series.
The solver (:mod:`groundfield.solver.image_2layer`), the post-solve
field evaluation (:meth:`groundfield.solver.result.FieldResult.potential`)
and :mod:`groundfield.solver.mutual` all iterate the same object, so the
series can no longer drift apart between them (review pass 10, 10.75;
before, the series existed in five code copies with two truncation
policies).

Theory
------
With the reflection coefficient
$K = (\\rho_2 - \\rho_1)/(\\rho_2 + \\rho_1)$ a point current $I$ at
depth $z_s < h_1$ produces at a point $(x, y, z)$ of the upper layer

$$
\\varphi = \\frac{\\rho_1 I}{4\\pi}\\Bigl[\\frac{1}{r_0^{+}}
  + \\frac{1}{r_0^{-}} + \\sum_{n \\ge 1} K^n \\sum_{\\sigma, \\tau = \\pm 1}
  \\frac{1}{r_n^{\\sigma\\tau}}\\Bigr],\\qquad
r_n^{\\sigma\\tau} = \\sqrt{s^2 + \\bigl(z - (\\sigma\\, 2 n h_1
  + \\tau z_s)\\bigr)^2},
$$

with $s$ the horizontal distance. The $n = 0$ pair (the source and its
mirror image in the soil surface) is the homogeneous half-space kernel.
Every image term is a point (or, for the line kernel, a line) source
at the transformed depth $z_\\text{img} = \\sigma\\, 2 n h_1 + \\tau z_s$.

Truncation
----------
The series is cut after the first $n$ whose geometric tail bound
$\\sum_{m > n} |K|^m = |K|^{n+1}/(1 - |K|)$ falls below ``tol``, or
at ``max_terms``. The report states the number of terms used, the
remaining tail bound and whether ``tol`` was reached.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

__all__ = ["ImageSeries", "SeriesTruncationWarning"]


class SeriesTruncationWarning(UserWarning):
    """The Tagg/Sunde image series hit ``max_terms`` before ``tol``.

    Emitted (in addition to the logger record) whenever a backend
    truncates the two-layer image-charge series without reaching the
    requested tail tolerance — typically at high layer contrast
    (:math:`|K| \\to 1`). Silence with
    ``warnings.simplefilter("ignore", SeriesTruncationWarning)`` or
    raise ``Engine.image_max_terms``.
    """


def _series_tail_bound(abs_K: float, n: int) -> float:
    """Upper bound of the neglected series tail after term ``n``.

    The image weights are :math:`K^m`; the neglected tail is bounded
    by the geometric remainder

    .. math:: \\sum_{m > n} |K|^m = \\frac{|K|^{n+1}}{1 - |K|}.

    This replaces the historic per-term test :math:`|K|^n < tol`,
    which underestimates the remainder by the factor
    :math:`1/(1-|K|)` — a factor 17 at the AP1 corner soil
    :math:`|K| = 0.94`.
    """
    if abs_K >= 1.0:  # defensive; |K| < 1 for finite resistivities
        return math.inf
    return abs_K ** (n + 1) / (1.0 - abs_K)


@dataclass(frozen=True)
class ImageTerm:
    """One image of the series: depth ``offset + sign * z_source``.

    Attributes
    ----------
    n
        Series index (``0`` for the source and its surface mirror).
    offset
        Depth offset :math:`\\sigma\\, 2 n h_1` in m.
    sign
        Sign :math:`\\tau = \\pm 1` applied to the source depth.
    weight
        Image weight :math:`K^n` (``1.0`` for ``n = 0``).
    """

    n: int
    offset: float
    sign: float
    weight: float

    def depth(self, z_source: np.ndarray | float) -> np.ndarray | float:
        """Image depth ``offset + sign * z_source`` in m."""
        return self.offset + self.sign * z_source


@dataclass(frozen=True)
class ImageSeries:
    """Truncated Tagg/Sunde image series of a two-layer soil.

    Build it with :meth:`for_soil` (or :meth:`from_coefficients`); the
    constructor arguments of the dataclass are the derived state.

    Attributes
    ----------
    K
        Reflection coefficient :math:`(\\rho_2-\\rho_1)/(\\rho_2+\\rho_1)`.
    h_1
        Thickness of the upper layer in m.
    max_terms
        Cap on the number of terms :math:`n \\ge 1`.
    tol
        Requested tail bound.
    n_terms
        Number of terms :math:`n \\ge 1` the truncation keeps.
    """

    K: float
    h_1: float
    max_terms: int
    tol: float
    n_terms: int

    @classmethod
    def from_coefficients(
        cls, K: float, h_1: float, max_terms: int, tol: float
    ) -> ImageSeries:
        """Series for reflection coefficient ``K`` and layer thickness ``h_1`` (m).

        Parameters
        ----------
        K
            Reflection coefficient, :math:`|K| < 1`.
        h_1
            Upper-layer thickness in m.
        max_terms
            Maximum number of terms :math:`n \\ge 1`.
        tol
            Stop once the tail bound :math:`|K|^{n+1}/(1-|K|)` is below
            this value.
        """
        abs_K = abs(K)
        n_used = 0
        for n in range(1, max_terms + 1):
            n_used = n
            if _series_tail_bound(abs_K, n) < tol:
                break
        return cls(
            K=float(K), h_1=float(h_1), max_terms=int(max_terms), tol=float(tol),
            n_terms=n_used,
        )

    @classmethod
    def for_soil(cls, soil, max_terms: int, tol: float) -> ImageSeries:
        """Series of a :class:`~groundfield.soil.models.TwoLayerSoil`."""
        return cls.from_coefficients(
            soil.reflection_coefficient, soil.h_1, max_terms, tol
        )

    # ------------------------------------------------------------------
    # Terms
    # ------------------------------------------------------------------

    def surface_terms(self) -> tuple[ImageTerm, ImageTerm]:
        """The ``n = 0`` pair: the source and its mirror in the surface."""
        return (ImageTerm(0, 0.0, +1.0, 1.0), ImageTerm(0, 0.0, -1.0, 1.0))

    def layer_terms(self, n: int) -> tuple[ImageTerm, ...]:
        """The four images of index ``n >= 1`` in the canonical order.

        The order ``(σ, τ) = (+,+), (+,−), (−,+), (−,−)`` is part of the
        contract: every consumer accumulates the images in this order,
        which keeps the floating-point sums reproducible.
        """
        K_n = self.K**n
        return tuple(
            ImageTerm(n, sign_n * 2.0 * n * self.h_1, float(sign_zs), K_n)
            for sign_n in (+1, -1)
            for sign_zs in (+1, -1)
        )

    def iter_layer_terms(self) -> Iterator[tuple[ImageTerm, ...]]:
        """Image quadruples of ``n = 1 … n_terms`` (see :meth:`layer_terms`)."""
        for n in range(1, self.n_terms + 1):
            yield self.layer_terms(n)

    def offsets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All images as arrays ``(offset, sign, weight)``, ``n = 0`` first.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Depth offsets in m, signs applied to the source depth, and
            the weights :math:`K^n`; ``2 + 4 n_terms`` entries each.
        """
        terms = list(self.surface_terms())
        for quad in self.iter_layer_terms():
            terms.extend(quad)
        return (
            np.array([t.offset for t in terms], dtype=float),
            np.array([t.sign for t in terms], dtype=float),
            np.array([t.weight for t in terms], dtype=float),
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @property
    def tail_bound(self) -> float:
        """Bound of the neglected tail after :attr:`n_terms`."""
        return _series_tail_bound(abs(self.K), self.n_terms)

    @property
    def converged(self) -> bool:
        """Whether the tail bound fell below :attr:`tol`."""
        return self.tail_bound < self.tol if self.n_terms else True

    def report(self) -> dict[str, float | int | bool]:
        """Convergence report for ``FieldResult.metadata``.

        Returns
        -------
        dict
            ``K``, ``n_terms_used``, ``max_terms``, ``tol``,
            ``tail_bound`` and ``converged``.
        """
        return {
            "K": float(self.K),
            "n_terms_used": int(self.n_terms),
            "max_terms": int(self.max_terms),
            "tol": float(self.tol),
            "tail_bound": float(self.tail_bound),
            "converged": bool(self.converged),
        }
