"""Image-charge backend dispatcher for n-layer soil (``image_nlayer``).

Mathematical / physical model
-----------------------------
The image-charge family of grounding solvers represents the layered
half-space Green's function as a (truncated) sum of point sources at
mirrored positions in the soil. This sum is **closed-form** for two
regimes:

- **n = 1 (homogeneous soil).** A single source plus its air-mirror
  at $z = -z_s$ reproduces the Neumann boundary at $z = 0$
  exactly. This is the classical ``image`` backend.
- **n = 2 (two-layer soil).** The recursive reflection
  $\\Gamma_1(\\lambda) \\equiv K_1$ is constant in
  $\\lambda$, and the geometric expansion of the multiple
  reflection between the air boundary and the layer interface gives
  the **Tagg / Sunde series**
  $$
  \\varphi(s, z) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}\\,
  \\sum_{n=0}^{\\infty} K_1^{n}\\,
  \\Bigl(\\tfrac{1}{r_n^{++}} + \\tfrac{1}{r_n^{+-}}
       + \\tfrac{1}{r_n^{-+}} + \\tfrac{1}{r_n^{--}}\\Bigr).
  $$
For $n \\ge 3$ the upward reflection $\\Gamma_1(\\lambda)$
becomes a non-trivial function of $\\lambda$ (see
:func:`groundfield.solver._layered.reflection_gamma`). Expanding it
as a closed-form *real* image-charge series exists in principle
(Stefanescu / Sunde 1968, ch. 3.5) but the implementation is fragile
for hard contrasts because of the doubly-nested geometric expansion.
Within the ``groundfield`` engine family this regime is therefore
covered by

- ``cim`` — Complex Image Method (closed form, complex images).
- ``mom_sommerfeld`` — direct numerical Sommerfeld quadrature
  (reference engine).
- ``bem`` — boundary-element collocation with CIM kernel.

Dispatch behaviour
------------------
``solve_image_nlayer`` is a small wrapper that recognises the layer
count and forwards the call to the matching closed-form backend:

=========  ===============  ===========================
n_layers   actual backend   note
=========  ===============  ===========================
1          ``image``        homogeneous case
2          ``image_2layer`` Tagg / Sunde geometric series
≥ 3        — (raises)       use ``cim`` / ``mom_sommerfeld`` / ``bem``
=========  ===============  ===========================

The backend tag of the returned :class:`FieldResult` is rewritten to
``"image_nlayer"`` so that comparisons across engines see a single
identifier per backend selection.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- All electrodes must lie inside the upper layer.

References
----------
- Sunde 1968, ch. 3.5; Tagg 1964, ch. 5.
- Stefanescu, S. & Schlumberger, C. (1930).
- Dawalibi, F. P. & Barbeito, N. (1991).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import as_layer_stack
from groundfield.solver.image import solve_image
from groundfield.solver.image_2layer import solve_image_2layer
from groundfield.solver.result import FieldResult
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_image_nlayer"]

_log = get_logger(__name__)


def solve_image_nlayer(
    world: "World",
    engine: "Engine",
    *,
    max_terms: int = 200,
    tol: float = 1e-6,
) -> FieldResult:
    """Image-charge dispatcher for n-layer soil.

    Routes to ``image`` for ``n = 1`` and to ``image_2layer`` for
    ``n = 2``. For ``n \\ge 3`` raises a clear :class:`ValueError`
    pointing the caller to the engines that actually support that
    regime.

    Parameters
    ----------
    world
        World whose soil is any layered soil model.
    engine
        Engine configuration; ``engine.segment_length`` controls the
        discretisation.
    max_terms, tol
        Forwarded to :func:`solve_image_2layer` for ``n = 2``;
        ignored otherwise.

    Returns
    -------
    FieldResult
        ``backend`` is rewritten to ``"image_nlayer"`` so that
        cross-engine reports use one unified label.
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'image_nlayer' supports HomogeneousSoil, "
            "TwoLayerSoil, and MultiLayerSoil. "
            f"Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    stack = as_layer_stack(world.soil)
    n = stack.n_layers
    _log.info("image_nlayer: dispatching for n_layers=%d", n)

    if n == 1:
        # Forward to the homogeneous backend; HomogeneousSoil is the
        # only soil it accepts. If the user supplied a degenerate
        # MultiLayerSoil with one layer, build an equivalent
        # HomogeneousSoil on the fly.
        if isinstance(world.soil, HomogeneousSoil):
            res = solve_image(world, engine)
        else:
            from groundfield.world import World

            tmp = World(
                name=world.name,
                soil=HomogeneousSoil(resistivity=float(stack.rhos[0])),
                electrodes=list(world.electrodes),
                conductors=list(world.conductors),
                sources=list(world.sources),
                boundary=world.boundary,
            )
            res = solve_image(tmp, engine)
    elif n == 2:
        # Forward to the closed-form Tagg/Sunde series.
        if isinstance(world.soil, TwoLayerSoil):
            res = solve_image_2layer(
                world, engine, max_terms=max_terms, tol=tol
            )
        else:
            # MultiLayerSoil with exactly two layers: cast to TwoLayerSoil.
            from groundfield.world import World

            two = TwoLayerSoil(
                rho_1=float(stack.rhos[0]),
                rho_2=float(stack.rhos[1]),
                h_1=float(stack.h[0]),
            )
            tmp = World(
                name=world.name,
                soil=two,
                electrodes=list(world.electrodes),
                conductors=list(world.conductors),
                sources=list(world.sources),
                boundary=world.boundary,
            )
            res = solve_image_2layer(
                tmp, engine, max_terms=max_terms, tol=tol
            )
    else:
        raise ValueError(
            f"image_nlayer: layer count {n} ≥ 3 is not supported by "
            "the real image-charge series (Γ_1(λ) is no longer "
            "constant in λ). Use one of the engines designed for "
            "this regime: 'cim' (complex images), 'mom_sommerfeld' "
            "(direct Sommerfeld quadrature), or 'bem'."
        )

    # Rewrite backend tag so that cross-engine reports use the unified label.
    res = res.model_copy(
        update={
            "backend": "image_nlayer",
            "metadata": {**res.metadata, "n_layers": int(n), "dispatched_to": res.backend},
        }
    )
    return res
