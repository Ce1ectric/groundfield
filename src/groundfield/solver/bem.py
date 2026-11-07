"""Deprecated backend name ``bem``: an alias of the point-matching ``mom``.

Status (review pass 10, 10.76)
------------------------------
``bem`` assembled **the same** point-matching system as
:mod:`groundfield.solver.mom` and solved it with the same routine: one
pulse current per wire segment, the reaction matrix

$$
Z_{ij} \\;=\\; \\frac{\\rho_1}{4\\pi}\\, G(\\mathbf{r}_i, \\mathbf{r}_j)
\\quad (i \\ne j),
$$

i.e. the potential at the midpoint of segment $i$ of a unit *point*
source at the midpoint of segment $j$, the analytic line
self-potential on the diagonal, and the constraints

$$
\\sum_{j=1}^{N} Z_{ij}\\, I_j \\;=\\; \\varphi_c
\\quad \\forall\\, i \\in c,
\\qquad \\sum_{j \\in c} I_j \\;=\\; I_{c,\\text{in}},
$$

over the galvanic clusters $c$ (a source-free cluster floats with
$I_{c,\\text{in}} = 0$, review pass 10, 10.29). $G$ is the closed-form
kernel of the image family: $1/r + 1/r_{\\text{air}}$ for homogeneous
soil and the Tagg/Sunde series of
:mod:`groundfield.solver.image_2layer` for two layers. ``bem`` and
``mom`` were one computation under two names; the measured difference
was 0.0 to 2.3e-16 relative, from evaluating the segment potentials as
``Z @ I`` instead of through the reaction operator.

The name is therefore a **deprecated alias** of ``mom``:
``Engine(backend="bem")`` and :func:`solve_bem` run ``mom`` and emit a
:class:`DeprecationWarning`; the alias is removed no earlier than
groundfield 2027.5. The numbers are those of ``mom`` (bit for bit).
A cross-validation table must not count ``bem`` and ``mom`` as two
engines.

Up to 0.15.0 this module described its matrix as a *collocation* of
$G$ that differed from a *Galerkin* average in ``mom``. Neither half
was true: both backends point-match at the segment midpoints with a
point-source near field (10.31). A genuine collocation boundary-element
method in the sense of Colominas, Navarrina & Casteleiro (2007, 2012)
integrates the source exactly along each segment; that would raise the
near-field coupling by about 1.6 % to 1.9 % on a 3 m rod and is not
implemented.

References
----------
- Colominas, I., Navarrina, F., & Casteleiro, M. (2007). Numerical
  simulation of transferred potentials in earthing grids considering
  layered soil models. *IEEE PWRD* 22(3).
- Colominas, I., París, J., Navarrina, F., & Casteleiro, M. (2012).
  Improvement of computer methods for grounding analysis in layered
  soils by using high-efficient convergence acceleration techniques.
  *Adv. Eng. Soft.* 44.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from groundfield.solver.mom import solve_mom
from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = ["solve_bem"]


def solve_bem(world: World, engine: Engine) -> FieldResult:
    """Deprecated point-matching method-of-moments solver (alias of ``mom``).

    Runs :func:`groundfield.solver.mom.solve_mom` (review pass 10,
    10.76); ``bem`` assembled the identical point-matching system, see
    the module docstring.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` (m) controls the
        discretisation, ``engine.image_max_terms`` /
        ``engine.image_series_tol`` the Tagg/Sunde truncation for two
        layers.

    Returns
    -------
    FieldResult
        The ``mom`` result, labelled ``backend="bem"`` with
        ``metadata['dispatched_to'] = 'mom'``.

    Warns
    -----
    DeprecationWarning
        Always: use ``backend="mom"`` (:func:`solve_mom`) instead; the
        alias is removed no earlier than groundfield 2027.5.
    """
    warn_external(
        "Backend name 'bem' (groundfield.solver.bem.solve_bem) is deprecated; "
        "use 'mom' (solve_mom) instead — it assembles the identical "
        "point-matching system. The alias will be removed no earlier than "
        f"{DEPRECATION_REMOVAL}.",
        DeprecationWarning,
    )
    res = solve_mom(world, engine)
    return res.model_copy(
        update={
            "backend": "bem",
            "metadata": {**res.metadata, "dispatched_to": res.backend},
        }
    )
