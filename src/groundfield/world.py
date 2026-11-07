"""Central ``World`` object: container for the entire physics.

A ``World`` bundles everything that belongs to the physical description
of a grounding system:

- a soil model (:class:`groundfield.soil.SoilModel`),
- one or more electrodes (:class:`groundfield.geometry.Electrode`),
- optional connection conductors
  (:class:`groundfield.conductors.Conductor`),
- one or more sources (:class:`groundfield.sources.Source`),
- the boundary conditions
  (:class:`groundfield.boundary.BoundaryConditions`).

The ``World`` is intentionally **free of numerics**. The numerical
evaluation (backend selection, mesh resolution, frequency list,
tolerances) is configured in a :class:`groundfield.solver.Engine` and
applied to the world via ``world.solve(engine)``.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor
from groundfield.geometry.electrodes import Electrode, _ElectrodeBase
from groundfield.soil.models import SoilModel
from groundfield.sources import Source
from groundfield.utils.suggest import did_you_mean

# Boundary-condition values that the v0.2.0 integral / image-charge
# backends actually implement implicitly. Anything else is accepted on
# the model but ignored at solve time — :meth:`World.set_boundary_conditions`
# emits a :class:`UserWarning` to make that visible.
_DEFAULT_BOUNDARY_VALUES: dict[str, Any] = {
    "far_field": "dirichlet",
    "surface": "neumann",
    "reference_node": None,
}

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult

__all__ = ["World"]


class World(BaseModel):
    """Top-level container for a grounding field problem.

    Notes
    -----
    A ``World`` is usually not instantiated directly but built via
    :func:`groundfield.create_world`. The helper methods ``add_*`` and
    the top-level factories ``gf.create_*`` populate the container
    incrementally.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(default="world", description="Human-readable world name.")
    soil: SoilModel | None = Field(default=None, description="Soil model.")
    electrodes: list[Electrode] = Field(default_factory=list)
    conductors: list[Conductor] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    boundary: BoundaryConditions = Field(default_factory=BoundaryConditions)
    concrete_shell_corrections: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Mapping from **galvanic-cluster anchor** name to the total "
            "lumped concrete-shell series resistance "
            "$R_\\text{shell,total}$ in Ω of that cluster, populated by "
            "foundation electrodes built with a "
            "``FoundationElectrodeSpec`` whose ``concrete_rho_ohm_m`` is "
            "set (ADR-0012 V1). ``GroundingSystemSpec.build_at`` writes "
            "each electrode's own contribution first and then folds the "
            "site's entries onto the anchor once the bonds are in place, "
            "so the key is always the anchor even when the foundation is "
            "not the first electrode of its spec list. Several encased "
            "electrodes in one cluster combine **in parallel** — each "
            "leaks through its own radial shell into the soil — so the "
            "folded value is $(\\sum_i 1/R_i)^{-1}$. Consumed by "
            "``TnNetworkGenerator._build_pen_backbone`` and "
            "``_build_radial_trunk`` to inject the resistance on the "
            "corresponding PEN service drop. The field is additive — "
            "worlds built without the OSM / concrete path keep an empty "
            "dict and behave as before."
        ),
    )

    # ------------------------------------------------------------------
    # Convenience API — lookup and mutation
    # ------------------------------------------------------------------

    def get_electrode(self, name: str) -> Electrode:
        """Return an electrode by name."""
        for e in self.electrodes:
            if e.name == name:
                return e
        raise KeyError(f"Electrode '{name}' not found in world.")

    def get_conductor(self, name: str) -> Conductor:
        """Return a conductor by name."""
        for c in self.conductors:
            if c.name == name:
                return c
        raise KeyError(f"Conductor '{name}' not found in world.")

    def add_electrode(self, electrode: _ElectrodeBase) -> _ElectrodeBase:
        """Add an electrode and check for unique name."""
        if any(e.name == electrode.name for e in self.electrodes):
            raise ValueError(f"Electrode name '{electrode.name}' already taken.")
        self.electrodes.append(electrode)
        return electrode

    def add_conductor(self, conductor: Conductor) -> Conductor:
        """Add a conductor.

        Raises
        ------
        ValueError
            If the name is already taken.
        KeyError
            If ``start_electrode`` / ``end_electrode`` names no
            electrode of the world (review finding 10.66: such a
            conductor used to drop out of the network silently). The
            message names the closest node names.
        """
        if any(c.name == conductor.name for c in self.conductors):
            raise ValueError(f"Conductor name '{conductor.name}' already taken.")
        self._check_conductor_ends(conductor)
        self.conductors.append(conductor)
        return conductor

    def add_source(self, source: Source) -> Source:
        """Add a source.

        Raises
        ------
        ValueError
            If the name is already taken, the source feeds a conductor
            (a conductor is not a node), or ``attached_to`` and
            ``return_to`` are the same node.
        KeyError
            If ``attached_to`` / ``return_to`` names no electrode of the
            world (review finding 10.66). The message names the closest
            node names.
        """
        if any(s.name == source.name for s in self.sources):
            raise ValueError(f"Source name '{source.name}' already taken.")
        self._check_source_terminals(source)
        self.sources.append(source)
        return source

    # ------------------------------------------------------------------
    # Validation of the cross references (review finding 10.66)
    # ------------------------------------------------------------------

    def _node_names(self) -> list[str]:
        """Names a source or a conductor end can refer to, in world order."""
        return [e.name for e in self.electrodes]

    def _unknown_node(self, owner: str, field: str, name: str) -> KeyError:
        """``KeyError`` for a reference to a node that does not exist."""
        nodes = self._node_names()
        hint = did_you_mean(name, nodes, n=3, cutoff=0.5)
        if not hint:
            shown = ", ".join(repr(n) for n in nodes[:8])
            more = f" (+{len(nodes) - 8} more)" if len(nodes) > 8 else ""
            hint = f" Known nodes: {shown}{more}." if nodes else " It has none."
        return KeyError(
            f"{owner}: {field}={name!r} is not an electrode of world "
            f"{self.name!r}.{hint}"
        )

    def _check_conductor_ends(self, conductor: Conductor) -> None:
        nodes = set(self._node_names())
        for field in ("start_electrode", "end_electrode"):
            name = getattr(conductor, field)
            if name is not None and name not in nodes:
                raise self._unknown_node(f"Conductor {conductor.name!r}", field, name)

    def _check_source_terminals(self, source: Source) -> None:
        nodes = set(self._node_names())
        conductors = {c.name for c in self.conductors}
        owner = f"Source {source.name!r}"
        for field in ("attached_to", "return_to"):
            name = getattr(source, field)
            if name is None or name in nodes:
                continue
            if name in conductors:
                raise ValueError(
                    f"{owner}: {field}={name!r} names a conductor, not a "
                    "node. A source feeds a node of the network: attach it "
                    "to an electrode, or create a junction at the conductor "
                    "end (gf.create_junction) and connect the conductor and "
                    "the source to it. Up to 0.15.0 a source on a conductor "
                    "injected nothing."
                )
            raise self._unknown_node(owner, field, name)
        if source.return_to is not None and source.return_to == source.attached_to:
            raise ValueError(
                f"{owner}: attached_to and return_to are the same node "
                f"{source.attached_to!r}, so the source drives no current. "
                "Set return_to=None to return through remote earth."
            )

    def check(self, *, require_source: bool = True) -> None:
        """Validate the cross references and the excitation of the world.

        Called by every solve through :meth:`solve` and by the shared
        injection step of the solver pipeline, so a world edited after
        its objects were created, or loaded from JSON, is checked before
        any numerics run (review finding 10.66; up to 0.15.0 each of the
        cases below solved silently to zero currents or ``nan``
        impedances).

        Parameters
        ----------
        require_source
            Require at least one electrode and at least one current
            source (default ``True``, the state a world must be in to be
            solved). ``False`` checks only the references of a world
            that is still being assembled.

        Raises
        ------
        KeyError
            A source terminal or a conductor end names no node of the
            world; the message names the closest node names.
        ValueError
            A source feeds a conductor or returns to its own node; both
            terminals of a source are bonded into one galvanic cluster by
            ideal conductors (the source is short-circuited); a current
            source has magnitude 0 or a non-finite magnitude / phase; the
            world has no electrode or no current source (with
            ``require_source``).
        NotImplementedError
            The world contains a :class:`~groundfield.sources.VoltageSource`:
            no backend implements impressed voltages.
        """
        if require_source and self.soil is None:
            raise ValueError("World has no soil model. Set one before calling solve().")
        if require_source and not self.electrodes:
            raise ValueError("World contains no electrodes.")
        for c in self.conductors:
            self._check_conductor_ends(c)
        n_current = 0
        for src in self.sources:
            self._check_source_terminals(src)
            if src.kind != "current":
                raise NotImplementedError(
                    f"Source {src.name!r} is a {type(src).__name__}; no "
                    "backend implements impressed voltages (up to 0.15.0 it "
                    "solved to all-zero potentials). Use a CurrentSource: the "
                    "problem is linear, so a voltage U at the feeding point "
                    "is the current I = U / Z_in, with Z_in = "
                    "result.input_impedance() of a 1 A solve."
                )
            magnitude = float(src.magnitude)
            phase = float(src.phase_deg)
            if not (math.isfinite(magnitude) and math.isfinite(phase)):
                raise ValueError(
                    f"Source {src.name!r}: magnitude ({magnitude!r} A) and "
                    f"phase_deg ({phase!r}) must be finite."
                )
            if magnitude == 0.0:
                raise ValueError(
                    f"Source {src.name!r} has magnitude 0 A: it injects "
                    "nothing. Remove it or give it a non-zero magnitude."
                )
            n_current += 1
        self._check_short_circuited_sources()
        if require_source and n_current == 0:
            raise ValueError(
                f"World {self.name!r} has no current source, so every "
                "potential and current would be zero and every impedance "
                "nan. Add one with gf.create_source(world, attached_to=...)."
            )

    def _check_short_circuited_sources(self) -> None:
        """Reject a two-terminal source whose terminals share a cluster."""
        two_terminal = [s for s in self.sources if s.return_to is not None]
        if not two_terminal:
            return
        # Local import: the solver package imports this module.
        from groundfield.solver._topology import _build_clusters

        cluster_id = _build_clusters(self.electrodes, self.conductors)
        for src in two_terminal:
            ra = cluster_id.get(src.attached_to, src.attached_to)
            rb = cluster_id.get(src.return_to, src.return_to)
            if ra == rb:
                raise ValueError(
                    f"Source {src.name!r}: attached_to={src.attached_to!r} "
                    f"and return_to={src.return_to!r} are bonded into one "
                    "galvanic cluster by ideal conductors, so the source is "
                    "short-circuited and drives no current through the "
                    "soil. Give the bond a finite cross_section, or return "
                    "the current elsewhere."
                )

    def set_boundary_conditions(self, **kwargs: Any) -> BoundaryConditions:
        """Update individual fields of the boundary configuration.

        Parameters
        ----------
        **kwargs
            Fields of :class:`BoundaryConditions` (e.g. ``far_field``,
            ``surface``, ``reference_node``).

        Returns
        -------
        BoundaryConditions
            The updated boundary-conditions object.

        Warns
        -----
        UserWarning
            If any provided value differs from the defaults consumed
            by the v0.2.0 integral / image-charge backends
            (``far_field="dirichlet"``, ``surface="neumann"``,
            ``reference_node=None``). The non-default value is stored
            on the model and round-trips through serialisation, but
            no backend reads it. The fields are reserved for the
            upcoming FEM backend; see
            :class:`groundfield.boundary.BoundaryConditions` for the
            full implementation-status note.
        UserWarning
            If a field is reverted from a previously-set non-default
            value back to the default. The previous non-default value
            was never consumed by any backend, so a silent revert
            would suggest a change of behaviour that the user never
            actually experienced. The revert warning makes that
            visible.
        """
        # Snapshot the previous boundary state so we can detect both
        # "non-default value set" and "non-default value reverted to
        # default" transitions on the keys the caller touched.
        previous = self.boundary.model_dump()

        new = self.boundary.model_copy(update=kwargs)
        # Force re-validation through a fresh model construction
        self.boundary = BoundaryConditions(**new.model_dump())

        # Warn if the caller asked for a value the v0.2.0 backends do
        # not actually implement. We only warn on the keys the caller
        # touched (so a no-op call after construction stays quiet).
        non_default = {
            k: v
            for k, v in kwargs.items()
            if k in _DEFAULT_BOUNDARY_VALUES and v != _DEFAULT_BOUNDARY_VALUES[k]
        }
        if non_default:
            warnings.warn(
                "BoundaryConditions field(s) "
                f"{sorted(non_default)} set to a non-default value, "
                "but the v0.2.0 integral / image-charge backends ignore "
                "this setting and report potentials relative to remote "
                "earth (φ → 0 at infinity, Neumann at z = 0). The value "
                "is preserved on the model for forward-compatibility "
                "with the upcoming FEM backend. See "
                "groundfield.boundary.BoundaryConditions for the full "
                "implementation-status note.",
                UserWarning,
                stacklevel=2,
            )

        # Revert detection: a key the caller now sets back to the
        # default *was* previously non-default. The previous value
        # never reached any backend; warning the user closes that
        # silent-no-op feedback gap.
        reverted = {
            k: previous[k]
            for k, v in kwargs.items()
            if k in _DEFAULT_BOUNDARY_VALUES
            and v == _DEFAULT_BOUNDARY_VALUES[k]
            and previous.get(k) != _DEFAULT_BOUNDARY_VALUES[k]
        }
        if reverted:
            warnings.warn(
                "BoundaryConditions field(s) "
                f"{sorted(reverted)} reverted to the default value. The "
                "previous non-default setting "
                f"{reverted!r} was never consumed by the v0.2.0 integral "
                "/ image-charge backends, so this revert does not change "
                "any computed result. See "
                "groundfield.boundary.BoundaryConditions for the full "
                "implementation-status note.",
                UserWarning,
                stacklevel=2,
            )
        return self.boundary

    # ------------------------------------------------------------------
    # ADR-0012 concrete-shell registry — cleanup helper
    # ------------------------------------------------------------------

    def reset_concrete_corrections(self) -> dict[str, float]:
        """Clear and return the ADR-0012 V1 concrete-shell registry.

        The :attr:`concrete_shell_corrections` dict accumulates one entry
        per foundation-electrode anchor that the generator pipeline
        materialised with a non-``None`` ``concrete_rho_ohm_m`` field
        (see ADR-0012, V1 "lumped" path). Re-using a single :class:`World`
        across several :meth:`TnNetworkGenerator.build` calls — the
        canonical pattern for Monte-Carlo studies that flip
        ``concrete_rho_ohm_m`` per realisation — would otherwise leak
        stale shell resistances from earlier samples into later ones.

        The helper is the explicit, opt-in counterpart of the "build,
        solve, throw away the world" pattern: when the user *does*
        want to keep the world but re-seed the corrections, calling
        ``world.reset_concrete_corrections()`` immediately before
        ``generator.build(world=...)`` (or before mutating the
        generator config and re-building) produces a clean slate.

        Returns
        -------
        dict[str, float]
            The previous contents of the registry, before clearing. The
            returned dict is a shallow copy so the caller can inspect or
            log the dropped entries without holding a live reference to
            the (now empty) registry on the world.

        Notes
        -----
        Calling this on a world that was never touched by the
        concrete-encasement code path is a no-op; the returned dict is
        empty.
        """
        previous = dict(self.concrete_shell_corrections)
        self.concrete_shell_corrections.clear()
        return previous

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def solve(
        self,
        engine: "Engine",
        *,
        snapshot_sources: bool = True,
    ) -> "FieldResult":
        """Run the simulation with the given ``Engine``.

        Delegates to :meth:`Engine.solve`, so ``world.solve(engine)``,
        ``engine.solve(world)`` and :func:`groundfield.run_simulation`
        follow the same source-snapshot semantics: every :attr:`sources`
        entry is snapshotted before the backend runs and restored on
        exit, and the result records the sources as built.

        Parameters
        ----------
        engine
            The :class:`~groundfield.solver.engine.Engine` instance that
            drives the backend.
        snapshot_sources
            Deprecated. ``True`` (default) additionally guards the
            sources around ``engine.solve`` here, which protects against
            :class:`Engine` subclasses that override ``solve`` and
            mutate the world before delegating. ``False`` drops only
            that extra guard and emits a :class:`DeprecationWarning`:
            :meth:`Engine.solve` always takes the snapshot (it costs a
            deep copy of a few flat models), so the opt-out saves
            nothing. It will be removed no earlier than ``2027.5``.

        Returns
        -------
        FieldResult
            The result of :meth:`Engine.solve`.

        Raises
        ------
        TypeError
            If ``engine`` is not an :class:`Engine`.
        KeyError, ValueError, NotImplementedError
            If the world fails :meth:`check` (dangling source or
            conductor references, no current source, voltage sources).

        Notes
        -----
        Solving never rewrites the input world, whichever entry point is
        used (review finding 10.72; up to the calendar release only
        ``World.solve`` and ``run_simulation`` restored the sources,
        ``Engine.solve`` let a backend rewrite them).

        The roll-back is **identity-preserving** (since 0.15.0): the
        field values recorded in the snapshot are written back onto the
        very objects that :func:`~groundfield.api.create_source`
        returned, and :attr:`sources` keeps its list object. A handle
        the caller is holding therefore stays live across an arbitrary
        number of solves, so the canonical sweep pattern

        .. code-block:: python

            src = gf.create_source(world, attached_to="g1", magnitude=10.0)
            r1 = world.solve(engine)
            src.magnitude = 20.0     # still attached to ``world``
            r2 = world.solve(engine)

        does what it reads like. Before 0.15.0 the snapshot list was
        rebound onto the world, which silently detached the caller's
        handles after the first solve and made every subsequent
        mutation a no-op.
        """
        # Local import to avoid a circular dependency at module load.
        from groundfield.solver.engine import Engine, _guarded_sources
        from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

        if not isinstance(engine, Engine):
            raise TypeError(
                f"Expected an Engine, got {type(engine).__name__}. "
                "Build one with gf.create_engine(backend='image')."
            )
        # Central validation for every backend (10.66); the shared
        # injection step of the solver pipeline repeats it for callers
        # of Engine.solve.
        self.check()
        if not snapshot_sources:
            warn_external(
                "World.solve(snapshot_sources=False) is deprecated: "
                "Engine.solve always snapshots and restores the sources, "
                "so the opt-out saves nothing. The argument will be "
                f"removed no earlier than {DEPRECATION_REMOVAL}.",
                DeprecationWarning,
            )
            return engine.solve(self)
        # Engine.solve guards the sources itself; this outer guard (the
        # same identity-preserving roll-back, F31) also covers Engine
        # subclasses that override ``solve`` and mutate the world before
        # delegating to it.
        with _guarded_sources(self):
            return engine.solve(self)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Compact textual summary of the world."""
        soil_str = f"{self.soil.kind}" if self.soil is not None else "<no soil model>"
        return (
            f"World '{self.name}': soil={soil_str}, "
            f"electrodes={len(self.electrodes)}, "
            f"conductors={len(self.conductors)}, "
            f"sources={len(self.sources)}, "
            f"boundary.far_field={self.boundary.far_field}"
        )
