"""Central ``World`` object: container for the entire physics.

A ``World`` bundles everything that belongs to the physical description
of a grounding system:

- a soil model (:class:`groundfield.soil.SoilModel`),
- one or more electrodes (:class:`groundfield.geometry.Electrode`),
- optional leakage-free junction nodes
  (:class:`groundfield.geometry.junction.Junction`),
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
import weakref
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor
from groundfield.geometry.electrodes import Electrode, _ElectrodeBase
from groundfield.geometry.junction import Junction
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

#: Name prefixes of the solver's internal owners and nodes: ``__layer_`` for
#: the part of an electrode in one soil layer, ``__cond_`` for the pseudo-nodes
#: of a distributed conductor. A user node (electrode or junction) with such a
#: name could take the place of an internal one and broke the solve with a
#: singular matrix (audit A3S-03), so these prefixes are reserved.
RESERVED_NAME_PREFIXES: tuple[str, ...] = ("__layer_", "__cond_")

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from collections.abc import Container

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
    junctions: list[Junction] = Field(
        default_factory=list,
        description=(
            "Leakage-free network nodes (cable joints, instrument "
            "terminals). They share the node namespace with the "
            "electrodes; conductors and sources refer to them by name. "
            "Omitted from the serialised world while empty, so worlds "
            "without junctions keep their JSON and fingerprint."
        ),
    )
    conductors: list[Conductor] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    boundary: BoundaryConditions = Field(default_factory=BoundaryConditions)
    concrete_shell_corrections: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "**Deprecated** (since 2026.11, removal no earlier than 2027.5): "
            "the solver consumes concrete shells only as the per-length "
            "coefficient ``concrete_shell_coefficient_ohm_m`` of the "
            "electrodes and never reads this registry; solving a world with "
            "entries emits a DeprecationWarning. Still loaded from JSON. "
            "Mapping from **galvanic-cluster anchor** name to the total "
            "lumped concrete-shell series resistance "
            "$R_\\text{shell,total}$ in Ω of that cluster (ADR-0012 V1). "
            "Up to 0.15.0 the generators filled it for every encased "
            '``FoundationElectrodeSpec`` (``concrete_model="lumped"``), '
            "folded onto the site's anchor, several encased electrodes "
            "**in parallel** ($(\\sum_i 1/R_i)^{-1}$), and the TN "
            "generator put the value on the PEN service drop. Since "
            "2026.11 no generator writes it: the shell is the coefficient "
            "on every wire of the foundation (cross-braces included), and "
            "``concrete_model`` is deprecated as well. Worlds without "
            "entries keep an empty dict."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_empty_junctions(self, handler):  # noqa: ANN001, ANN202
        """Serialise ``junctions`` only when the world has any.

        Keeps the JSON (and :func:`~groundfield.io.world_fingerprint`) of
        every world without junctions identical to earlier releases,
        which also load it (``extra="forbid"``).
        """
        data = handler(self)
        if isinstance(data, dict) and not data.get("junctions"):
            data.pop("junctions", None)
        return data

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

    def get_junction(self, name: str) -> Junction:
        """Return a junction by name."""
        for j in self.junctions:
            if j.name == name:
                return j
        raise KeyError(f"Junction '{name}' not found in world.")

    def add_electrode(self, electrode: _ElectrodeBase) -> _ElectrodeBase:
        """Add an electrode and check for unique name.

        Electrodes and junctions share one node namespace.

        Raises
        ------
        ValueError
            If the name is already taken by an electrode or a junction, or
            starts with a reserved prefix (:data:`~groundfield.world.RESERVED_NAME_PREFIXES`).
        """
        _check_reserved_name(electrode.name, "Electrode")
        index = _name_index(self)
        if index.taken(self, "electrodes", electrode.name):
            raise ValueError(f"Electrode name '{electrode.name}' already taken.")
        if index.taken(self, "junctions", electrode.name):
            raise ValueError(
                f"Electrode name '{electrode.name}' already taken by a junction."
            )
        self.electrodes.append(electrode)
        index.added(self, "electrodes", electrode.name)
        return electrode

    def add_junction(self, junction: Junction) -> Junction:
        """Add a leakage-free junction node.

        Raises
        ------
        ValueError
            If the name is already taken by an electrode or a junction
            (one node namespace), or starts with a reserved prefix
            (:data:`~groundfield.world.RESERVED_NAME_PREFIXES`).
        """
        _check_reserved_name(junction.name, "Junction")
        index = _name_index(self)
        if index.taken(self, "junctions", junction.name):
            raise ValueError(f"Junction name '{junction.name}' already taken.")
        if index.taken(self, "electrodes", junction.name):
            raise ValueError(
                f"Junction name '{junction.name}' already taken by an electrode."
            )
        self.junctions.append(junction)
        index.added(self, "junctions", junction.name)
        return junction

    def add_conductor(self, conductor: Conductor) -> Conductor:
        """Add a conductor.

        Raises
        ------
        ValueError
            If the name is already taken.
        KeyError
            If ``start_electrode`` / ``end_electrode`` names no
            electrode or junction of the world (review finding 10.66: such a
            conductor used to drop out of the network silently). The
            message names the closest node names.
        """
        index = _name_index(self)
        if index.taken(self, "conductors", conductor.name):
            raise ValueError(f"Conductor name '{conductor.name}' already taken.")
        self._check_conductor_ends(conductor, index.node_view(self))
        self.conductors.append(conductor)
        index.added(self, "conductors", conductor.name)
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
            If ``attached_to`` / ``return_to`` names no electrode or
            junction of the world (review finding 10.66). The message
            names the closest node names.
        """
        index = _name_index(self)
        if index.taken(self, "sources", source.name):
            raise ValueError(f"Source name '{source.name}' already taken.")
        self._check_source_terminals(source)
        self.sources.append(source)
        index.added(self, "sources", source.name)
        return source

    # ------------------------------------------------------------------
    # Validation of the cross references (review finding 10.66)
    # ------------------------------------------------------------------

    def _node_names(self) -> list[str]:
        """Names a source or a conductor end can refer to, in world order."""
        return [e.name for e in self.electrodes] + [j.name for j in self.junctions]

    def _unknown_node(self, owner: str, field: str, name: str) -> KeyError:
        """``KeyError`` for a reference to a node that does not exist."""
        nodes = self._node_names()
        hint = did_you_mean(name, nodes, n=3, cutoff=0.5)
        if not hint:
            shown = ", ".join(repr(n) for n in nodes[:8])
            more = f" (+{len(nodes) - 8} more)" if len(nodes) > 8 else ""
            hint = f" Known nodes: {shown}{more}." if nodes else " It has none."
        return KeyError(
            f"{owner}: {field}={name!r} is not an electrode or junction of "
            f"world {self.name!r}.{hint}"
        )

    def _check_conductor_ends(
        self, conductor: Conductor, nodes: "Container[str] | None" = None
    ) -> None:
        if nodes is None:
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
            Require at least one electrode and at least one source of
            non-zero magnitude (default ``True``, the state a world must
            be in to be solved). ``False`` checks only the references of
            a world that is still being assembled.

        Raises
        ------
        KeyError
            A source terminal or a conductor end names no node of the
            world; the message names the closest node names.
        ValueError
            A name is used twice: two electrodes, two junctions, an
            electrode and a junction (one node namespace), two conductors
            or two sources (audit round 2, A2S-06: a world loaded from
            JSON or edited through its lists with two electrodes of one
            name failed in the solver with a singular matrix, an electrode
            and a junction of one name were merged into one node).
            An electrode or junction name starts with a reserved prefix
            (:data:`~groundfield.world.RESERVED_NAME_PREFIXES`, audit A3S-03).
            A source feeds a conductor or returns to its own node; both
            terminals of a source are bonded into one galvanic cluster by
            ideal conductors (the source is short-circuited); voltage
            sources form a loop, through ideal bonds or through remote
            earth (two of them on one cluster to remote earth, a ring of
            two-terminal ones), which fixes one potential difference
            twice; a current source has magnitude 0; a source has a
            non-finite magnitude / phase; the world has no electrode or
            no source of non-zero magnitude (with ``require_source``).

        Notes
        -----
        A :class:`~groundfield.sources.VoltageSource` of 0 V is accepted:
        it is an ideal short between its terminals (or to remote earth)
        whose current the solve returns (up to 0.15.0 voltage sources
        were dropped with a warning and solved to zero potentials).
        """
        self._check_unique_names()
        if require_source and self.soil is None:
            raise ValueError("World has no soil model. Set one before calling solve().")
        if require_source and not self.electrodes:
            raise ValueError("World contains no electrodes.")
        nodes = set(self._node_names())
        for c in self.conductors:
            self._check_conductor_ends(c, nodes)
        n_driving = 0
        zero_volt: list[str] = []
        for src in self.sources:
            self._check_source_terminals(src)
            unit = "V" if src.kind == "voltage" else "A"
            magnitude = float(src.magnitude)
            phase = float(src.phase_deg)
            if not (math.isfinite(magnitude) and math.isfinite(phase)):
                raise ValueError(
                    f"Source {src.name!r}: magnitude ({magnitude!r} {unit}) and "
                    f"phase_deg ({phase!r}) must be finite."
                )
            if magnitude == 0.0:
                if src.kind == "voltage":
                    # An ideal short (0 V): its current is solved for.
                    zero_volt.append(src.name)
                    continue
                raise ValueError(
                    f"Source {src.name!r} has magnitude 0 A: it injects "
                    "nothing. Remove it or give it a non-zero magnitude."
                )
            n_driving += 1
        self._check_short_circuited_sources()
        if require_source and n_driving == 0:
            if zero_volt:
                raise ValueError(
                    f"World {self.name!r} has no source with a non-zero "
                    f"magnitude: its voltage source(s) {zero_volt} impress 0 V "
                    "and it has no current source, so every potential and "
                    "current would be zero. Give a source a non-zero magnitude."
                )
            raise ValueError(
                f"World {self.name!r} has no current source and no voltage "
                "source, so every potential and current would be zero and "
                "every impedance nan. Add one with gf.create_source(world, "
                "attached_to=...)."
            )

    def _check_unique_names(self) -> None:
        """Reject a name used twice in one namespace (audit round 2, A2S-06).

        :meth:`add_electrode`, :meth:`add_junction`, :meth:`add_conductor`
        and :meth:`add_source` enforce this when objects are added; a world
        loaded from JSON or edited through its lists is checked here.
        """
        kinds: dict[str, list[str]] = {}
        for e in self.electrodes:
            kinds.setdefault(e.name, []).append("electrode")
        for j in self.junctions:
            kinds.setdefault(j.name, []).append("junction")
        for name, k in kinds.items():
            _check_reserved_name(name, k[0].capitalize(), world=self.name)
        clashes = [
            f"{name!r} ({' and '.join(k)})" for name, k in kinds.items() if len(k) > 1
        ]
        if clashes:
            raise ValueError(
                f"World {self.name!r}: node name(s) used more than once: "
                f"{', '.join(clashes)}. Electrodes and junctions share one "
                "namespace; rename them."
            )
        for label, items in (("conductor", self.conductors), ("source", self.sources)):
            counts = Counter(x.name for x in items)
            twice = sorted(name for name, n in counts.items() if n > 1)
            if twice:
                raise ValueError(
                    f"World {self.name!r}: {label} name(s) used more than once: "
                    f"{', '.join(repr(n) for n in twice)}; rename them."
                )

    def _check_short_circuited_sources(self) -> None:
        """Reject short-circuited sources and loops of voltage sources.

        A two-terminal source whose terminals share a cluster is
        short-circuited. Voltage sources are edges between cluster roots
        and remote earth: an edge that closes a cycle (two sources on one
        cluster to remote earth, a ring of two-terminal sources) fixes
        one potential difference twice, and the nodal system is singular
        (inconsistent for unequal voltages).
        """
        two_terminal = [s for s in self.sources if s.return_to is not None]
        voltage = [s for s in self.sources if s.kind == "voltage"]
        if not two_terminal and len(voltage) < 2:
            return
        # Local import: the solver package imports this module.
        from groundfield.solver._topology import _build_clusters

        cluster_id = _build_clusters(
            self.electrodes, self.conductors, junctions=self.junctions
        )
        for src in two_terminal:
            ra = cluster_id.get(src.attached_to, src.attached_to)
            rb = cluster_id.get(src.return_to, src.return_to)
            if ra == rb:
                drives = (
                    "would impose a voltage across one equipotential"
                    if src.kind == "voltage"
                    else "drives no current through the soil"
                )
                raise ValueError(
                    f"Source {src.name!r}: attached_to={src.attached_to!r} "
                    f"and return_to={src.return_to!r} are bonded into one "
                    "galvanic cluster by ideal conductors, so the source is "
                    f"short-circuited and {drives}. Give the bond a finite "
                    "cross_section, or return the current elsewhere."
                )
        self._check_voltage_loops(voltage, cluster_id)

    @staticmethod
    def _check_voltage_loops(voltage, cluster_id: dict[str, str]) -> None:
        """Reject voltage sources that close a loop (union-find over clusters)."""
        earth = object()
        parent: dict[object, object] = {}
        members: dict[object, list[str]] = {}

        def find(x: object) -> object:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for src in voltage:
            a = find(cluster_id.get(src.attached_to, src.attached_to))
            r = find(
                earth
                if src.return_to is None
                else cluster_id.get(src.return_to, src.return_to)
            )
            if a == r:
                loop = members.get(a, []) + [src.name]
                raise ValueError(
                    f"Voltage source {src.name!r} closes a loop of voltage "
                    f"sources ({', '.join(repr(n) for n in loop)}) through "
                    "ideal bonds or remote earth: the loop fixes one "
                    "potential difference twice, so the nodal system has no "
                    "unique solution. Remove one source of the loop, or put "
                    "a finite conductor (a junction and a lumped series "
                    "resistance) in series with one of them."
                )
            parent[a] = r
            members[r] = members.pop(r, []) + members.pop(a, []) + [src.name]

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

        Up to 0.15.0 the :attr:`concrete_shell_corrections` dict
        accumulated one entry per foundation-electrode anchor that the
        generator pipeline materialised with a non-``None``
        ``concrete_rho_ohm_m`` field (ADR-0012, V1 "lumped" path), and a
        :class:`World` re-used across several builds leaked stale shell
        resistances into later samples. Since 2026.11 the registry is
        deprecated, no generator writes it and no solver reads it (the
        shell is the per-length coefficient of the encased wires); the
        helper clears entries of worlds loaded from 0.15.0 JSON, which
        would otherwise warn at every solve.

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
        KeyError, ValueError
            If the world fails :meth:`check` (dangling source or
            conductor references, no source of non-zero magnitude, a
            short-circuited source, a loop of voltage sources).
        BackendCapabilityError
            If the backend cannot represent the world (for example a
            voltage source on ``fem``).

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
        junctions = f"junctions={len(self.junctions)}, " if self.junctions else ""
        return (
            f"World '{self.name}': soil={soil_str}, "
            f"electrodes={len(self.electrodes)}, "
            f"{junctions}"
            f"conductors={len(self.conductors)}, "
            f"sources={len(self.sources)}, "
            f"boundary.far_field={self.boundary.far_field}"
        )


def _check_reserved_name(name: str, kind: str, *, world: str | None = None) -> None:
    """Reject a node name with a reserved prefix (audit A3S-03).

    Parameters
    ----------
    name
        Name of an electrode or junction.
    kind
        ``"Electrode"`` or ``"Junction"`` (for the message).
    world
        Name of the world, if the check runs on a whole world.

    Raises
    ------
    ValueError
        If ``name`` starts with one of :data:`~groundfield.world.RESERVED_NAME_PREFIXES`.
    """
    for prefix in RESERVED_NAME_PREFIXES:
        if name.startswith(prefix):
            where = f"World {world!r}: " if world is not None else ""
            raise ValueError(
                f"{where}{kind} name {name!r} starts with the reserved prefix "
                f"{prefix!r}: the solver names the part of an electrode in one "
                "soil layer '__layer_<electrode>__L<k>' and the nodes of a "
                "distributed conductor '__cond_<conductor>__...', and a user "
                "node of such a name could take the place of an internal one "
                "(the solve failed with a singular matrix). Rename it."
            )


# ---------------------------------------------------------------------
# Name index of the add_* methods (O(1) per add)
# ---------------------------------------------------------------------


class _NameIndex:
    """Names of the objects in a world's lists, for the ``add_*`` checks.

    Building a network adds thousands of objects one by one, and a linear
    scan per add made that quadratic: 1000 houses took 13 s, 11 s of them
    in these checks (audit round 3; generated mesh foundations are one
    electrode per wire since 2026.11). The index is a cache, not state of
    the world: it lives outside the model (equality, copies, pickles and
    JSON are untouched) and is trusted for a list only while that same
    list object has the length it had when the index was last synced, so
    an append or a new list made by hand triggers a rebuild. A name found
    in it is confirmed by a scan before an error is raised, so an object
    renamed in place never causes a false "already taken"; a duplicate
    created by such a rename is reported by :meth:`World.check`, which
    every solve runs.
    """

    __slots__ = ("world", "_lists")

    _FIELDS = ("electrodes", "junctions", "conductors", "sources")

    def __init__(self, world: World) -> None:
        self.world = weakref.ref(world)
        # field -> (the list object, its length, the set of its names)
        self._lists: dict[str, tuple[list, int, set[str]]] = {}

    def _names(self, world: World, field: str) -> set[str]:
        items = getattr(world, field)
        entry = self._lists.get(field)
        if entry is None or entry[0] is not items or entry[1] != len(items):
            entry = (items, len(items), {x.name for x in items})
            self._lists[field] = entry
        return entry[2]

    def taken(self, world: World, field: str, name: str) -> bool:
        """Whether ``name`` is used in the list ``field`` (confirmed by a scan)."""
        if name not in self._names(world, field):
            return False
        if any(x.name == name for x in getattr(world, field)):
            return True
        self._lists.pop(field, None)  # stale (an object renamed in place)
        return False

    def added(self, world: World, field: str, name: str) -> None:
        """Record that ``name`` was appended to the list ``field``."""
        items = getattr(world, field)
        entry = self._lists.get(field)
        if entry is not None and entry[0] is items and entry[1] == len(items) - 1:
            entry[2].add(name)
            self._lists[field] = (items, len(items), entry[2])
        else:
            self._lists.pop(field, None)

    def node_view(self, world: World) -> "_NodeNames":
        """The node names (electrodes and junctions) as a container."""
        return _NodeNames(
            self._names(world, "electrodes"), self._names(world, "junctions")
        )


class _NodeNames:
    """``in`` over the electrode and junction names of a world."""

    __slots__ = ("electrodes", "junctions")

    def __init__(self, electrodes: set[str], junctions: set[str]) -> None:
        self.electrodes = electrodes
        self.junctions = junctions

    def __contains__(self, name: object) -> bool:
        return name in self.electrodes or name in self.junctions


#: One index per live world, keyed by ``id(world)`` and dropped by a
#: finalizer when the world is collected (``World`` is not hashable).
_NAME_INDEXES: dict[int, _NameIndex] = {}


def _name_index(world: World) -> _NameIndex:
    """The name index of ``world`` (created on first use)."""
    index = _NAME_INDEXES.get(id(world))
    if index is None or index.world() is not world:
        index = _NameIndex(world)
        _NAME_INDEXES[id(world)] = index
        weakref.finalize(world, _NAME_INDEXES.pop, id(world), None)
    return index
