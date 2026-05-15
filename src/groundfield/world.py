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

import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor
from groundfield.geometry.electrodes import Electrode, _ElectrodeBase
from groundfield.soil.models import SoilModel
from groundfield.sources import Source

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
            "Mapping from electrode anchor name to lumped concrete-"
            "shell series resistance $R_\\text{shell,total}$ in Ω, "
            "populated by foundation electrodes built with a "
            "``FoundationElectrodeSpec`` whose "
            "``concrete_rho_ohm_m`` is set (ADR-0012 V1). Consumed by "
            "``TnNetworkGenerator._build_pen_backbone`` to inject the "
            "resistance on the corresponding PEN service drop. The "
            "field is additive — worlds built without the OSM / "
            "concrete path keep an empty dict and behave as before."
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
        """Add a conductor."""
        if any(c.name == conductor.name for c in self.conductors):
            raise ValueError(f"Conductor name '{conductor.name}' already taken.")
        self.conductors.append(conductor)
        return conductor

    def add_source(self, source: Source) -> Source:
        """Add a source."""
        if any(s.name == source.name for s in self.sources):
            raise ValueError(f"Source name '{source.name}' already taken.")
        self.sources.append(source)
        return source

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
            if k in _DEFAULT_BOUNDARY_VALUES
            and v != _DEFAULT_BOUNDARY_VALUES[k]
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
        # silent-no-op feedback gap (fourth 2026-05-12 audit pass).
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
    # Run
    # ------------------------------------------------------------------

    def solve(
        self,
        engine: "Engine",
        *,
        snapshot_sources: bool = True,
    ) -> "FieldResult":
        """Run the simulation with the given ``Engine``.

        Delegates to :meth:`Engine.solve`, so users may write either
        ``world.solve(engine)`` or ``engine.solve(world)``.

        Parameters
        ----------
        engine
            The :class:`~groundfield.solver.engine.Engine` instance that
            drives the backend.
        snapshot_sources
            If ``True`` (default), every :attr:`sources` entry is
            deep-copied before the backend runs and restored on exit.
            This defends against backends that mutate
            ``Source.return_to`` in flight (see
            :class:`groundfield.generators.measurement.MeasurementSetupConfig.build`).
            Power users who have verified that their backend does not
            mutate the source list (typical in long
            :func:`~groundfield.engines.compare_engines` sweeps or
            :func:`~groundfield.engines.convergence_study` runs) may set
            ``snapshot_sources=False`` to skip the deep-copy cost
            (sixth 2026-05-14 audit pass).

        Notes
        -----
        The default ``snapshot_sources=True`` makes the contract
        explicit: solving never rewrites the input world. The opt-out
        is documented in ``docs/concepts.md`` ("Engine re-use across
        ``World.solve`` calls") and exercised in
        ``notebooks/32_audit_pass6_fixes.ipynb``.
        """
        # Local import to avoid a circular dependency at module load.
        from groundfield.solver.engine import Engine

        if not isinstance(engine, Engine):
            raise TypeError(
                f"Expected an Engine, got {type(engine).__name__}. "
                "Build one with gf.create_engine(backend='image')."
            )
        if not snapshot_sources:
            # Caller has opted out — backends are now contractually
            # responsible for not mutating ``self.sources``.
            return engine.solve(self)
        # Snapshot every source via Pydantic's deep-copy semantics so
        # backends that mutate a source field in flight cannot leak
        # the change back into the caller's world (fifth 2026-05-13
        # audit pass; opt-out sixth 2026-05-14 audit pass).
        sources_snapshot = [s.model_copy(deep=True) for s in self.sources]
        try:
            return engine.solve(self)
        finally:
            self.sources = sources_snapshot

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Compact textual summary of the world."""
        soil_str = (
            f"{self.soil.kind}" if self.soil is not None else "<no soil model>"
        )
        return (
            f"World '{self.name}': soil={soil_str}, "
            f"electrodes={len(self.electrodes)}, "
            f"conductors={len(self.conductors)}, "
            f"sources={len(self.sources)}, "
            f"boundary.far_field={self.boundary.far_field}"
        )
