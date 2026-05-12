"""Cross-engine comparison for self-validation.

This module provides :func:`compare_engines` — a small convenience
helper that runs the same :class:`World` through several
:class:`Engine` configurations and checks the consistency of the
results. It implements ADR-0001 (``docs/adr/0001-two-layer-method.md``):
two engines side by side, validating each other.

Usage
-----

>>> import groundfield as gf
>>> world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
>>> gf.create_electrode(world, "rod", name="g1",
...                     position=(0, 0, 0.0), length=1.5)
>>> gf.create_source(world, attached_to="g1", magnitude=1.0)
>>> report = gf.compare_engines(
...     world,
...     engines={
...         "image": gf.create_engine(backend="image", segment_length=0.05),
...         # "image_2layer": gf.create_engine(backend="image_2layer", ...),
...     },
...     rel_tolerance=0.05,
... )
>>> report.is_consistent
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "EngineComparison",
    "compare_engines",
]


@dataclass
class EngineComparison:
    """Outcome of a cross-engine comparison.

    Attributes
    ----------
    results
        Mapping ``engine_label -> FieldResult``.
    rel_tolerance
        Relative tolerance the results were checked against.
    cluster_impedance_table
        ``{cluster_root: {engine_label: Z(f0).real}}`` where ``f0`` is
        the first frequency in :attr:`FieldResult.frequencies`.
    deviations
        ``{cluster_root: max relative deviation}``.
    is_consistent
        ``True`` if and only if ``max(deviations) <= rel_tolerance``
        and no engine returned a stub result.
    notes
        Diagnostic strings (e.g. "stub backend", "frequency lists do
        not match").
    """

    results: dict[str, "FieldResult"]
    rel_tolerance: float
    cluster_impedance_table: dict[str, dict[str, float]] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    is_consistent: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a line-oriented textual summary."""
        labels = list(self.results.keys())
        lines = [
            f"Cross-engine comparison, tolerance {self.rel_tolerance*100:.1f} %",
            f"Engines: {', '.join(labels)}",
            "",
            "Cluster impedances Re{Z(f0)} in Ω:",
        ]
        for cluster, by_engine in self.cluster_impedance_table.items():
            row = "  " + cluster.ljust(12) + " "
            row += " | ".join(f"{lbl}: {by_engine.get(lbl, float('nan')):.3f}"
                              for lbl in labels)
            row += f"  Δ_max = {self.deviations.get(cluster, 0)*100:5.2f} %"
            lines.append(row)
        lines.append("")
        lines.append(
            "Consistent." if self.is_consistent
            else "INCONSISTENT — see notes."
        )
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def compare_engines(
    world: "World",
    engines: dict[str, "Engine"],
    *,
    rel_tolerance: float = 0.05,
    sample_points: np.ndarray | None = None,
) -> EngineComparison:
    """Run ``world`` through every engine and compare the results.

    Parameters
    ----------
    world
        World to evaluate. **Not modified** by this function.
    engines
        Mapping ``label -> Engine``. Must contain at least two entries.
    rel_tolerance
        Maximum allowed relative deviation of the cluster impedances
        (default 5 %). The tolerance applies to every cluster present
        in the result.
    sample_points
        Optional. Array of shape ``(M, 3)``: additional field points
        whose potentials will be compared. Deviations are reported in
        :attr:`EngineComparison.notes`.

    Returns
    -------
    EngineComparison
        Structured report (see :class:`EngineComparison`).

    Notes
    -----
    The check uses the **real part** of the cluster impedance at the
    first frequency. Clusters with ``Σ I = 0`` (purely passive
    observers) have an undefined impedance and are skipped; the skip
    is recorded in ``notes``.
    """
    if len(engines) < 2:
        raise ValueError("compare_engines requires at least 2 engines.")

    results: dict[str, FieldResult] = {
        label: engine.solve(world) for label, engine in engines.items()
    }

    cmp = EngineComparison(results=results, rel_tolerance=rel_tolerance)

    # Detect stub backends
    for label, res in results.items():
        if res.metadata.get("stub"):
            cmp.notes.append(
                f"Engine '{label}' returned a stub result "
                f"(metadata['stub']=True). Comparison not meaningful."
            )

    # Frequency lists must match
    freq_first = next(iter(results.values())).frequencies
    for label, res in results.items():
        if res.frequencies != freq_first:
            cmp.notes.append(
                f"Engine '{label}' has a different frequency list "
                f"({res.frequencies} vs. {freq_first}). Comparison "
                "uses the first frequency only."
            )

    # Cluster sets must be identical
    cluster_keys_first = {tuple(sorted(v))
                          for v in next(iter(results.values())).clusters.values()}
    for label, res in results.items():
        keys = {tuple(sorted(v)) for v in res.clusters.values()}
        if keys != cluster_keys_first:
            cmp.notes.append(
                f"Engine '{label}' reports a different cluster structure. "
                "Cluster comparison skipped."
            )
            cmp.is_consistent = False
            return cmp

    # Cluster impedances (one representative per cluster)
    representative: dict[str, str] = {}
    for label, res in results.items():
        for ename, members in res.clusters.items():
            root = sorted(members)[0]
            representative.setdefault(root, root)

    max_dev = 0.0
    for cluster_root in representative.values():
        per_engine: dict[str, float] = {}
        for label, res in results.items():
            try:
                Z = res.cluster_impedance(cluster_root)[0]
            except (KeyError, IndexError):
                continue
            if not np.isfinite(Z.real):
                continue
            per_engine[label] = float(Z.real)
        if len(per_engine) < 2:
            cmp.notes.append(
                f"Cluster '{cluster_root}': Σ I = 0 or Z undefined — "
                "skipped."
            )
            continue
        cmp.cluster_impedance_table[cluster_root] = per_engine
        zs = np.array(list(per_engine.values()))
        zmean = zs.mean()
        if zmean == 0.0:
            continue
        dev = float(np.max(np.abs(zs - zmean)) / abs(zmean))
        cmp.deviations[cluster_root] = dev
        max_dev = max(max_dev, dev)

    # Optional point-sample for the potential
    if sample_points is not None and len(results) >= 2:
        try:
            phi_table = {
                lbl: res.potential(sample_points).real
                for lbl, res in results.items()
            }
            phis = np.stack(list(phi_table.values()))
            mean = phis.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(np.abs(mean) > 0,
                               np.max(np.abs(phis - mean), axis=0) / np.abs(mean),
                               0.0)
            sample_max = float(rel.max()) if rel.size else 0.0
            cmp.notes.append(
                f"Potential point-sample at {len(sample_points)} points: "
                f"max relative deviation = {sample_max*100:.2f} %."
            )
            max_dev = max(max_dev, sample_max)
        except RuntimeError as e:
            cmp.notes.append(f"Potential sample not evaluated: {e}")

    cmp.is_consistent = (max_dev <= rel_tolerance) and not any(
        n.startswith("Engine '") and "stub result" in n for n in cmp.notes
    )
    return cmp
