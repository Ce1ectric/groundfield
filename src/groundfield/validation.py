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
...         "mom": gf.create_engine(backend="mom", segment_length=0.05),
...     },
...     rel_tolerance=0.05,
... )
>>> report.is_consistent
True

At least **two** engines are required — a single-entry mapping raises
``ValueError``, because one engine cannot validate itself.
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
        ``{cluster_root: worst pairwise relative deviation}`` with
        ``dev = (max(Z) - min(Z)) / min(|Z|)`` over the engines — the
        largest relative disagreement between any two engines,
        referenced to the smaller of the two values (see
        :func:`compare_engines` for the change in 0.15.0). Clusters
        whose impedance is undefined for at least one engine do not
        appear here.
    is_consistent
        ``True`` if and only if **at least one** quantity was actually
        evaluated (a cluster impedance, or a sample point with a defined
        reference), every deviation is ``<= rel_tolerance``, and no
        engine returned a stub result. A comparison in which nothing
        could be evaluated is reported as *not* consistent — see
        :func:`compare_engines`.
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
        Maximum allowed **pairwise** relative deviation of the cluster
        impedances (default 5 %), i.e. the gate is

        .. math::

            \\max_{i,j} \\frac{|Z_i - Z_j|}{\\min_k |Z_k|}
            \\;\\le\\; \\texttt{rel_tolerance}

        over the engines :math:`i, j, k`. Two engines differing by 5 %
        of the smaller value therefore sit exactly on the threshold.
        The tolerance applies to every cluster present in the result.
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

    **Metric definition (changed in 0.15.0).** The deviation is the
    full spread of the compared values normalised by the *smallest*
    magnitude among them,
    ``dev = (max(Z) - min(Z)) / min(|Z|)``, which is identical to the
    worst pairwise relative deviation ``max_ij |Z_i - Z_j| / |Z_j|``
    for same-sign impedances. Up to 0.14.x the deviation was measured
    against the ensemble *mean*
    (``max_k |Z_k - mean(Z)| / |mean(Z)|``), which for two engines
    reports exactly half the true disagreement and shrinks further as
    engines are added — a ``rel_tolerance=0.05`` gate then admitted a
    10.5 % disagreement. Reported deviations are therefore roughly
    twice as large as in 0.14.x for the same world; a comparison that
    was marginally green may now legitimately turn red.

    **Vacuous comparisons (changed in 0.15.0).** A report is
    ``is_consistent=True`` only if at least one quantity was actually
    evaluated — a cluster impedance, or a ``sample_points`` potential
    with a non-zero reference. A world without sources (or with a
    misspelled ``Source.attached_to``) has no defined cluster impedance
    anywhere, so nothing can be validated; such a run now yields
    ``is_consistent=False`` plus a "comparison vacuous" note instead of
    a false green (up to 0.14.x the running maximum deviation never left
    its initial ``0.0`` and passed the tolerance gate).
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
    # Number of quantities for which a deviation was actually computed
    # (cluster impedances + sample points with a defined reference).
    # ``max_dev`` alone cannot distinguish "everything agreed" from
    # "nothing was evaluated" — both leave it at 0.0.
    n_compared = 0
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
        # Worst *pairwise* relative deviation, referenced to the
        # smallest magnitude among the compared impedances:
        #   dev = max_ij |Z_i - Z_j| / min_k |Z_k|
        #       = (max Z - min Z) / min |Z|      (same-sign Z).
        # Measuring against the ensemble mean instead (as up to 0.14.x)
        # halves the reported number for two engines and lets a 5 %
        # gate pass a 10.5 % disagreement.
        zs = np.asarray(list(per_engine.values()), dtype=float)
        z_ref = float(np.min(np.abs(zs)))
        if z_ref == 0.0:
            cmp.notes.append(
                f"Cluster '{cluster_root}': Z = 0 for at least one "
                "engine — relative deviation undefined, skipped."
            )
            continue
        dev = float(zs.max() - zs.min()) / z_ref
        cmp.deviations[cluster_root] = dev
        max_dev = max(max_dev, dev)
        n_compared += 1

    # Optional point-sample for the potential
    if sample_points is not None and len(results) >= 2:
        try:
            phi_table = {
                lbl: res.potential(sample_points).real
                for lbl, res in results.items()
            }
            phis = np.stack(list(phi_table.values()))
            # Same pairwise metric as for the cluster impedances:
            # spread over the smallest magnitude, per sample point.
            spread = phis.max(axis=0) - phis.min(axis=0)
            phi_ref = np.min(np.abs(phis), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(phi_ref > 0.0, spread / phi_ref, 0.0)
            n_undefined = int(np.count_nonzero(phi_ref <= 0.0))
            sample_max = float(rel.max()) if rel.size else 0.0
            note = (
                f"Potential point-sample at {len(sample_points)} points: "
                f"max relative deviation = {sample_max*100:.2f} % "
                "(max pairwise, referenced to min |phi|)."
            )
            if n_undefined:
                note += (
                    f" {n_undefined} point(s) excluded: phi = 0 for at "
                    "least one engine, relative deviation undefined."
                )
            cmp.notes.append(note)
            max_dev = max(max_dev, sample_max)
            n_compared += int(rel.size) - n_undefined
        except RuntimeError as e:
            cmp.notes.append(f"Potential sample not evaluated: {e}")

    # A comparison that evaluated *nothing* has validated nothing:
    # ``max_dev`` is still its initial 0.0, which would otherwise pass
    # the tolerance gate and report a false green (typical cause: the
    # world has no source, or ``Source.attached_to`` is misspelled, so
    # every cluster carries Σ I = 0 and was skipped above).
    if n_compared == 0:
        cmp.notes.append(
            "No comparable quantity (every cluster impedance undefined, "
            "no usable sample point) — comparison vacuous, nothing was "
            "validated."
        )
    cmp.is_consistent = (
        n_compared > 0
        and (max_dev <= rel_tolerance)
        and not any(
            n.startswith("Engine '") and "stub result" in n
            for n in cmp.notes
        )
    )
    return cmp
