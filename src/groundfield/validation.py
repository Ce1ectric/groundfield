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

What is compared (review pass 10, 10.71)
-----------------------------------------
Every compared quantity is complex and is compared at **every**
frequency, real and imaginary part separately:

* the cluster impedance of every cluster
  (:meth:`FieldResult.cluster_impedance`, Ω),
* the input impedance of every current source
  (:meth:`FieldResult.input_impedance`, Ω),
* optionally the potential at ``sample_points`` (V).

Each (quantity, frequency, component) is one row of the report's
tolerance table (:attr:`EngineComparison.rows`). Up to 0.15.0 only the
real part of the cluster impedance at the *first* frequency was
compared, so an earth-return model that doubled the reactance at
1 kHz passed, engines with different frequency lists were compared at
different frequencies, and a non-finite result was silently dropped.
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
    "ComparisonRow",
    "EngineComparison",
    "compare_engines",
]

#: Default floor of the component reference, as a fraction of the
#: smallest compared magnitude (see :func:`compare_engines`).
DEFAULT_COMPONENT_FLOOR = 1e-3

_COMPONENTS = (("Re", np.real), ("Im", np.imag))


@dataclass(frozen=True)
class ComparisonRow:
    """One row of the tolerance table of :class:`EngineComparison`.

    Attributes
    ----------
    quantity
        ``"Z_cluster[<cluster root>]"`` (Ω), ``"Z_in[<source>]"`` (Ω) or
        ``"phi[<sample point index>]"`` (V; the worst sample point).
    frequency
        Frequency in Hz.
    component
        ``"Re"`` or ``"Im"``.
    values
        ``engine_label -> value of the component`` (Ω or V).
    reference
        Denominator of the relative deviation (same unit as
        ``values``): ``max(min_k |c_k|, component_floor · min_k |Z_k|)``.
    deviation
        Pairwise spread ``(max c - min c) / reference``; ``inf`` when an
        engine returned a non-finite value while another did not.
    tolerance
        The relative tolerance the row was checked against.
    passed
        ``deviation <= tolerance``.
    """

    quantity: str
    frequency: float
    component: str
    values: dict[str, float]
    reference: float
    deviation: float
    tolerance: float
    passed: bool


@dataclass
class EngineComparison:
    """Outcome of a cross-engine comparison.

    Attributes
    ----------
    results
        Mapping ``engine_label -> FieldResult``.
    rel_tolerance
        Relative tolerance the results were checked against.
    frequencies
        The common frequency list in Hz.
    rows
        The explicit tolerance table: one :class:`ComparisonRow` per
        compared quantity, frequency and component (real / imaginary
        part).
    impedances
        ``{"Z_cluster[<root>]" | "Z_in[<source>]": {engine_label: [Z(f)
        per frequency]}}`` — the complex values the rows were built
        from, in Ω.
    cluster_impedance_table
        ``{cluster_root: {engine_label: Re Z(f0)}}`` with ``f0`` the
        first frequency — a compact summary kept for compatibility; the
        comparison itself covers :attr:`rows`.
    deviations
        ``{cluster_root: worst deviation}`` of the cluster impedance over
        all frequencies and both components. Clusters whose impedance is
        undefined for every engine (purely passive observers) do not
        appear here.
    input_deviations
        ``{source_name: worst deviation}`` of the input impedance, same
        convention.
    is_consistent
        ``True`` if and only if **at least one** quantity was actually
        evaluated, every row passed (no deviation above
        ``rel_tolerance``, no non-finite value next to a finite one), and
        no engine returned a stub result. A comparison in which nothing
        could be evaluated is reported as *not* consistent — see
        :func:`compare_engines`.
    notes
        Diagnostic strings (stub results, skipped or non-finite
        quantities, the sample-point summary).
    """

    results: dict[str, "FieldResult"]
    rel_tolerance: float
    frequencies: list[float] = field(default_factory=list)
    rows: list[ComparisonRow] = field(default_factory=list)
    impedances: dict[str, dict[str, list[complex]]] = field(default_factory=dict)
    cluster_impedance_table: dict[str, dict[str, float]] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    input_deviations: dict[str, float] = field(default_factory=dict)
    is_consistent: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a line-oriented textual summary with the tolerance table."""
        labels = list(self.results.keys())
        lines = [
            f"Cross-engine comparison, tolerance {self.rel_tolerance * 100:.1f} %",
            f"Engines: {', '.join(labels)}",
            f"Frequencies: {', '.join(f'{f:g} Hz' for f in self.frequencies)}",
            "",
            "Quantity             f [Hz]  part  "
            + "  ".join(f"{lbl:>14s}" for lbl in labels)
            + "   dev [%]  ok",
        ]
        for row in self.rows:
            vals = "  ".join(
                f"{row.values.get(lbl, float('nan')):14.6g}" for lbl in labels
            )
            lines.append(
                f"{row.quantity:20s} {row.frequency:7g}  {row.component:4s}  "
                f"{vals}  {row.deviation * 100:8.3f}  "
                f"{'yes' if row.passed else 'NO'}"
            )
        lines.append("")
        lines.append(
            "Consistent." if self.is_consistent else "INCONSISTENT — see notes."
        )
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def _component_rows(
    quantity: str,
    frequency: float,
    per_engine: dict[str, complex],
    *,
    rel_tolerance: float,
    component_floor: float,
) -> list[ComparisonRow] | None:
    """Rows for the real and imaginary part of one complex quantity.

    Returns ``None`` when the quantity is undefined for every engine
    (all values non-finite) or when every magnitude is zero, i.e. when
    there is nothing to compare.
    """
    labels = list(per_engine)
    z = np.array([per_engine[lbl] for lbl in labels], dtype=complex)
    finite = np.isfinite(z)
    if not finite.any():
        return None
    if not finite.all():
        return [
            ComparisonRow(
                quantity=quantity,
                frequency=frequency,
                component=name,
                values={lbl: float(part(v)) for lbl, v in zip(labels, z, strict=True)},
                reference=float("nan"),
                deviation=float("inf"),
                tolerance=rel_tolerance,
                passed=False,
            )
            for name, part in _COMPONENTS
        ]
    scale = float(np.min(np.abs(z)))
    if scale == 0.0:
        return None
    rows = []
    for name, part in _COMPONENTS:
        c = part(z)
        reference = max(float(np.min(np.abs(c))), component_floor * scale)
        deviation = float(c.max() - c.min()) / reference
        rows.append(
            ComparisonRow(
                quantity=quantity,
                frequency=frequency,
                component=name,
                values={lbl: float(v) for lbl, v in zip(labels, c, strict=True)},
                reference=reference,
                deviation=deviation,
                tolerance=rel_tolerance,
                passed=deviation <= rel_tolerance,
            )
        )
    return rows


def _safe(method, arg: str) -> list[complex] | None:
    """Evaluate ``method(arg)`` and map lookup failures to ``None``."""
    try:
        return [complex(v) for v in method(arg)]
    except (KeyError, IndexError, ValueError):
        return None


def compare_engines(
    world: "World",
    engines: dict[str, "Engine"],
    *,
    rel_tolerance: float = 0.05,
    sample_points: np.ndarray | None = None,
    component_floor: float = DEFAULT_COMPONENT_FLOOR,
) -> EngineComparison:
    """Run ``world`` through every engine and compare the results.

    Parameters
    ----------
    world
        World to evaluate. **Not modified** by this function.
    engines
        Mapping ``label -> Engine``. Must contain at least two entries,
        all with the same frequency list.
    rel_tolerance
        Maximum allowed **pairwise** relative deviation (default 5 %)
        of every row of the tolerance table, i.e. per compared
        quantity, frequency and component (real / imaginary part):

        .. math::

            \\frac{\\max_k c_k - \\min_k c_k}
                  {\\max(\\min_k |c_k|,\\;
                         \\eta\\, \\min_k |Z_k|)}
            \\;\\le\\; \\texttt{rel_tolerance}

        over the engines :math:`k`, where :math:`c_k` is the real or
        imaginary part of the complex value :math:`Z_k` and
        :math:`\\eta` is ``component_floor``. Two engines differing by
        5 % of the smaller value sit exactly on the threshold.
    sample_points
        Optional. Array of shape ``(M, 3)`` in m: additional field
        points whose complex potentials are compared at every frequency
        (same metric, per point; the worst point is reported).
    component_floor
        Dimensionless floor :math:`\\eta` of the component reference,
        as a fraction of the smallest compared magnitude (default
        1e-3). A real or imaginary part that is smaller than
        :math:`\\eta |Z|` for every engine is compared on that absolute
        scale instead of relative to itself, so a reactance that is
        numerically zero (galvanic worlds) or negligible against the
        resistance does not turn round-off into a large relative
        deviation.

    Returns
    -------
    EngineComparison
        Structured report with the explicit tolerance table
        (:attr:`EngineComparison.rows`).

    Raises
    ------
    ValueError
        If fewer than two engines are given, or if the engines (or
        their results) do not share one frequency list.

    Notes
    -----
    **What is compared (changed in review pass 10, 10.71).** The
    cluster impedance of every cluster and the input impedance of every
    current source, as complex numbers at **every** frequency, real and
    imaginary part separately; optionally the complex potential at
    ``sample_points``. Up to 0.15.0 only :math:`\\mathrm{Re}\\,Z` of the
    clusters at the first frequency was compared, so a reactance that
    differed by a factor two at 1 kHz passed. A quantity that is
    undefined for *every* engine (a purely passive cluster with
    :math:`\\sum I = 0`) is skipped with a note; a quantity that is
    non-finite for *some* engines only is a failure (up to 0.15.0 the
    non-finite engine was dropped and the rest compared). Engines with
    different frequency lists raise instead of being compared at
    different frequencies.

    **Metric definition (changed in 0.15.0).** The deviation is the
    full spread of the compared values normalised by the *smallest*
    magnitude among them, which is the worst pairwise relative deviation
    ``max_ij |c_i - c_j| / min |c|`` for same-sign values. Up to 0.14.x
    the deviation was measured against the ensemble *mean*, which for
    two engines reports half the true disagreement.

    **Vacuous comparisons (changed in 0.15.0).** A report is
    ``is_consistent=True`` only if at least one row was evaluated. A
    world without sources (or with a misspelled ``Source.attached_to``)
    has no defined impedance anywhere; such a run yields
    ``is_consistent=False`` plus a "comparison vacuous" note.
    """
    if len(engines) < 2:
        raise ValueError("compare_engines requires at least 2 engines.")

    freq_lists = {
        label: [float(f) for f in getattr(eng, "frequencies", [])]
        for label, eng in engines.items()
    }
    if len({tuple(f) for f in freq_lists.values()}) > 1:
        detail = "; ".join(f"{lbl}: {f}" for lbl, f in freq_lists.items())
        raise ValueError(
            "compare_engines needs one frequency list for all engines, got "
            f"{detail}. Compare at common frequencies (Engine.frequencies)."
        )

    results: dict[str, FieldResult] = {
        label: engine.solve(world) for label, engine in engines.items()
    }
    freqs = [float(f) for f in next(iter(results.values())).frequencies]
    for label, res in results.items():
        if [float(f) for f in res.frequencies] != freqs:
            raise ValueError(
                f"Engine '{label}' returned the frequencies {res.frequencies}, "
                f"not {freqs}."
            )

    cmp = EngineComparison(
        results=results, rel_tolerance=rel_tolerance, frequencies=freqs
    )

    # Detect stub backends
    for label, res in results.items():
        if res.metadata.get("stub"):
            cmp.notes.append(
                f"Engine '{label}' returned a stub result "
                f"(metadata['stub']=True). Comparison not meaningful."
            )

    # Cluster sets must be identical
    cluster_keys_first = {
        tuple(sorted(v)) for v in next(iter(results.values())).clusters.values()
    }
    for label, res in results.items():
        keys = {tuple(sorted(v)) for v in res.clusters.values()}
        if keys != cluster_keys_first:
            cmp.notes.append(
                f"Engine '{label}' reports a different cluster structure. "
                "Cluster comparison skipped."
            )
            cmp.is_consistent = False
            return cmp

    # Quantities: cluster impedances (one representative per cluster)
    # and the input impedance of every current source.
    roots = sorted(
        {sorted(m)[0] for res in results.values() for m in res.clusters.values()}
    )
    quantities: list[tuple[str, str, dict[str, list[complex] | None]]] = []
    for root in roots:
        quantities.append(
            (
                f"Z_cluster[{root}]",
                root,
                {
                    lbl: _safe(res.cluster_impedance, root)
                    for lbl, res in results.items()
                },
            )
        )
    source_names = [
        s.name
        for s in next(iter(results.values())).sources
        if getattr(s, "kind", None) == "current"
    ]
    for name in source_names:
        quantities.append(
            (
                f"Z_in[{name}]",
                name,
                {lbl: _safe(res.input_impedance, name) for lbl, res in results.items()},
            )
        )

    n_compared = 0
    for quantity, key, per_engine in quantities:
        n_freq = len(freqs)
        values = {
            lbl: (
                v if v is not None and len(v) == n_freq else [complex("nan")] * n_freq
            )
            for lbl, v in per_engine.items()
        }
        cmp.impedances[quantity] = values
        worst = None
        for k, f in enumerate(freqs):
            rows = _component_rows(
                quantity,
                f,
                {lbl: v[k] for lbl, v in values.items()},
                rel_tolerance=rel_tolerance,
                component_floor=component_floor,
            )
            if rows is None:
                continue
            cmp.rows.extend(rows)
            n_compared += len(rows)
            dev = max(r.deviation for r in rows)
            worst = dev if worst is None else max(worst, dev)
            if not np.isfinite(dev):
                bad = sorted(lbl for lbl, v in values.items() if not np.isfinite(v[k]))
                cmp.notes.append(
                    f"{quantity} at {f:g} Hz: non-finite for engine(s) "
                    f"{', '.join(bad)} while finite for the others — counted "
                    "as a failure."
                )
        if worst is None:
            cmp.notes.append(
                f"{quantity}: undefined for every engine (Σ I = 0 or Z = 0) — skipped."
            )
            continue
        if quantity.startswith("Z_cluster["):
            cmp.deviations[key] = worst
            re0 = {
                lbl: float(v[0].real)
                for lbl, v in values.items()
                if np.isfinite(v[0].real)
            }
            if re0:
                cmp.cluster_impedance_table[key] = re0
        else:
            cmp.input_deviations[key] = worst

    # Optional point-sample for the potential, complex, every frequency.
    if sample_points is not None:
        try:
            pts = np.asarray(sample_points, dtype=float)
            n_pts = len(pts)
            sample_max = 0.0
            n_undefined = 0
            for k, f in enumerate(freqs):
                phis = np.stack(
                    [
                        np.asarray(res.potential(pts, frequency_index=k), dtype=complex)
                        for res in results.values()
                    ]
                )
                scale = np.min(np.abs(phis), axis=0)
                usable = np.isfinite(phis).all(axis=0) & (scale > 0.0)
                n_undefined = max(n_undefined, int(np.count_nonzero(~usable)))
                if not usable.any():
                    continue
                for name, part in _COMPONENTS:
                    c = part(phis)[:, usable]
                    ref = np.maximum(
                        np.min(np.abs(c), axis=0), component_floor * scale[usable]
                    )
                    dev = (c.max(axis=0) - c.min(axis=0)) / ref
                    j = int(np.argmax(dev))
                    point = int(np.nonzero(usable)[0][j])
                    row = ComparisonRow(
                        quantity=f"phi[{point}]",
                        frequency=f,
                        component=name,
                        values={
                            lbl: float(v)
                            for lbl, v in zip(results, c[:, j], strict=True)
                        },
                        reference=float(ref[j]),
                        deviation=float(dev[j]),
                        tolerance=rel_tolerance,
                        passed=bool(dev[j] <= rel_tolerance),
                    )
                    cmp.rows.append(row)
                    n_compared += int(np.count_nonzero(usable))
                    sample_max = max(sample_max, row.deviation)
            note = (
                f"Potential point-sample at {n_pts} points and {len(freqs)} "
                f"frequencies: max relative deviation = {sample_max * 100:.2f} % "
                "(max pairwise, referenced to min |phi|, Re and Im)."
            )
            if n_undefined:
                note += (
                    f" {n_undefined} point(s) excluded: phi = 0 or non-finite "
                    "for at least one engine, relative deviation undefined."
                )
            cmp.notes.append(note)
        except (RuntimeError, NotImplementedError) as e:
            cmp.notes.append(f"Potential sample not evaluated: {e}")

    failed = [r for r in cmp.rows if not r.passed]
    if failed:
        worst_row = max(failed, key=lambda r: r.deviation)
        cmp.notes.append(
            f"{len(failed)} of {len(cmp.rows)} rows exceed the tolerance; worst: "
            f"{worst_row.quantity} {worst_row.component} at "
            f"{worst_row.frequency:g} Hz, deviation "
            f"{worst_row.deviation * 100:.3g} % > {rel_tolerance * 100:.3g} %."
        )

    # A comparison that evaluated *nothing* has validated nothing.
    if n_compared == 0:
        cmp.notes.append(
            "No comparable quantity (every impedance undefined, no usable "
            "sample point) — comparison vacuous, nothing was validated."
        )
    cmp.is_consistent = (
        n_compared > 0
        and all(r.passed for r in cmp.rows)
        and not any(n.startswith("Engine '") and "stub result" in n for n in cmp.notes)
    )
    return cmp
