# Validation (cross-engine)

The :mod:`groundfield.validation` module contains
:func:`compare_engines`, the cross-engine consistency check that
underpins ADR-0001. Given two :class:`Engine` instances and a
:class:`World`, it solves both and reports a structured
:class:`EngineComparison` summarising the agreement of
cluster impedances, electrode currents and surface potentials.

This is the post-solve counterpart of :mod:`groundfield.diagnostics`
(pre-solve structural checks).

## Deviation metric (changed in 0.15.0)

For every cluster the report contains the **worst pairwise relative
deviation** of the cluster impedance $Z$ over the compared engines,

$$
\mathrm{dev}
= \max_{i,j} \frac{|Z_i - Z_j|}{\min_k |Z_k|}
= \frac{\max_k Z_k - \min_k Z_k}{\min_k |Z_k|} ,
$$

the second form holding whenever all $Z_k$ have the same sign (the
normal case: $\mathrm{Re}\{Z\} > 0$ for a passive grounding system).
`rel_tolerance` is compared against this number, so two engines that
differ by 5 % of the smaller value sit exactly on a
`rel_tolerance=0.05` threshold. The same metric is used for the
optional `sample_points` potential comparison, per sample point.

Up to 0.14.x the deviation was measured against the ensemble **mean**,
$\max_k |Z_k - \bar{Z}| / |\bar{Z}|$. For two engines that is exactly
half the true disagreement (and it shrinks further as engines are
added), so a `rel_tolerance=0.05` gate silently admitted a spread of
10.5 %. Consequences of the change:

- reported deviations are roughly a factor 2 larger than in 0.14.x for
  the same world and engines — no physics changed, only the metric;
- a comparison that was *marginally* consistent under 0.14.x may now
  legitimately be reported as inconsistent. Loosen `rel_tolerance` only
  after checking which engine is the outlier;
- clusters for which at least one engine reports $Z = 0$ have an
  undefined relative deviation and are now skipped with a note instead
  of being silently counted as agreeing.

## Vacuous comparisons (changed in 0.15.0)

A cluster whose net injected current vanishes ($\sum I = 0$: purely
passive observer electrodes) has no defined impedance and is skipped.
If **every** cluster is skipped and no `sample_points` potential has a
usable (non-zero) reference, nothing was validated — the typical causes
are a world without any source, or a `Source.attached_to` that does not
match any electrode or conductor name. Such a run now reports

```
is_consistent = False
notes: "No comparable quantity (every cluster impedance undefined, no
        usable sample point) — comparison vacuous, nothing was
        validated."
```

Up to 0.14.x the same situation returned `is_consistent=True` (the
running maximum deviation never left its initial `0.0`), so a CI job
asserting `report.is_consistent` passed green having compared nothing.

## Frequency-order behaviour

Since 0.5.0 :class:`Engine.frequencies` is order-preserving. A
non-monotonic frequency list raises a dedicated
:class:`EngineFrequencyOrderWarning` (subclass of
:class:`UserWarning`) so a single
``warnings.simplefilter("once", EngineFrequencyOrderWarning)`` will
collapse a ``compare_engines`` 4 × 4 matrix to one emission. The
opt-in :meth:`Engine.with_frequencies(*, preserve_order=True)`
constructor silences the warning explicitly.

## API reference

::: groundfield.validation
