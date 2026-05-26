# Validation (cross-engine)

The :mod:`groundfield.validation` module contains
:func:`compare_engines`, the cross-engine consistency check that
underpins ADR-0001. Given two :class:`Engine` instances and a
:class:`World`, it solves both and reports a structured
:class:`EngineComparison` summarising the agreement of
cluster impedances, electrode currents and surface potentials.

This is the post-solve counterpart of :mod:`groundfield.diagnostics`
(pre-solve structural checks).

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
