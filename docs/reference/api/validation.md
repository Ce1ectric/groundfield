# Validation (cross-engine)

`compare_engines(world, engines={...})` solves the same world with two
or more engines and reports, in one tolerance table, how far their
results are apart. Engines that differ in method (image series vs
point-matching MoM vs Sommerfeld quadrature), in kernel or only in the
discretisation make it the standard check that a number does not
depend on a numerical choice (ADR-0001: two engines side by side,
validating each other). The validation of the engines against
closed-form references and against each other is collected on the
[validation page](../../explanation/validation.md).

## What is compared

Every compared quantity is complex and is compared at **every**
frequency, real and imaginary part separately:

- the cluster impedance of every cluster (Ω),
- the input impedance of every current or voltage source (Ω; a voltage
  source since 2026.11),
- optionally the potential at `sample_points` (V).

Each (quantity, frequency, part) is one `ComparisonRow` with the
deviation

$$
\mathrm{dev} = \frac{\max_k c_k - \min_k c_k}{\max\bigl(\min_k \lvert c_k\rvert,\ \epsilon\,\min_k \lvert Z_k\rvert\bigr)},
$$

the worst pairwise spread of the part $c_k$ over the engines, relative
to the smallest value; the floor $\epsilon$ (`component_floor`,
default 10⁻³ of the smallest magnitude) keeps a part that is nearly zero
for every engine (the reactance at 50 Hz) from dominating. A row passes
when $\mathrm{dev} \le$ `rel_tolerance`. A value that is non-finite or
zero for some engines only fails its row.

`is_consistent` is `True` only if at least one quantity was compared,
every row passed and no engine returned a stub result: a comparison in
which nothing could be evaluated (no source, a source on an unknown
node) is reported as inconsistent rather than as a silent pass. The
engines must use the same frequency list.

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0))
gf.create_electrode(world, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=5.0)
gf.create_source(world, attached_to="g1", magnitude=1.0)

report = gf.compare_engines(
    world,
    engines={
        "image_2layer": gf.create_engine(segment_length=0.5),
        "mom": gf.create_engine(backend="mom", segment_length=0.5),
        "refined": gf.create_engine(segment_length=0.25),
    },
    rel_tolerance=0.02,
)
print(report.summary())

assert report.is_consistent
assert len(report.rows) == 4    # Z_cluster[g1] and Z_in[source_0], Re and Im, at 50 Hz
```

A single engine raises `ValueError`: one engine cannot validate itself.
The deprecated names `cim` and `bem` run `image` and `mom` and add no
independent information.

## API reference

::: groundfield.validation
