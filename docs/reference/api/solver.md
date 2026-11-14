# Solver

The solver layer holds the numerics: the `Engine` (configuration and
`solve`), the `FieldResult` (potentials, currents, impedances,
metadata) and the multi-port functions `solve_mutual_matrix` /
`solve_mutual_field`. The backends themselves are registered in the
[backend registry](registry.md); their theory is on the
[engine pages](../../explanation/engines/index.md).

## Mathematical and physical context

Every backend solves the same problem — the quasi-static potential of
the electrodes in layered soil, coupled to the conductor network — for
each frequency of `Engine.frequencies`. The electrodes are cut into $N$
segments (length at most `segment_length`) carrying leakage currents
$I_k$; the potential of segment $i$ is

$$
\varphi_i = \sum_{k=1}^{N} G_{ik}\, I_k ,
$$

with $G_{ik}$ the potential at segment $i$ per unit current on segment
$k$ (the Green's function of the layered half-space integrated over the
segments). The galvanic clusters are equipotential, the conductor
branches add Kirchhoff's laws, and the source currents are the
right-hand side of the augmented system. The backends differ in how
they evaluate $G$ and which current ansatz they use:

| Backend | Green's function | Current ansatz |
|---|---|---|
| `image` | free space + surface image, closed form | uniform current per unit length on each straight wire of an electrode (one unknown per wire line of a grid, arm of a star, straight run of a polyline, and soil layer; user decision Q30); exact line-source kernel by default |
| `image_2layer` | Tagg/Sunde image series, closed form; layered Sommerfeld correction for wires through the interface | as `image` |
| `mom` | the same closed-form kernels | point matching: one unknown current per segment, midpoint point sources |
| `mom_sommerfeld` | numerical Sommerfeld quadrature of the layered Green's function, any number of layers | as `mom` |
| `fem` (experimental) | axisymmetric finite elements of the equivalent hemisphere | volume solution |

The uniform-current ansatz gives accurate impedances at low cost; near
grids, stars, strip ends and bonded multi-electrode clusters its
surface potentials are impedance-grade, not field-grade (see
[Known limitations](../known-limitations.md)).

`Engine(backend="image")`, the default, forwards a `TwoLayerSoil` to
`image_2layer` and a `MultiLayerSoil` with at most two distinct layers
to the image dispatcher; three or more layers need `mom_sommerfeld`.
`FieldResult.metadata["effective_backend"]` and `["kernel"]` record
what ran.

## Engine settings

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"image"` | registered backend name (see [registry](registry.md)) |
| `frequencies` | `[50.0]` | Hz; the given order is kept (a non-monotonic list warns with `EngineFrequencyOrderWarning`, `with_frequencies(..., preserve_order=True)` silences it) |
| `segment_length` | `0.5` | maximum segment length in m |
| `kernel` | `None` | `None`: the backend's preferred kernel (`"line"` for the image family); `"line"` or `"point"` explicitly; `"point"` reproduces the numbers of releases up to 0.15.0 |
| `earth_inductive_model` | `"sommerfeld"` | earth return of inductive conductors: `"sommerfeld"`, `"carson_series"`, `"perfect_mirror"` (see [Coupling](coupling.md)) |
| `sommerfeld_assembly` | `"tabulated"` | assembly of the Sommerfeld correction of inductive conductors: `"tabulated"` (one table of the spectral integral per pair of depth groups and frequency; default since 2026.11) or `"exact"` (one quadrature per segment pair, the default up to 0.15.0) |
| `max_segments` | `None` | resource limit of this engine: the largest number of leakage segments and of nodal unknowns it solves, `ResourceLimitError` above it before anything is allocated; `None` takes `groundfield.diagnostics.HARD_LIMIT` (20 000); recorded in `metadata["max_segments"]` |
| `warn_segments` | `None` | size of the soft notice of this engine, counted like `max_segments`: at or above it the solve warns that it may take minutes (an *info* finding of `diagnose`); `None` takes `groundfield.diagnostics.SOFT_LIMIT` (5 000); not checked against `max_segments` (a notice above the hard limit never appears) |
| `image_max_terms`, `image_series_tol` | 300, 10⁻⁶ | cap and stop tolerance of the Tagg/Sunde series (tail bound $\lvert K\rvert^{n+1}/(1-\lvert K\rvert)$) |
| `options` | `{}` | backend-specific settings, e.g. `{"mom_sommerfeld": {"epsrel": 1e-9}}`, `{"fem": {"n_radial": 120}}` |

`tolerance` and `max_iterations` have no effect and are deprecated
(see [Deprecations](../deprecations.md)).

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "rod", name="a", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_electrode(world, "ring", name="b", center=(20.0, 0.0, 0.8), radius=2.0)
engine = gf.create_engine(frequencies=[50.0])

# Mutual grounding-impedance matrix of the two earths in one assembly:
# Z[i, j] = potential of cluster i per ampere leaking from cluster j.
Z = gf.solve_mutual_matrix(world, engine, anchors=["a", "b"])   # Ω

# The same entries from an ordinary solve with 1 A at "a":
gf.create_source(world, attached_to="a", magnitude=1.0)
result = engine.solve(world)
assert abs(Z[0, 0] - result.cluster_impedance("a")[0]) < 1e-9    # 51.40 Ω
assert abs(Z[1, 0] - result.electrode_potentials["b"][0]) < 1e-9  # 2.26 Ω
assert result.metadata["effective_backend"] == "image_2layer"
```

`solve_mutual_matrix` returns the galerkin off-diagonal (the reaction of
the leakage distributions: reciprocal, and the matrix that reproduces a
solve of the clusters joined by branches) for every call since 2026.11.
`offdiag="probe"` with `probe_points` samples the potential at one point
per cluster instead, the definition up to 0.15.0; probe points passed
without it are ignored with a `ProbePointsIgnoredWarning`.

`FieldResult.potential(points, frequency_index=0)` evaluates the
potential in V at arbitrary points of shape `(M, 3)` (homogeneous and
two-layer soil), `to_dataframe()` returns the per-electrode table, and `metadata["provenance"]` records the package
and library versions, the engine configuration, a fingerprint of the
world and the solve time stamp. Results are saved and loaded with
[`save_result` / `load_result`](results-io.md).

## API reference

::: groundfield.solver.engine

::: groundfield.solver.result

::: groundfield.solver.mutual

::: groundfield.solver.image
    options:
      members:
        - ShallowSegmentWarning
