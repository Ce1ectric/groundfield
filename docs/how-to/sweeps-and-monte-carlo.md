# Run sweeps and Monte Carlo studies

This guide computes the impedance of many variants of one world — a grid of
parameter values, or many realisations of a stochastic network — with one
call of `gf.sweep`, and spreads the work over several processes. Read it
when a study needs more than a handful of solves.

`gf.sweep` builds one world per combination of parameter values, solves
it, and returns a long-format pandas table with one row per combination
and frequency. The same function runs a Monte Carlo study when one of the
axes is the seed of a stochastic generator.

**Prerequisites:** the basic workflow (soil → world → electrodes → source →
engine → `solve`) of tutorial
[T01](../tutorials/T01_first_grounding_impedance.ipynb); pandas is part of
the core install.

## 1. Sweep a parameter grid

Write a *world factory* — a function that takes the swept parameters as
keyword arguments and returns a fresh world — and pass the values per
axis. The default response is the input impedance of the world's current
source (`quantity="input"`; `"cluster"` and `"electrode"` select the other
impedances, see tutorial [T03](../tutorials/T03_interconnected_systems_and_impedances.ipynb)).

```python
import groundfield as gf


def rod_world(rho, length):
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=rho))
    gf.create_electrode(world, "rod", name="rod", position=(0.0, 0.0, 0.0), length=length, wire_radius=0.008)
    gf.create_source(world, attached_to="rod", magnitude=1.0)
    return world


engine = gf.create_engine(segment_length=0.5, frequencies=[50.0])
df = gf.sweep(rod_world, engine, axes={"rho": [50.0, 100.0, 300.0], "length": [1.5, 3.0, 6.0]})
print(df.pivot(index="rho", columns="length", values="abs_Z").round(3))
```

<!-- doctest-check
import numpy as np
table = df.pivot(index="rho", columns="length", values="abs_Z")
# Homogeneous soil: the impedance is proportional to rho for every rod length.
assert np.allclose(table.div(table.index.to_series(), axis=0).std(axis=0), 0.0, atol=1e-12)
assert len(df) == 9 and df.attrs["impedance_quantity"] == "input"
-->

**What you get:** one row per combination and frequency. The columns are
the axis names, `frequency_Hz` and the response (`Z_re`, `Z_im`, `abs_Z`,
`arg_Z_deg`, `U_E_*`, `I_*`). In homogeneous soil the impedance of every
rod is proportional to $\rho$, which the hidden check of this page
asserts. For a quick look:

```python
fig = gf.plot_sweep_lines(df, x="rho", color="length", log_x=True)
```

## 2. Let the discretisation follow the parameters

When the discretisation should depend on the parameters, pass a function
that returns the engine for a combination instead of an engine:

```python
df_ds = gf.sweep(
    rod_world,
    lambda rho, length: gf.create_engine(segment_length=length / 10.0),
    axes={"rho": [100.0], "length": [1.5, 6.0]},
)
print(df_ds[["length", "abs_Z"]])
```

<!-- doctest-check
assert list(df_ds["length"]) == [1.5, 6.0]
-->

## 3. Monte Carlo over a stochastic network

A `TnNetworkGenerator` draws a whole realisation from its seed, so the
seed is a natural sweep axis. Put the factory into a module (a `.py` file
next to your script or notebook) — that is also what parallel runs need:

<!-- doctest-module: mc_factories.py -->
```python
# mc_factories.py
from groundfield.generators import (
    ManhattanGridPlacement,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    Uniform,
)

CFG = TnNetworkConfig(
    soil=TwoLayerSoilSpec(rho_1=Uniform(low=50.0, high=500.0), rho_2=100.0, h_1=Uniform(low=5.0, high=15.0)),
    building_counts={"residential": 4},
    placement=ManhattanGridPlacement(spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=2),
)


def network(seed):
    return TnNetworkGenerator(CFG, seed=seed).build()
```

```python
import groundfield as gf
from mc_factories import network

engine = gf.create_engine(segment_length=1.0, frequencies=[50.0])
mc = gf.sweep(network, engine, axes={"seed": range(16)})
print(mc["abs_Z"].describe().round(3))
print("5 % / 95 % quantiles:", mc["abs_Z"].quantile([0.05, 0.95]).round(3).tolist(), "Ohm")
```

<!-- doctest-check
import numpy as np
assert len(mc) == 16 and mc["abs_Z"].nunique() == 16
# Any realisation can be repeated from its seed alone.
repeat = engine.solve(network(7)).input_impedance()[0]
assert abs(repeat - mc.loc[mc["seed"] == 7, "abs_Z"].item()) < 1e-12 * abs(repeat)
-->

**What you get:** the distribution of the impedance over 16
realisations, here as `describe()` and the 5 % / 95 % quantiles. Record
the seed range with the results: any single realisation can be rebuilt
from its seed (tutorial [T10](../tutorials/T10_reproducible_studies.ipynb)),
and the hidden check repeats realisation 7 to 1e-12.

## 4. Run in parallel

`n_jobs` spreads the combinations over worker processes (`-1`: one per
CPU). The rows come back in the order of the serial run. `sweep` pins the
workers to a single-threaded BLAS; the values are bit-identical to the
serial run when the calling process runs a single-threaded BLAS too
(`OPENBLAS_NUM_THREADS=1`), otherwise they agree to round-off. The factory, the engine and a
custom response must be importable by the workers: module-level
functions, not lambdas or functions defined in a notebook — `sweep`
raises a clear `TypeError` otherwise.

```python
import pandas as pd

parallel = gf.sweep(network, engine, axes={"seed": range(16)}, n_jobs=2)
pd.testing.assert_frame_equal(parallel, mc, check_exact=False, rtol=1e-12, atol=0.0)
print("serial and parallel runs agree:", len(parallel), "rows")
```

**How to check it:** `pd.testing.assert_frame_equal` above compares the
parallel table with the serial one to 1e-12 (bit for bit only with a
single-threaded BLAS in the calling process: on a four-core CI runner one
of the 16 values differed in the last digit). Starting the workers costs
about a second each, so parallel runs pay off for sweeps whose single
solves take longer than that. Warnings raised in the workers are re-issued
in the calling process.

## Summary

- `gf.sweep(factory, engine, axes={...})` → one row per combination and
  frequency; `quantity=` selects the impedance.
- A seed axis turns a stochastic generator into a Monte Carlo study.
- `n_jobs=` parallelises; keep the factory in an importable module.

## See also

- Tutorial [T03](../tutorials/T03_interconnected_systems_and_impedances.ipynb):
  the input, cluster and electrode impedances that `quantity=` selects.
- Tutorial [T06](../tutorials/T06_tn_network_synthetic_and_osm.ipynb):
  the TN network generator used for the Monte Carlo study.
- [Plot results](plot-results.md): `plot_sweep_lines` and
  `plot_sweep_heatmap`.
- [Speed up large studies](speed-up-large-studies.md): what to change
  before you parallelise.
- API: [`sweep`](../reference/api/postprocess.md) and the
  [generators](../reference/api/generators.md).
