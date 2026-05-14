# Examples

A guided tour through `groundfield`. Each example is a complete,
self-contained Python script — copy it into a new file or a
Jupyter notebook and run it. Examples build on each other but
every page also recaps what you need to follow it on its own.

If you're brand new to the project, work through the examples in
order. Each one introduces a couple of new concepts and shows the
output you should see.

## Pre-requisites

```bash
pip install groundfield
# optional: closes the loop with the sister project groundinsight
pip install "groundfield[groundinsight]"
```

For Monte-Carlo sweeps you'll also want `joblib` and `pandas`,
both standard scientific-Python tools. They come with most
distributions; otherwise `pip install joblib pandas pyarrow`.

## The eight examples

| # | Topic | What you learn |
|---|---|---|
| [01](01_first_solve.md) | First solve: a single rod | Build a `World` by hand, run `engine.solve(world)`, plot the radial potential |
| [02](02_substation_dwight.md) | Substation grounding vs. Dwight 1936 | Cluster impedance, comparison against closed-form references |
| [03](03_tn_network_basics.md) | TN-Ortsnetz generator basics | `TnNetworkConfig`, `TnNetworkGenerator`, surface-potential plot |
| [04](04_grounding_measurement.md) | Galvanic fall-of-potential measurement | Auxiliary electrode + voltage probe, fall-of-potential characteristic |
| [05](05_inductive_coupling.md) | Inductive coupling on the measurement leads | Overhead leads with Neumann coupling, Carson earth-return, measurement-error quantification |
| [06](06_parameter_sweep_soil.md) | Parameter sweep over soil resistivity | Deterministic sweep over $\rho_1$, log-log $\|Z(\rho_1)\|$ trend |
| [07](07_monte_carlo.md) | Monte-Carlo sweep with `joblib` | Stochastic distributions, parallel workers, persistent results, statistical bands |
| [08](08_pipeline_groundinsight.md) | Full pipeline to `groundinsight` | $\rho$-$f$ fit, JSON `BusType` export, downstream fault analysis |

## How to use this catalogue

* If you only have 30 minutes, run examples 01 → 03 → 06. Those
  cover "build a world", "build a network from a config", and
  "sweep over a parameter axis" — most engineering use cases
  are mash-ups of those three.
* If your goal is the measurement-error question, jump to
  04 → 05.
* If you are about to launch a serious parameter study, read
  example 07 *and* the [performance guide](../performance.md)
  before you commit a 12-hour run.
* If you want to feed reduced models into `groundinsight` for
  fault calculations, the full chain is in example 08; it builds
  on the `rho-f` fitting machinery covered there.

## Related material

* [Concepts](../concepts.md) — the theoretical scaffold (PDE
  formulation, soil models, electrode types).
* [Engine theory](../engines/index.md) — what each backend does
  numerically and when to pick it.
* [Performance guide](../performance.md) — empirical timings
  and Monte-Carlo guidance.
* [ADR-0009 — World generators](../adr/0009-world-generators.md) —
  the design behind `TnNetworkGenerator` and the spec layer.
