# How-to guides

Short, task-oriented recipes for readers who know what they want to do.
Each guide assumes you know the basic workflow
(soil → world → electrodes → source → engine → `solve`) from the
[quickstart](../getting-started/quickstart.md) or the first
[tutorials](../tutorials/index.md). The code blocks are complete and are
executed by the test suite (`tests/test_doc_snippets.py`), so what you copy
is what CI ran.

| Guide | Task |
|---|---|
| [Install the optional extras](install-extras.md) | OpenStreetMap ingest, groundinsight bridge, compiled kernels |
| [Run sweeps and Monte Carlo studies](sweeps-and-monte-carlo.md) | parameter grids, stochastic networks, parallel workers |
| [Speed up large studies](speed-up-large-studies.md) | segment budget, kernel and earth-return choices, many excitations |
| [Plot results](plot-results.md) | geometry, surface potential, profiles, cross-sections, current sharing, sweeps |
| [Model the inductive coupling of cables and leads](inductive-coupling.md) | Neumann inductance, earth-return models, skin effect |
| [Assess touch voltages](assess-touch-voltages.md) | touch points at accessible metal parts, transferred potential, open-air grids, EN 50522 limits |
| [Export to CSV, VTK and ParaView](export-csv-vtk-paraview.md) | tables with provenance, geometry and field files |
| [Work with OpenStreetMap offline](osm-offline.md) | Overpass cache, building filters, coordinates |
| [Read and act on warnings](warnings-and-diagnostics.md) | `diagnose()`, the common warnings and what to do about them |
| [Contribute to the documentation](contribute-to-the-docs.md) | page types, tutorial notebooks, the snippet markers, local build |

Which engine to use for a task, and where each engine must not be used,
is on [choosing an engine](../explanation/engines/index.md); the theory
behind the engines and the validation numbers are in the
[explanation](../explanation/index.md) section; every public name is in the
[reference](../reference/index.md).
