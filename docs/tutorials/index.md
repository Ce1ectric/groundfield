# Tutorials

Ten tutorials take you from a single earth rod to reproducible studies of
whole low-voltage networks. Each one is a Jupyter notebook: the page shows
the stored outputs of a complete run, and you can download the notebook
and run it yourself. Every tutorial explains what is computed and why,
gives the physics in a few sentences, and checks its key numbers against a
second method — a closed-form formula, a limit, another engine or a
convergence study — in `assert` cells.

| | Tutorial | You learn |
|---|---|---|
| T01 | [Your first grounding impedance](T01_first_grounding_impedance.ipynb) | soil, world, electrode, source, engine; reading a result; checks against Dwight (1936) and the analytic surface potential |
| T02 | [Layered soil](T02_layered_soil.ipynb) | two- and three-layer soil, the homogeneous limits, the layered Green's function as series and as Sommerfeld integral, rods through the interface |
| T03 | [Interconnected systems and the three impedances](T03_interconnected_systems_and_impedances.ipynb) | ideal bonds and PEN branches; input, cluster and electrode impedance; split factor; the mutual impedance matrix $Z_G$ |
| T04 | [Frequency response and the rho-f model](T04_frequency_response_and_rho_f.ipynb) | inductive conductors and earth-return models; the standard rho-f form and vector fitting, validated on unseen data |
| T05 | [Export to groundinsight](T05_export_to_groundinsight.ipynb) | BusType documents, round-trip accuracy, `rho_at_fit`, passivity, evaluation inside groundinsight |
| T06 | [TN networks, synthetic and from OpenStreetMap](T06_tn_network_synthetic_and_osm.ipynb) | `TnNetworkGenerator`, `LvNetworkLayout` from cached Overpass data, penetration of foundation earth electrodes |
| T07 | [The fall-of-potential measurement](T07_fall_of_potential_measurement.ipynb) | two-terminal sources, the 61.8 % rule, measurement leads on junction nodes, lead coupling |
| T08 | [Touch and step voltages](T08_touch_and_step_assessment.ipynb) | touch-voltage envelope, image vs `mom` near grids, EN 50522 limits, harmonic fault currents |
| T09 | [Choosing an engine and checking convergence](T09_choosing_an_engine_and_convergence.ipynb) | the backend family, `compare_engines`, line vs point kernel, convergence against an exact reference |
| T10 | [Reproducible studies](T10_reproducible_studies.ipynb) | seeds, network plans, provenance, result files, re-runs and reference checks |

## Where to start

- **New to groundfield:** T01 → T02 → T03. They cover the workflow, the
  soil models and the impedances of connected systems.
- **Reduced models for network studies:** T03 → T04 → T05.
- **Studies on real LV networks:** T03 → T06 → T07, then T10 before a
  large parameter study.
- **Safety assessment:** T01 → T03 → T08.
- **Validation and method questions:** T02 and T09, and the
  [explanation](../explanation/index.md) pages of the engines.

## Running the notebooks yourself

Install groundfield with the extras the tutorials use and Jupyter:

```bash
pip install "groundfield[geo,groundinsight]" jupyterlab
```

Download a notebook from its page (or take it from `docs/tutorials/` in a
checkout) and open it from its own directory: T06 reads its OpenStreetMap
data from `data/osm/` next to the notebooks. No tutorial needs network
access; each runs in well under two minutes on a laptop. The numbers in
your run may differ from the stored ones in the last digits on another
platform; the `assert` cells use tolerances that allow for that.

The test suite executes every tutorial (`tests/test_tutorials.py`, marker
`slow`), so the stored notebooks run with the version of groundfield they
are published with.

The terms used (Ortsnetz, PEN, foundation earth electrode, EPR, …) are
explained in the [glossary](../reference/glossary.md), units and sign
conventions in [units and conventions](../reference/units-and-conventions.md).
Task-oriented recipes are in the [how-to guides](../how-to/index.md).
