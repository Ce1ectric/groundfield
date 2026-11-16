# Examples

These case studies take the tools of the [tutorials](../tutorials/index.md)
to complete engineering questions. Each one models a real type of
installation and answers what a grounding engineer has to answer for it:
the earthing impedance, the earth potential rise for a fault, and the touch
and step voltages against the permissible values of EN 50522. Each also
asks which engine to trust and why, and ends with the limits of the example.

Each example is a Jupyter notebook. The page shows the stored outputs of a
complete run, and you can download the notebook and run it yourself. As in
the tutorials, the key numbers are checked in `assert` cells against a
second method: another engine, a convergence study or a closed-form
estimate. If you have not used groundfield before, start with the
[quickstart](../getting-started/quickstart.md). For the choice of engine,
see [choosing an engine](../explanation/engines/index.md).

| | Example | The question |
|---|---|---|
| E02 | [Single-family house](E02_single_family_house.ipynb) | What earthing resistance and touch voltage does the foundation earth electrode of a detached house give, and what do concrete moisture, the soil model, a rod or an outer ring change? |

## E02 — Single-family house

The earthing of a detached house with a foundation earth electrode in
concrete, alone, with a driven rod, or with a ring earth electrode 1 m
outside, in homogeneous and two-layer soil. It shows:

- how concrete moisture and the soil model move the earthing resistance;
- when a rod pays off (only once it reaches a more conductive layer);
- why an outer ring roughly halves the touch voltage and keeps working when
  the concrete dries out.

It turns the permissible touch voltage of EN 50522 into an admissible
earth potential rise. It compares `image`, `mom` and `fem` and explains
where they differ. It runs in about half a minute.
