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
| E01 | [Secondary substation and its low-voltage network](E01_secondary_substation_and_lv_network.ipynb) | How much does the low-voltage network lower the earthing impedance of a 10/0.4 kV station, which earth potential rise and touch voltages does a fault cause, and which potentials reach the houses? |
| E02 | [Single-family house](E02_single_family_house.ipynb) | What earthing resistance and touch voltage does the foundation earth electrode of a detached house give, and what do concrete moisture, the soil model, a rod or an outer ring change? |
| E03 | [HV substation grid in two-layer soil](E03_hv_substation_grid_two_layer_soil.ipynb) | Does the mesh earthing grid of a high-voltage substation meet the permissible touch voltage of EN 50522 in two-layer soil, and what makes it pass? |

## E01 — Secondary substation and its low-voltage network

A 10/0.4 kV secondary substation earth (a ring with four rods) connected
through the PEN to the foundation earth electrodes of 30 houses and two
cable cabinets, in two-layer soil (200 over 30 Ω·m). The case study
quantifies the global earthing effect, with numbers for 1 to 80 houses and
five soils: 9.9 Ω for the station alone, 0.29 Ω with the network. It covers:

- the earth potential rise and the touch voltages for a resonant-earthed
  and a low-impedance-earthed fault, against EN 50522;
- the potentials the PEN transfers to the houses, including a house
  without an earth electrode;
- what changes at 1 kHz.

It also asks how far the numbers can be trusted: a discretisation ladder,
`image` against `mom`, `mom` against `mom_sommerfeld`, and what `fem` can
and cannot do here. It runs in about one minute.

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

## E03 — HV substation grid in two-layer soil

A 60 m × 40 m mesh grid with corner rods carries an earth fault current of
2 kA. It sits in a resistive top layer over a conductive bottom, and in the
reverse. The notebook:

- checks the segment length;
- compares the grid impedance and the earth potential rise with the
  homogeneous limits, IEEE 80 and Dwight;
- maps the touch voltage and evaluates the step voltages;
- assesses the design against the EN 50522 permissible touch voltage for
  0.5 s, and shows the measures that make it pass in each soil: a grading
  conductor, perimeter rods, shoes or crushed rock.

It explains why the touch voltages near a grid come from `mom`, not from
`image_2layer`. It also shows how the layer contrast moves the impedance and
the touch voltage in opposite directions. It runs in about a minute.
