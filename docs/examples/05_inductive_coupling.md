# 05 — Analysing inductive coupling

Below 1 kHz the static / quasi-static formulation of `groundfield`
correctly captures the resistive part of every grounding installation,
but the longitudinal-branch impedance of a *cable* picks up a
significant $j\omega L$ contribution. `groundfield` supports two
inductance corrections (ADR-0004 / ADR-0005 / ADR-0006):

| `inductance_model` | Kernel                                | Suitable for                          |
|--------------------|---------------------------------------|---------------------------------------|
| `None` (default)   | DC, purely resistive                  | Quick parameter sweeps                |
| `"neumann"`        | Neumann self / mutual integral        | Cable + above-ground / shallow lines  |

The earth-return contribution (Carson 1926 and the rigorous
Sommerfeld correction) lives on the `Engine` instead of on the
conductor; it is controlled via the `earth_inductive_model`
attribute.

## Setup

Two parallel buried cables, 5 m apart, 30 m long, both at 0.6 m
depth. Cable A carries a 1 A test current (return through remote
earth via a 1.5 m driven rod at one end); cable B is a passive
observer whose induced voltage drop we want to read out across a
frequency sweep.

```python
import groundfield as gf

def build_two_parallel_cables(omega_frequencies):
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

    # Cable A: driven at x = 0, returns at x = 30.
    gf.create_electrode(world, "rod", name="A_start",
                        position=(0.0, 0.0, 0.5), length=1.5,
                        wire_radius=0.01)
    gf.create_electrode(world, "rod", name="A_end",
                        position=(30.0, 0.0, 0.5), length=1.5,
                        wire_radius=0.01)
    gf.create_conductor(
        world, name="cable_A",
        start="A_start", end="A_end",
        conductor_type="pen",
        wire_radius=0.005,
        cross_section="from_radius",
        coupling_to_soil="isolated",
        discretize_segment_length=1.0,
        inductance_model="neumann",      # activate Neumann kernel
    )

    # Cable B: parallel observer.
    gf.create_electrode(world, "rod", name="B_start",
                        position=(0.0, 5.0, 0.5), length=1.5,
                        wire_radius=0.01)
    gf.create_electrode(world, "rod", name="B_end",
                        position=(30.0, 5.0, 0.5), length=1.5,
                        wire_radius=0.01)
    gf.create_conductor(
        world, name="cable_B",
        start="B_start", end="B_end",
        conductor_type="pen",
        wire_radius=0.005,
        cross_section="from_radius",
        coupling_to_soil="isolated",
        discretize_segment_length=1.0,
        inductance_model="neumann",
    )

    # Source on cable A only.
    gf.create_source(world, attached_to="A_start", magnitude=1.0,
                     return_to="A_end")
    return world

frequencies = [50.0, 150.0, 250.0, 500.0, 1000.0]
world = build_two_parallel_cables(frequencies)
engine = gf.create_engine(
    backend="image",
    segment_length=0.5,
    frequencies=frequencies,
    earth_inductive_model="carson_series",   # Carson earth-return
)
result = engine.solve(world)

# Read the induced potential drop on cable B at every frequency.
phi_b_start = result.electrode_potentials["B_start"]
phi_b_end = result.electrode_potentials["B_end"]
for f, ps, pe in zip(frequencies, phi_b_start, phi_b_end):
    induced = pe - ps
    print(f"f = {f:6.0f} Hz   |U_B| = {abs(induced):8.4f} V   "
          f"phase = {induced and __import__('cmath').phase(induced):+6.2f} rad")
```

## Earth-return models

The Carson 1926 series accounts for the current return through the
soil and dominates the inductive coupling at typical LV / MV
frequencies. The rigorous Sommerfeld kernel
(`earth_inductive_model="sommerfeld"`) is the absolute reference
for layered soil; for homogeneous soil and frequencies well below
1 kHz the cheaper Carson series and the perfect-mirror baseline
agree closely.

## What to look at

- The induced cable-B end-to-end potential rises linearly with
  $\omega$ — the magnitude doubles when frequency doubles.
- The phase is close to $+\pi/2$ relative to the source current,
  consistent with $j\omega M\,I_A$ inductive coupling through the
  mutual inductance $M$.
- Switching `earth_inductive_model` from `"perfect_mirror"` to
  `"carson_series"` reduces the induced voltage by a factor that
  depends on $\delta = 1/\sqrt{\pi f \mu_0 \sigma}$ (the Carson
  penetration depth). For 100 Ω·m at 50 Hz that's $\delta \approx
  1.4\;\mathrm{km}$ — much larger than the cable separation, so
  the correction is moderate.

## Where to go next

- Read the **engine theory** pages for the full inductance
  kernels: `docs/engines/mom_sommerfeld.md` for the rigorous
  reference, `docs/engines/image.md` and `docs/engines/image_2layer.md`
  for the closed-form layered backends. The ADRs ADR-0004
  (inductive coupling), ADR-0005 (Carson) and ADR-0006
  (Sommerfeld) document the physics in depth.
- Visualise the magnetic-coupling pattern via the postprocess
  helpers — see [09 — All plots in action](09_plot_gallery.md).
