# 04 — Interconnected grounding system with line connections

Real grounding installations are *clusters* of electrodes joined by
copper bonds, PEN cables and bare earthing strips. `groundfield`
models such systems via the :func:`create_conductor` API — every
conductor either fuses its two endpoint electrodes into one
galvanic cluster (ideal bond, `cross_section=None`) or enters the
nodal-analysis system as a finite-impedance branch
(`cross_section="from_radius"` or an explicit area in
$\mathrm{m}^2$).

## Setup

A small substation grounding: a 5 m ring at burial depth 0.6 m,
three driven rods inside the ring, every electrode tied together
with bare-copper bonds. A 30 m PEN service drop (insulated cable,
finite impedance) feeds a house foundation electrode (1 m × 1 m,
2 × 2 grid mesh) 25 m away.

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

# --- Substation grounding cluster -------------------------------------
gf.create_electrode(
    world, "ring", name="trafo_ring",
    center=(0.0, 0.0, 0.6), radius=5.0, wire_radius=0.005,
)
for k, theta in enumerate((0.0, 2.094, 4.189)):  # 0, 120, 240 deg
    gf.create_electrode(
        world, "rod", name=f"trafo_rod_{k}",
        position=(2.0 * (theta and 1.0) * float(__import__('math').cos(theta)),
                  2.0 * float(__import__('math').sin(theta)),
                  0.5),
        length=2.0, wire_radius=0.01,
    )
# Galvanic bonds (ideal short, cross_section=None).
for k in range(3):
    gf.create_conductor(
        world, name=f"trafo_bond_{k}",
        start="trafo_ring", end=f"trafo_rod_{k}",
        conductor_type="bare_copper",
    )

# --- House foundation electrode ---------------------------------------
gf.create_electrode(
    world, "grid_mesh", name="house_foundation",
    corner=(24.5, -0.5, 0.8),
    size=(1.0, 1.0), n_x=2, n_y=2,
    wire_radius=0.005,
)

# --- PEN service drop: substation -> house, finite-impedance branch ---
gf.create_conductor(
    world, name="pen_service",
    start="trafo_ring", end="house_foundation",
    conductor_type="pen",
    wire_radius=0.005,
    cross_section="from_radius",        # finite impedance
    coupling_to_soil="isolated",        # insulated cable, no leakage
    discretize_segment_length=2.0,      # split into 2 m sub-segments
)

# --- Source + solve ---------------------------------------------------
gf.create_source(world, attached_to="trafo_ring", magnitude=1.0)
engine = gf.create_engine(
    backend="image", segment_length=0.5, frequencies=[50.0],
)
result = engine.solve(world)

print("substation cluster Z =",
      result.cluster_impedance("trafo_ring")[0])
print("house cluster Z      =",
      result.cluster_impedance("house_foundation")[0])
```

## How the clusters are formed

The three ideal bonds (cross_section=None) fuse the ring and the
three rods into **one** cluster — they share a common potential at
every frequency. The PEN service drop has a finite cross-section
(`"from_radius"` resolves to $\pi r_w^2$) and therefore lives as a
*branch* in the nodal system: the substation cluster and the house
cluster are independent galvanic islands joined by a single
$R + j\omega L$ branch. That branch's series resistance is
$R = \rho_\text{Cu}\,L/A$ and — when `inductance_model="neumann"`
is set — the self- and mutual-inductance integrals across the
discretised PEN sub-segments are added on top.

## What to look at

- `result.cluster_impedance("trafo_ring")` gives the *measured*
  grounding impedance of the substation cluster (everything
  fused by the bonds) as seen at $50\;\mathrm{Hz}$. The house
  foundation only contributes via the PEN branch — its own
  cluster keeps a separate potential.
- `result.electrode_currents` shows how the substation's 1 A
  injection distributes itself between the ring, the rods and the
  PEN branch.

## Where to go next

- Add the Neumann inductance kernel (`inductance_model="neumann"`)
  and watch the PEN branch impedance grow with frequency — see
  [05 — Inductive coupling](05_inductive_coupling.md).
- Replace the homogeneous soil with a layered profile — see
  [03 — Multi-layer soil models](03_multilayer_soil.md).
- Build the same kind of cluster for a whole village with the
  imperative builder — see
  [06 — TN-Model with `OrtsnetzLayout`](06_tn_model.md).
