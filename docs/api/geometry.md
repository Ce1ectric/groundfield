# Geometry

The ``geometry`` subpackage describes the physical shape of every
buried metallic object — rods, rings, strips, polylines, stars, mesh
and grid-mesh electrodes — together with the wire-segment
representation that the field solver consumes.

## Physical background

A grounding electrode is a finite metal body buried in conducting
soil. For frequencies below 1 kHz it can be treated as a thin-wire
boundary along which the potential is enforced and the current
distribution is unknown (or, in the cheaper image backends,
prescribed as uniform). All standard geometries are reduced to a
collection of straight cylindrical segments

$$
\Gamma_i = \{ \mathbf{x}(\xi) = \mathbf{x}_i^{(0)}
                  + \xi\,(\mathbf{x}_i^{(1)} - \mathbf{x}_i^{(0)}) :
                  \xi \in [0, 1] \},\qquad i = 1\ldots N,
$$

each carrying a wire radius $a_i$ and a material tag. The
discretisation length $\Delta s$ is set by
:attr:`groundfield.solver.engine.Engine.segment_length`. The
geometry layer is intentionally decoupled from the solver, so the
same electrodes can be fed into the closed-form image backend, MoM,
or FEM without modification.

For mesh and grid electrodes the segment count per wire is aligned to a
whole multiple of the mesh-span count, so the grid crossings fall on
segment **endpoints** rather than on the midpoint point sources. This
avoids a discretisation trap in which a longitudinal and a transverse
segment midpoint would coincide on a crossing — collapsing to the
singularity clamp and inflating the grid resistance by ~2× for the
uniform-current image backend. The effective $\Delta s$ is therefore
never coarser than requested and the grid resistance is stable for any
`segment_length` (see `notebooks/44_grid_discretization_robustness.ipynb`).

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

# Vertical driven rod
rod = gf.create_electrode(
    world, "rod", name="g1",
    position=(0.0, 0.0, 0.0), length=1.5, wire_radius=0.005,
)

# Buried ring earth at house-foundation depth
ring = gf.create_electrode(
    world, "ring", name="g2",
    center=(10.0, 0.0, 0.8), radius=2.0, wire_radius=0.005,
)

# Rectangular mesh (e.g. substation grid)
mesh = gf.create_electrode(
    world, "grid_mesh", name="g3",
    corner=(-10.0, -10.0, 0.5), size=(20.0, 20.0),
    n_x=11, n_y=11, wire_radius=0.005,
)

# Pre-solve resolution / segment-budget check
for warning in gf.check_segment_resolution(
    world, gf.create_engine(backend="image", segment_length=0.1)
):
    print("WARN:", warning)
```

For programmatic generation of large typical worlds (5 – 200
single-family houses, stochastic electrode mixes) see
[Generators](generators.md).

### Strip electrode concrete shell (new in 0.6.0)

:class:`StripElectrode` carries an optional
``concrete_shell_coefficient_ohm_m`` field (default ``0.0`` —
historic behaviour). When set, the per-meter Sunde coefficient
$C = \rho_c/(2\pi)\,\ln(r_b/r_a)$ in Ω·m enters the
``image`` / ``image_2layer`` self-action: every segment of length
$\Delta s$ in this strip has its MoM diagonal augmented by
$C/\Delta s$, the radial voltage drop through the concrete shell.
This is the V2 distributed path of
[ADR-0012](../adr/0012-foundation-concrete-encasement.md). The
field is populated automatically by
:meth:`~groundfield.generators.grounding.GroundingSystemSpec.build_at`
when a :class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`
with ``concrete_model="distributed"`` is materialised; it is not
intended to be set by hand on isolated strip electrodes outside
the generator pipeline (but does work there, see the closed-form
verification in ``tests/test_concrete_encasement.py``).

### Star electrode (Sternerder)

:class:`StarElectrode` is the classic $n$-point star: ``n_arms`` equal
horizontal wires of length $L$ radiate from a common centre node at one
depth, uniformly spaced by $360^\circ / n$. All arms share the centre,
so the star is a single galvanic cluster. Its closed-form grounding
resistance follows Dwight (1936, Eqs. 23–26),

$$
R = \frac{\rho}{2\,n\,\pi\,L}\,
    \Bigl(\ln\tfrac{2L}{a} + \ln\tfrac{2L}{s}
          + c_0 + c_1\tfrac{s}{L}
          + c_2\tfrac{s^2}{L^2} + c_3\tfrac{s^4}{L^4}\Bigr),
\qquad s = 2t,
$$

with per-$n$ coefficients $c_k$ tabulated for $n \in \{3, 4, 6, 8\}$ in
:func:`groundfield.references.dwight1936.n_point_star` ($a$ = wire
radius, $t$ = burial depth, $s = 2t$ the image distance). The image
backend reproduces this to ~1 % (see
``tests/test_dwight_references.py``).

```python
star = gf.create_electrode(
    world, "star", name="g4",
    center=(0.0, 0.0, 0.5), n_arms=4, arm_length=5.0,
    orientation_deg=0.0, wire_radius=0.005,
)
```

## API reference

::: groundfield.geometry
