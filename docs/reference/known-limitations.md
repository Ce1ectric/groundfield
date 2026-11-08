# Known limitations

What `groundfield` does not do, does only approximately, or does only
with certain backends — with the size of the effect where it has been
measured. The numbers are recorded in the docstrings of the modules
named in each entry (they are the authority and are updated with the
code) and in the [changelog](../changelog.md) of the release that
measured them. The [validation page](../explanation/validation.md)
shows what agrees with independent references and how well.

## Scope of the physics

- **Quasi-static, below 1 kHz.** Displacement currents in the soil and
  wave propagation along conductors are neglected. The ratio of
  displacement to conduction current, $\omega\varepsilon_0\varepsilon_r\rho$,
  is below $10^{-2}$ up to 1 kHz for $\rho \le 1000\,\Omega\,\mathrm{m}$ and
  $\varepsilon_r = 81$, but reaches about $4.5\cdot10^{-2}$ in rock of
  10 kΩ·m ([Soil](api/soil.md)). Higher frequencies are accepted without
  a warning but are outside the validated range; impulse and lightning
  transients are out of scope.
- **Linear, isotropic, horizontally layered soil.** No soil ionisation,
  no lateral changes of resistivity, no anisotropy, no frequency
  dependence of $\rho$; soil and metal are non-magnetic.
- **No capacitive coupling** between conductors (negligible below
  1 kHz for buried conductors).
- **No impressed voltage sources.** A `VoltageSource` raises
  `NotImplementedError`; solve with a current source and scale, the
  problem is linear.
- **Not computed:** the inductive reduction factor of cable screens and
  earth wires; mutual impedances (`BranchType`) for groundinsight — one
  `BusType` describes one terminal.

## Current ansatz and source kernels

- **Uniform current per electrode (image family).** `image` and
  `image_2layer` assume a uniform current per unit length on each
  electrode (and soil layer). Impedances of rods and rings are within
  about 1 % of the per-segment `mom` solution, those of grids and stars a
  few per cent above it (default line kernel: 20 m × 20 m grid with
  4 × 4 meshes +3.9 %, 40 m × 40 m with 8 × 8 meshes +5.3 %, 4-arm star
  +4.3 %); the generators build mesh foundations as one electrode per
  wire line, which brings them to +0.3 … +0.6 %. Surface
  potentials near grids, meshes, stars, strip ends and bonded
  multi-electrode clusters are worse. Against the per-segment `mom`
  solution, with the default line kernel, touch voltage with the feet 1 m
  outside the footprint and step voltage from 1 m to 2 m outside: a
  20 m × 20 m grid with 4 × 4 meshes at a corner +20 % / −21 % (the step
  from the point above the corner 1 m outwards −28 %, as in the
  [`image` engine table](../explanation/engines/image.md#validation)); a
  4-arm star with 5 m arms beyond a tip +8 % / −9 %; rings, rods and
  closed loops within 2 % (the point kernel isolates the ansatz: +19 % /
  −21 %, +7 % / −10 %, within 1 %; `groundfield.postprocess.safety`).
  The safety functions warn with `UniformCurrentFieldWarning`; use
  `backend="mom"` for such field values.
- **Point kernel** (`kernel="point"`, and always in `mom` and
  `mom_sommerfeld`). Its row sums carry an error of about $-0.23/L$, so
  grounding resistances converge 0.9–1.9 % below the exact
  uniform-current value (`Engine.kernel`). Horizontal wires close to the
  layer interface need segments much shorter than $2\lvert z - h_1\rvert$:
  a ring 5 cm below the interface of 100 over 1000 Ω·m reads −46 % at
  0.5 m segments (changelog 2026.11.0, 10.28). A horizontal point-kernel
  segment within 0.6 segment lengths of the interface, on either side of
  it, warns with `PointKernelInterfaceWarning` (3–17 % low in the
  measured cases with an electrode through the interface, −2.5 % for a
  ring 0.2 m above it); one closer than 0.387 segment lengths, or one
  whose three-point rule would reach above the surface, raises
  `BackendCapabilityError` (a ring 1 µm above the interface read −1931 Ω
  instead of 4.18 Ω before). Use the line kernel or shorter segments.
- **Line kernel in two-layer soil.** Tagg/Sunde images farther than six
  segment lengths and the smooth part of the cross-layer correction stay
  in point form: a straight 10 m strip deviates by up to 0.05 % from its
  exact uniform-current resistance, not monotonically in the segment
  length (`Engine.kernel`). With an electrode through the interface, a
  galvanic distributed conductor needs sub-pieces of the order of $h_1$
  (changelog 2026.11.0, 10.24).
- **Cost of the line kernel.** About 2.7 times the solve time of the
  point kernel for 16 house grids ($N$ = 1920 segments) with numba
  installed, 5.7 times without (`Engine.kernel`); the extra work grows
  like $N^2$.

## Soil layers

- **Three or more layers** are solved only by `mom_sommerfeld` (and the
  experimental `fem`). `FieldResult.potential` — and with it touch and
  step voltages, potential plots, potential-path CSV and the VTK field
  export — supports homogeneous and two-layer soil only and raises
  `NotImplementedError` for three or more layers.
- **Rod ends at a layer interface.** The thin-wire model has no end
  face, so the contact of a rod end with the layer beyond is all or
  nothing: against an axisymmetric model of the solid rod (4.5 m,
  $a$ = 8 mm, 1000 / 30 Ω·m) the thin wire reads +21 % with its tip at
  the interface, +7 % 1 mm above it and −4.6 % 1 mm below it (within a
  few per cent beyond two wire radii). A rod head or tip within one wire
  radius of an interface warns with `InterfaceContactWarning` and is
  listed by `diagnose()` (`interface_contact`).
- **High layer contrast.** The Tagg/Sunde series converges like
  $\lvert K\rvert^n$; the default cap of 300 terms covers
  $\lvert K\rvert = 0.94$ and warns (`SeriesTruncationWarning`) beyond.
  `mom_sommerfeld` may not converge for $\lvert K\rvert \to 1$
  (`SommerfeldConvergenceWarning`); use `image_2layer` for two layers.

## Backends

- **`fem` is experimental.** It replaces every cluster by an equivalent
  hemisphere: biased for clusters with several electrodes (−35 % for a
  2 m ring with three 4 m rods), blind to the coupling between clusters,
  and wrong when a layer interface lies within the electrode or the
  hemisphere (−92 % for a 10 m ring over $h_1$ = 2 m, −70 % for a 10 m
  strip below a 0.5 m top layer); it raises when the electrode reaches a
  layer its hemisphere does not (`groundfield.solver.fem`). It ignores
  inductance, `return_to`, junctions and concrete shells
  (`BackendCapabilityWarning` or `BackendCapabilityError`).
- **Concrete shells** (`concrete_shell_coefficient_ohm_m`) are
  represented by the image family and `solve_mutual_*` only.
- **Aliases.** `cim`, `bem` and `image_nlayer` run `image` and `mom`;
  comparing them with their targets adds no independent information.

## Earth return and conductors

- **`carson_series`** is a per-metre approximation for long parallel
  conductors over homogeneous soil; in layered soil it uses $\rho_1$
  (with a warning), and it rejects conductor segments in the soil-surface
  plane. **`perfect_mirror`** is a frequency-independent diagnostic
  baseline, not a model of buried conductors below 1 kHz. Use the default
  `sommerfeld` for numbers.
- **Cost of the exact Sommerfeld assembly** grows with the number of
  segment pairs and frequencies; `sommerfeld_assembly="tabulated"` is the
  faster option for large networks (opt-in in this release).
- **Skin effect** is opt-in (`resistance_model="skin"`, solid round
  conductors); the proximity effect between conductors is not modelled.

## Post-processing and export

- **EN 50522 permissible touch voltage** is interpolated in Table B.4 for
  fault durations from 50 ms to 10 s and clamped outside; 80 V for
  long-duration faults needs `long_duration=True`.
- **Touch-voltage envelope** samples a ring of directions around the
  footprint and the centres of closed meshes — of grids, rings and the
  generated mesh foundations (bonded wires, one per wire line), whose
  cells lie under the building: pass `interior=False` there. The inside
  of a closed building outline (a closed polyline) is sampled only with
  `interior=True`.
- **Reduced models.** The standard form is a fixed ansatz — check
  `max_relative` (warning above 5 %). A vector fit belongs to one soil
  (`rho_at_fit`). Passivity is checked from 0 Hz to twice the highest
  fitted frequency only.

## Generators and OSM

- **Foundation outline.** By default a footprint becomes its oriented
  minimum bounding rectangle; `foundation_outline="polygon"` follows the
  outline (ring-style specs only). Outlines smaller than 1 m are widened
  with a warning. Every `building=*` object is a footprint unless excluded
  with `exclude_building_tags`.
- **Measurement leads** of the in-line set-up are laid side by side; their
  mutual inductance depends on the logarithm of that spacing, which is a
  modelling choice, not a measured value.
- **Overpass** answers are capped at 16 MiB; live queries need network
  access (tests and tutorials use stored answers).

## Diagnostics and problem size

- **Problem size.** More than 20 000 segments (`HARD_LIMIT`) raise
  `ResourceLimitError` before any allocation; the dense system grows like
  $N^2$ in memory and work per excitation.
- **Touching electrodes.** `diagnose` reports wires of different
  electrodes that are closer than five times the sum of their radii, but
  not two electrodes that meet at a common segment end point (for example
  a rod through a node of a separate grid). Metal in contact is one
  galvanic cluster: bond such electrodes with a conductor.
