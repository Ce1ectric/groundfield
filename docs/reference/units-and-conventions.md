# Units and conventions

## Units

Every physical input and output is in SI units, without prefixes.

| Quantity | Unit | Examples |
|---|---|---|
| coordinates, lengths, radii, depths | m | `position`, `radius`, `wire_radius`, `h_1`, `segment_length` |
| cross section | **m²** | `cross_section=95e-6` for 95 mm² (a value above 10⁻² m² warns with `CrossSectionUnitWarning`) |
| resistivity of soil, concrete, metal | Ω·m | `rho`, `rho_1`, `concrete_rho_ohm_m`, `resistivity=1.68e-8` (copper) |
| concrete-shell coefficient | Ω·m | `concrete_shell_coefficient_ohm_m` |
| current | A (RMS) | `CurrentSource.magnitude` |
| potential, voltage | V (RMS) | `electrode_potentials`, `touch_voltage` |
| impedance | Ω | `input_impedance`, `cluster_impedance`, `solve_mutual_matrix` |
| frequency | Hz | `Engine.frequencies` |
| inductance | H; per unit length H/m | `coupling.inductance` |
| per-unit-length impedance | Ω/m | `references.earth_return`, `references.pollaczek` |
| time (fault duration) | s | `permissible_touch_voltage_en50522(t_clear_s)` |
| angles | degrees | `phase_deg`, `orientation_deg` |

Names follow two rules: the physics model uses bare SI values and the
literature symbols (`rho`, `rho_1`, `h_1`, `h`); the generator specs
carry a lower-case unit suffix (`depth_m`, `concrete_rho_ohm_m`,
`frequency_hz`, `source_magnitude_a`).

## Coordinates

- Right-handed Cartesian $(x, y, z)$ in metres. In worlds built from
  map data, $x$ points east and $y$ north (local frame of the
  projection origin).
- **$z$ is the depth below the soil surface, positive downwards.** The
  soil surface is $z = 0$, the soil is $z > 0$, the air $z < 0$
  (overhead conductors). A ring "0.8 m deep" has `center=(x, y, 0.8)`.
- A rod's `position` is its upper end; it extends downwards by
  `length`.
- ParaView and other viewers assume $z$ upwards: apply a transform with
  scale $(1, 1, -1)$ to VTK files.
- **Projected coordinates** (UTM, Gauss–Krüger) may be used as they are.
  When a horizontal coordinate of a world exceeds 10 km, the solvers
  (`image`, `image_2layer`, `mom`, `mom_sommerfeld`, `solve_mutual_*`) work
  in a local frame: the world shifted exactly by the centre of its
  horizontal bounding box, rounded to whole metres; point sources are
  reported in the user's coordinates, and `FieldResult.potential`
  evaluates in a local frame of the sources in the same way (since
  2026.11). Segment counts and straight runs allow the round-off of the
  coordinates' own magnitude, 64 $\varepsilon |x|$ (0.46 µm at an easting
  of 32.6 · 10⁶ m). A world moved from (0, 0) to the UTM origin
  (32 604 321.123 m, 5 812 345.678 m) then changes its impedance by
  3 · 10⁻¹¹ with the point kernel (`mom`, `mom_sommerfeld`, `image` with
  `kernel="point"`), the effect of the rounding of its vertices; up to
  2026.11 an edge of a whole number of segments got one more there (a
  generated 10 m mesh foundation 508 instead of 440 segments, +2.6 · 10⁻⁴
  in `mom`). With the default line kernel the impedance moves by up to
  2.4 · 10⁻⁷ wherever the world stands (up to 1.2 · 10⁻⁷ already for a shift
  by (0.1, 0.2) m): the kernel switches from its exact integral to its
  far form at a midpoint distance of six segment lengths, where pairs six
  segments apart on a straight wire sit exactly, so round-off decides
  their form
  ([known limitations](known-limitations.md#current-ansatz-and-source-kernels)).
  Worlds within 10 km of the origin are solved exactly as before.

## Phasors and frequency

- Time convention $e^{+j\omega t}$, $\omega = 2\pi f$: a quantity
  $x(t) = \sqrt{2}\,\operatorname{Re}\{\underline{X}\,e^{j\omega t}\}$ is
  represented by its **RMS phasor** $\underline{X}$. An inductive
  impedance has a positive imaginary part, $Z = R + j\omega L$.
- The Laplace variable of the vector fits is $s = j\omega = j\,2\pi f$.
- Every result is a list with one value per frequency, in the order of
  `Engine.frequencies` (the order is never changed; a non-monotonic
  list warns with `EngineFrequencyOrderWarning`).
- Frequencies must be finite and non-negative; $f = 0$ (DC) is a valid
  operating point. The models are **quasi-static** and validated for
  $f < 1\,\mathrm{kHz}$: displacement currents in the soil and wave
  propagation along the conductors are neglected. Higher frequencies
  are accepted but outside the validated range.
- A fault-current spectrum is combined as a true RMS value,
  $U_\text{rms} = \sqrt{\sum_k \lvert\underline{U}_k\rvert^2}$
  (`touch_voltage_spectrum`).

## Potentials, currents and signs

- **Potential reference.** Every potential is measured against remote
  earth, $\varphi \to 0$ far from all electrodes. The soil surface is
  insulating (no current into the air).
- **Source current.** A `CurrentSource` with `magnitude` $I > 0$ and
  `phase_deg` $= 0$ drives $+I$ into the node `attached_to` and takes
  it back at `return_to` (or at remote earth).
- **Electrode currents.** `FieldResult.electrode_currents[name]` is the
  current an electrode leaks **into the soil**: positive at the feeding
  earth, negative at the electrode that collects the current of a
  two-terminal source. With one source of $+1$ A the potentials of the
  fed earthing system are positive (earth potential rise).
- **Impedances.** Input, cluster and electrode impedance are defined in
  the [quickstart](../getting-started/quickstart.md#three-impedances-of-one-solve);
  for a passive earthing system their real parts are positive.
- **Touch and step voltage.** $U_T = U_E - \varphi(\text{feet})$ and
  $U_S = \varphi(\mathbf{r}_1) - \varphi(\mathbf{r}_1 + 1\,\mathrm{m}\cdot\hat{\mathbf{e}})$,
  complex phasors in V for the given source current; compare
  magnitudes with the permissible values.
- When comparing with another tool or a measurement, check its sign
  convention for injected currents and node potentials.

## Material defaults

| Parameter | Default | Note |
|---|---|---|
| wire radius of electrodes and conductors | 5 mm | `wire_radius=0.005` |
| conductor resistivity | $1.68\cdot10^{-8}\,\Omega\,\mathrm{m}$ | copper |
| relative permittivity of the soil | stored, not used | quasi-static model (`InertConfigurationWarning` when changed) |
| magnetic permeability | $\mu_0 = 4\pi\cdot10^{-7}\,\mathrm{H/m}$ | soil and conductors non-magnetic |
