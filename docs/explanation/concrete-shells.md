# Concrete-encased electrodes

A foundation earth electrode (*Fundamenterder*, DIN 18014) lies inside the
concrete of the strip foundation: every ampere it leaks has to cross the
concrete before it reaches the soil. Concrete conducts like soil — wet
concrete at tens of Ω·m, dry concrete at thousands — so the encasement
adds a series resistance that can be negligible or dominant depending on
the moisture. This page explains the model, the single mechanism that
carries it into the solve since 2026.11, and how to convert the two input
conventions.

## Mathematical and physical context

Around a wire of radius $r_a$ the concrete forms a cylindrical shell of
thickness $t$ (outer radius $r_b = r_a + t$) and resistivity $\rho_c$;
outside lies the soil. Leakage current crosses the shell radially, and in
the cylindrically symmetric limit the shell and the soil add in series
along the radius (Sunde 1968, §4.4). Per metre of wire the shell
contributes

$$
R'_\text{shell} = \frac{\rho_c}{2\pi}\,\ln\frac{r_b}{r_a} \equiv C
\qquad\text{(in } \Omega\cdot\text{m)},
$$

and the soil sees a bare wire of the enlarged radius $r_b$. The
decomposition is exact for an infinite straight wire and a good
approximation for a finite electrode as long as the leakage current varies
slowly along the wire on the scale of the shell — always the case below
1 kHz. Concrete around the wire is local: modelling it as a horizontal
soil layer would put concrete under the whole neighbourhood, which is
why `groundfield` refuses the planar interpretation (ADR-0012). Fagan &
Lee (1970) and Ufer (1964) measured such concrete-encased electrodes;
their low resistance in moist concrete is the reason foundation
electrodes work at all.

## One mechanism: the shell coefficient

Every electrode kind (rod, ring, strip, polyline, star, mesh, grid mesh)
has the field `concrete_shell_coefficient_ohm_m` — the coefficient $C$
above. The image family (and `solve_mutual_matrix` /
`solve_mutual_field`) consumes it per segment: a segment of length
$\Delta s$ adds $C/\Delta s$ to its own diagonal entry of the reaction
matrix — the voltage drop of its share of the leakage current across its
piece of shell — and nothing to the off-diagonal entries, because shells
of different segments do not couple. The wire radius of an encased wire is
the outer radius $r_b$.

Under the uniform current of an owner (electrode, or electrode per layer)
of length $L$ the per-segment terms add up exactly to the lumped series
resistance

$$
Z(C) - Z(0) = \frac{C}{L},
$$

for every electrode kind, homogeneous or layered soil, in `Engine.solve`
and in `solve_mutual_matrix` alike
(`tests/test_concrete_shell_coefficient.py`). Backends outside the image
family reject electrodes with a non-zero coefficient
(`BackendCapabilityError`).

### Why one mechanism (findings 10.12–10.14)

Up to 0.15.0 the two variants of ADR-0012 travelled different roads. The
*distributed* variant put the shell on the wires; the *lumped* variant
recorded a total shell resistance in the world registry
`World.concrete_shell_corrections`, which only the TN generator turned
into a series resistance on the PEN service drop. Every other path —
`Engine.solve` of any other world, `solve_mutual_matrix`,
`solve_mutual_field`, the layout builder — ignored the registry but kept
the inflated wire radius. For the AP1 catalogue, which builds its
foundation matrix with `solve_mutual_matrix`, the concrete therefore
lowered the impedance instead of raising it at low soil resistivity. A
lumped shell spread over a mixed site also pushed rod and brace current
through the foundation's shell, and mesh braces declared unencased got the
inflated radius.

Since 2026.11 both `concrete_model` settings of the generators produce the
same representation:

1. every encased perimeter wire carries $C$ and the radius $r_b$;
2. cross-braces and rods stay bare ($r_a$, $C = 0$);
3. no shell resistance on PEN service drops (it would count twice);
4. substation and cabinet electrodes get $C$ when their specification is
   encased;
5. `concrete_model` only chooses the owner granularity of an axis-aligned
   ring foundation (`"lumped"`: one grid-mesh owner, `"distributed"`: one
   owner per wire); the two agree to well below the 2 % criterion of
   ADR-0012 (`tests/test_concrete_one_mechanism.py`).

`World.concrete_shell_corrections` is deprecated (it still loads and warns
at solve; removal no earlier than 2027.5). The golden case
`mutual_foundations_lumped_concrete` records the effect on a foundation
matrix: its diagonal rises because the lumped shell now reaches
`solve_mutual_matrix`; the release notes give the magnitudes for AP1-type
sites.

## Converting between the conventions

The helpers in `groundfield.geometry.electrodes` convert between the
material description, the coefficient and a lumped resistance:

| Helper | Formula | Use |
|---|---|---|
| `concrete_shell_coefficient(rho_c, wire_radius, thickness)` | $C = \frac{\rho_c}{2\pi}\ln\frac{r_a + t}{r_a}$ | from concrete resistivity and cover |
| `shell_coefficient_from_resistance(electrode, R)` | $C = R\,L$ | from a lumped shell resistance of the whole electrode |
| `shell_resistance_from_coefficient(electrode, C=None)` | $R = C/L$ | the lumped resistance a coefficient represents |

$L$ is `electrode.total_length()`, all wires of the electrode. A lumped
resistance of a site should be spread over its encased electrodes only —
spreading it over rods and braces was the error behind the mixed-site
deviations of finding 10.13. With $\rho_c = 100$ Ω·m, 5 cm cover and a
5 mm wire, $C = (100/2\pi)\ln 11 \approx 38.2$ Ω·m, i.e. 3.8 Ω for 10 m of
encased wire.

## Validity

| Condition | Statement |
|---|---|
| Geometry | thin shell around a thin wire; the shell must not overlap a neighbouring wire's shell |
| Current | radial through the shell; the series decomposition holds below 1 kHz for foundation sizes |
| Concrete | homogeneous $\rho_c$ per electrode; moisture states span about 30 Ω·m (wet) to tens of kΩ·m (dry), see ADR-0012 |
| Field quantities | the surface potential is that of the bare wire of radius $r_b$ fed through the shell; the shell resistance itself is a series element |
| Backends | `image`, `image_2layer`, `solve_mutual_*`; others raise `BackendCapabilityError` |

## Literature

- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, §4.4 (wire in concrete in conducting earth).
- Ufer, H. G. (1964). Investigation and testing of footing-type grounding
  electrodes for electrical installations. *IEEE Transactions on Power
  Apparatus and Systems* 83(10), 1042–1048.
- Fagan, E. J., & Lee, R. H. (1970). The use of concrete-enclosed
  reinforcing rods as grounding electrodes. *IEEE Transactions on Industry
  and General Applications* IGA-6(4), 337–348.
- DIN 18014:2023-04. *Fundamenterder — Planung, Ausführung und
  Dokumentation*. DIN, Berlin.
- Tagg, G. F. (1964). *Earth Resistances*. George Newnes, London, ch. 7.

Architecture decision: [ADR-0012](../adr/0012-foundation-concrete-encasement.md)
and its 2026.11 amendment.
