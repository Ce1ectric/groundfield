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
above. The image family, `solve_mutual_matrix` / `solve_mutual_field` and,
since 2026.11, `mom` and `mom_sommerfeld` consume it per segment: a
segment of length
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
(`tests/test_concrete_shell_coefficient.py`). `fem` consumes it as well:
its axisymmetric model as the surface conductance $1/(2\pi r_b C)$, its
three-dimensional model as the radial resistance $C$ per length between the
metal and its smoothed line source ([fem](engines/fem.md)). `mom` and
`mom_sommerfeld` rejected it up to 2026.11 (see
[In the moment method](#in-the-moment-method)).

### In the moment method

`mom` and `mom_sommerfeld` solve one pulse current $I_k$ per segment, all
segments of a node at one potential (point matching). A segment of length
$\Delta s_k$ drops $C I_k/\Delta s_k$ across its piece of shell, so the
segment system reads

$$
\bigl(K + \operatorname{diag}(C_k/\Delta s_k)\bigr)\,\mathbf I
= \varphi\,\mathbf 1, \qquad \sum_k I_k = I,
$$

with $K$ the soil reaction matrix of the wire of outer radius $r_b$; the
reported segment potentials, and with them the node potential, are those
of the metal, and the field in the soil is that of the leakage currents.
Unlike the uniform current of an owner, the current redistributes. The
equipotential solution minimises $\mathbf I^\mathsf{T}(K + S)\mathbf I$
over $\sum_k I_k = I$, so the shell raises $Z$ by at least $C/L$ — the
shell term alone is smallest for a current uniform per length — and by at
most $C/L$ plus the excess of the uniform current over the equipotential
one; a dry shell evens the current out ($Z - C/L$ is within $5\cdot10^{-6}$
of the uniform-current value of the bare wire at $C = 3.8\cdot10^{5}$ Ω·m
for a 3 m rod, an 8 m × 6 m grid and a three-arm star). Two methods check
the implementation (evidence of stream w2mom,
`tests/test_concrete_shells_mom_family.py`):

* **Same matrix physics.** The image family with one owner per segment
  (point kernel, every segment its own uniform-current owner, all bonded)
  solves the same system: it agrees with `mom` to $7\cdot10^{-16}$ for
  rods, a rod through the layer interface, rings, strips, grids, outlines
  and stars in 100 Ω·m and in 30 / 200 Ω·m ($h_1$ = 4 m), 5 cm of
  100 Ω·m concrete. `mom_sommerfeld` reproduces `mom` to about $10^{-8}$
  where neither engine warns ($10^{-9}$ … $1.5\cdot10^{-8}$ for a 10 m grid
  in four two-layer soils, bare and encased; audit W2B-07; with the layer
  interface 0.2 m below the wires $9\cdot10^{-8}$ under
  `PointKernelInterfaceWarning`, audit W2R2-09): that is the agreement of its Sommerfeld
  quadrature with the Tagg/Sunde series of `mom` without a shell, the
  shell increments $Z(C) - Z(0)$ of the two engines agree to
  $1.4\cdot10^{-10}$.
* **Closed forms.** The segments of a ring are equivalent (its matrix is
  circulant), so its current stays uniform and the shell adds exactly
  $C/(2\pi r)$: `mom` and `mom_sommerfeld` give it to $10^{-14}$ in
  homogeneous, two-layer and three-layer soil for $C$ = 11 … 382 Ω·m. For
  a 3 m rod from the surface in 100 Ω·m, Sunde's decomposition with
  Dwight's rod formula, $R = C/L + \frac{\rho}{2\pi L}\bigl(\ln\frac{4L}{r_b} -
  1\bigr)$, the image family's uniform current (line kernel) lies
  0.14 … 0.20 % above the formula and rises by exactly $C/L$ (5 cm of 30
  or 100 Ω·m concrete, $r_b$ = 55 mm, 0.5 … 0.125 m segments). `mom` lies
  below it by the point kernel's deficit, which grows as the segments
  shorten while they stay at least 2.99 wire radii long: with
  $r_b$ = 55 mm the bare rod lies 2.5 / 3.5 % below Dwight's formula at
  0.5 / 0.25 m segments, the encased one 1.4 … 2.7 % below Sunde's, rising
  by 1.008 … 1.022 $C/L$. A segment shorter than $2.99\,r_b$ takes the
  Galerkin self term on the diagonal (see [`mom`](engines/mom.md)), which
  lifts the deficit again: at 0.125 m ($2.3\,r_b$) the bare rod lies 2.4 %
  and the encased one 0.8 / 1.5 % below (100 / 30 Ω·m, rising by
  1.022 / 1.041 $C/L$). With $r_b$ = 20 mm (1.5 cm of 100 Ω·m), where
  0.5 … 0.0625 m segments all stay above $2.99\,r_b$, the deficit grows
  monotonically: bare 1.9 / 2.5 / 2.9 / 3.2 %, encased
  1.4 / 1.8 / 2.0 / 2.2 % at 0.5 / 0.25 / 0.125 / 0.0625 m. An own
  point-matching matrix written from the thin-wire formulas equals `mom`
  to $2\cdot10^{-16}$ wherever every segment is that long and departs from
  it by −2.2 % (bare rod, $r_b$ = 55 mm, 0.125 m), the self-term switch
  (audit W2B-07; evidence of stream fw2mom).

`mom` is therefore the per-segment reference against which the owner
model of the image family (one uniform current per straight wire) is
measured, also for encased electrodes. With 5 cm of 30 Ω·m (wet) concrete
around a 5 mm wire, 1 m segments (the outline 0.5 m), grids and stars
0.5 m deep and the outline 0.8 m, the owner model lies above one current
per segment by:

| Geometry, soil | line kernel against one owner per segment (line kernel), bare / encased | point kernel against `mom`, bare / encased |
|---|---|---|
| grids 10 m 2 × 2 … 40 m 8 × 8, 100 Ω·m | +0.36 … +0.70 % / +0.43 … +0.68 % | +0.29 … +0.66 % / +0.35 … +0.63 % |
| the same grids, 1 000 / 100 Ω·m, $h_1$ = 5 m | +0.43 … +0.64 % / +0.64 … +0.89 % | +0.33 … +0.50 % / +0.50 … +0.69 % |
| stars of 3 … 8 arms of 10 m, 100 Ω·m | +1.8 … +10.5 % / +2.7 … +13.6 % | +1.6 … +10.0 % / +2.4 … +13.0 % |
| the same stars, 1 000 / 100 Ω·m | +1.6 … +12.6 % / +3.1 … +19.7 % | +1.4 … +11.8 % / +2.7 … +18.8 % |
| outline 16 m × 1 m, 100 Ω·m | +0.38 % / +0.48 % | +0.39 % / +0.49 % |

The line-kernel column repeats the numbers quoted so far, which were
measured against the image family's per-segment owners (confirmed to the
printed digit); the `mom` column is the comparison with the point kernel,
the kernel of `mom`, which isolates the current model. For the generated
meshes of the next section (0.5 m segments) the owner model lies
+0.42 / +0.36 / +0.31 / +0.057 / +0.004 % (line kernel) and
+0.38 / +0.33 / +0.28 / +0.052 / +0.004 % (point kernel against `mom`)
above one current per segment for the house in 100 Ω·m at
$\rho_c$ = 50 / 100 / 150 / 1 000 / 5 000 Ω·m, at most +0.57 % and
+0.54 % for `medium_industry` (50 Ω·m); for the rectangles without
cross-braces of the table further down `mom` gives one owner per side
+0.230 % (80 m × 20 m) and +1.353 % (150 m × 3 m) above one current per
segment, bare and in 5 cm of 30 Ω·m (30 / 200 Ω·m, $h_1$ = 5 m, 1 m
segments), against +0.230 / +0.231 % and +1.336 / +1.339 % with the line
kernel.

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

1. every wire of an encased foundation carries $C$ and the radius $r_b$ —
   the perimeter and, for a mesh foundation, the cross-braces (see
   [Cross-braces of mesh foundations](#cross-braces-of-mesh-foundations));
2. rods and the other electrodes of a site stay bare ($r_a$, $C = 0$);
3. no shell resistance on PEN service drops (it would count twice);
4. substation and cabinet electrodes get $C$ when their specification is
   encased;
5. every rectangle without cross-braces (`style="ring"`, or `"mesh"` with
   `n_x = n_y = 1`) is one owner per side at every orientation (user
   decision Q38; up to then `"lumped"` made an axis-aligned one a single
   grid-mesh owner, see
   [Owners of a rectangle](#owners-of-a-rectangle-without-cross-braces)).
   `concrete_model` therefore has no effect and is deprecated (user
   decisions of 2026.11, removal no earlier than 2027.5): `"distributed"`
   still loads and warns, the default `"lumped"` and the files of 0.15.0
   load silently; drop the argument (see
   [Deprecations](../reference/deprecations.md)).

`World.concrete_shell_corrections` is deprecated (it still loads and warns
at solve; removal no earlier than 2027.5). The golden case
`mutual_foundations_lumped_concrete` records the effect on a foundation
matrix: its diagonal rises because the lumped shell now reaches
`solve_mutual_matrix`; the release notes give the magnitudes for AP1-type
sites. Since user decision Q38 its two square rings are one owner per side
as well, so the passive one carries equalising currents among its sides:
the matrix moves by a further 5.2e-4, and the field at the edges of the
passive foundation by +5.5 / −4.7 % (the default line kernel at 2 m and
1 m segments, the point kernel at 1 m; the golden engine's point kernel at
2 m segments gives +5.9 / −5.1 %, an effect of the coarse segments, not of
the kernel; audit R4B-13), 1.6 % of the largest value of the case's field
(the norm-wise measure of the golden harness; audit D3B-02).

## Cross-braces of mesh foundations

A mesh foundation (`FoundationElectrodeSpec(style="mesh")`, the
generators' default for buildings) joins its perimeter with cross-braces.
DIN 18014 lays the foundation earth electrode in the concrete together
with its connections, so the braces are encased like the perimeter: every
wire of an encased mesh carries $r_b$ and $C$ (user decision of 2026.11).
The shell resistance of the whole foundation is $C/L$ over every encased
wire, the per-wire shells $C/L_j$ in parallel;
`concrete_shell(spec, encased_length_m=...)` in
`groundfield.generators.materialise` reports it.

Two earlier rules preceded it. In 0.15.0 every wire of an encased mesh had
the radius $r_b$, but the braces carried no $C$: with `"distributed"` only
the perimeter wires had the shell, with `"lumped"` (the default) no wire
had it, and the lumped shell sat in `World.concrete_shell_corrections`,
which `Engine.solve` of a house never read (see
[Against 0.15.0](#against-0150)). The 2026.11 development line, between
review pass 10 and the user decision, kept the braces bare ($r_a$, no $C$);
it was never released, but the first table below is measured against it
because it isolates the braces.

The rule matters as soon as the concrete is drier than the soil. Bare
braces are a bare electrode inside the concrete: however dry the
concrete, the foundation could not exceed the spreading resistance of its
braces alone (9.60 Ω for the house mesh below in 100 Ω·m). Encased, the
shell term grows with $\rho_c$; for $\rho_c \gg \rho$ the current spreads
over the wires in proportion to their length and
$Z \to C/L + Z_\text{soil}$, with $Z_\text{soil}$ = 4.105 Ω for a uniform
current over the 60 m of the house mesh. Change of the input impedance
when the braces are encased as well, against the bare braces of the
development line:

| Mesh, soil | $\rho_c$ = 50 | 100 | 150 | 1 000 | 5 000 Ω·m |
|---|---|---|---|---|---|
| house, 100 Ω·m | −1.1 % | −0.03 % | +1.5 % | +47 % | +307 % |
| house, 30 / 100 Ω·m, $h_1$ = 5 m | +1.3 % | +6.2 % | +12 % | +154 % | +861 % |
| `medium_industry`, 100 Ω·m | −1.3 % | −0.03 % | +1.6 % | +46 % | +298 % |
| `medium_industry`, 30 / 100 Ω·m, $h_1$ = 5 m | +1.0 % | +4.8 % | +9.7 % | +120 % | +671 % |

House mesh: the generators' default (10 m, 2 × 2 cells, 0.8 m deep,
$r_a$ = 5 mm, $t$ = 5 cm); `medium_industry`: 20 m, 3 × 3 cells, same
wire and cover. Default engine (image family, line kernel, 0.5 m
segments). Concrete wetter than the soil lowers $Z$ a little (the wire
grows to $r_b$); concrete as conductive as the soil changes nothing but the
line kernel's self term. With the point kernel an encased mesh in concrete
of the soil's resistivity equals the bare mesh to round-off.

`Engine.solve` of the generated world and a hand assembly of its wires
agree to 4e-16: the wires as separate shell-free electrodes of radius
$r_b$, their mutual matrix from `solve_mutual_matrix`, $C/L_j$ added to
its diagonal, the wires bonded ($1/(\mathbf{1}^\mathsf{T} Z^{-1}
\mathbf{1})$). The assembly uses the solver's kernel and owner model, so it
checks how the generator wires $r_b$ and $C$, not the numbers. An
independent thin-wire boundary-element solution (its own line-source
kernel and image series, audit D2B) reproduces every entry of the table to
0.02 percentage points. Both rest on the owner model of the image family,
one uniform current per wire. With one current per segment, all segments
at one potential (the moment-method reference), encasing the braces of
the house mesh in 100 Ω·m changes $Z$ by +49.6 / +321.2 % (`mom`, 0.25 m
segments) instead of +46.9 / +307 % at $\rho_c$ = 1 000 / 5 000 Ω·m
(`medium_industry`: +49.5 / +313.7 % instead of +46.2 / +298 %).
groundfield's per-segment assembly with the line kernel gives
+49.2 / +318.5 % (`medium_industry` +49.2 / +312 %), and the
boundary-element solution agrees with it to 0.02 points; an own
point-matching solve of six hand-built wires (own segments, own kernel,
Sunde's $C$ by hand) reproduces `mom` on the generated mesh to
$4\cdot10^{-16}$ in $Z$ (audit W2B-06; evidence of stream fw2mom). The owner model is
1.6 / 2.8 % high for the mesh with bare braces (house; `medium_industry`
2.2 / 3.6 %; with the point kernel against `mom` 1.5 / 2.7 and
2.1 / 3.5 %) and at most 0.6 % high for the encased mesh (0.43 % at
50 Ω·m, 0.004 % at 5 000 Ω·m for the house in 100 Ω·m), whose shell evens
out the current over the wires. These owner-model numbers come from
groundfield's per-segment assembly with the line kernel at 0.25 m
segments; at 0.5 m segments it gives 0.42 %, and `mom` with its point
kernel puts the image family's point kernel 0.38 % above it (see
[In the moment method](#in-the-moment-method)).

The three-dimensional model of `fem` (each wire its own shell, the soil
meshed as a volume; 1.5 million nodes; evidence of stream w4fem3d) gives
+49.1 / +318.2 % for encasing the braces of the house mesh at
$\rho_c$ = 1 000 / 5 000 Ω·m — `mom` +49.6 / +321.3 % and the owner model
+46.9 / +307.0 % on the same world — and confirms the per-segment value
(+49.2 / +318.5 %) to 0.1 / 0.3 points. Against it the owner model is
+1.6 / +2.8 % high with bare braces and +0.14 / +0.03 % with encased ones;
`mom` lies 0.5 / 0.8 % and 0.2 / 0.06 % below it (re-measured after audit
W4B-01, evidence of stream fw4fem3d).

### Overlapping shells at junctions

Each wire carries its own radial shell; where wires meet, the shells
overlap within about $r_b$ of the junction and each wire counts the shared
concrete as its own. The three-dimensional model of `fem` can resolve the
concrete instead (`FemOptions.concrete_resistivity`): a square prism of
conformal radius $r_b$ around every straight encased wire, their union the
concrete, on the same mesh as the per-wire shells. The square keeps the
shell's resistance to the soil, but in the shell-dominated limit its
concrete is 3.8 % less resistive (its inner conformal radius is
0.914 $r_b$; audit W4B-04), and its faces and edges converge at first
order: it reads the conductance per metre 2.8 … 4.6 % high at the default
levels. So the junctions are
read from differences between configurations of the same length and the
same free ends — a straight 4 m bar against an L of two 2 m arms (the
corner), straight 4 m and 8 m bars against a T and a crossing (user
decision Q40). A 5 mm bar with 5 cm cover, 0.8 m deep in 100 Ω·m
(evidence of stream w4fem3d), each junction as the equivalent length of
wire (negative where the shared concrete leaks less than the overlapping
shells):

| Junction (not converged) | $\rho_c$ = 1 000 Ω·m | 5 000 Ω·m |
|---|---|---|
| corner | −1.5 cm | −1.4 cm |
| crossing | −6.0 cm | −2.8 cm |
| T-joint | +0.2 cm | +3.2 cm |

These numbers are not converged (audit W4B-05): one refinement of the prism
faces moved the corner by about 5 cm (+3.4 / +5.7 cm before), the crossing
and the T-joint were refined once only, and the impedance form of the same
runs gives −4 … −6 cm for every junction. A junction acts like a few cm of
wire, at most about $2\,r_b$, sign not settled. With each wire's own shell
an L of 4 m reads 0.35 % low, and the house mesh (60 m: 4 corners,
4 T-joints, 1 crossing) changes by about 1 % or less (0.35 … 0.85 % with
every sign aligned, re-measured after audit W4B-01). The image family, `mom`
and the 3-D model agree on the per-wire shell of these configurations to 0.01 … 0.3 %. No correction
is applied: the effect lies below the uncertainty of the concrete
resistivity and of the resolved model.

### Against 0.15.0

The release changes encased mesh foundations against 0.15.0 by the brace
rule above, by the shell that now reaches the solver for `"lumped"`
(findings 10.12–10.14), by the owners (one per wire, 10.39) and by the
default line kernel (10.28). Combined change of the input impedance of
the house mesh built with `GroundingSystemSpec.build_at`, default engine
of each version (0.15.0: point kernel; 2026.11: line kernel; 0.5 m
segments):

| House mesh, soil | `concrete_model` in 0.15.0 | $\rho_c$ = 50 | 100 | 150 | 1 000 | 5 000 Ω·m |
|---|---|---|---|---|---|---|
| 100 Ω·m | `"lumped"` | +7.0 % | +15.0 % | +23.0 % | +157 % | +782 % |
| 100 Ω·m | `"distributed"` | +2.2 % | +4.2 % | +6.6 % | +67.9 % | +399 % |
| 30 / 100 Ω·m, $h_1$ = 5 m | `"lumped"` | +16.5 % | +33.8 % | +51.0 % | +341 % | +1 701 % |
| 30 / 100 Ω·m, $h_1$ = 5 m | `"distributed"` | +4.8 % | +11.8 % | +20.2 % | +194 % | +1 045 % |

In 0.15.0 the house read 4.07 Ω in 100 Ω·m with `"lumped"` whatever the
concrete, and 7.20 Ω with `"distributed"` at 5 000 Ω·m; 2026.11 gives
35.9 Ω for both. The line kernel contributes +0.78 % without a shell and
+0.09 % at 5 000 Ω·m (100 Ω·m); with the point kernel on both versions the
changes are +6.2 … +781 % (`"lumped"`) and +1.5 … +398 %
(`"distributed"`) in 100 Ω·m. In a TN network the 0.15.0 lumped shell sat
on the PEN service drop of the house: the default TN network with four
such houses (`TnNetworkGenerator`, seed 0, 100 Ω·m, input impedance at the
substation) moves by −3.8 / −5.2 / −6.5 / −13.8 / −10.7 %
with `"lumped"` and by −0.3 / +1.0 / +2.5 / +34.5 / +120.9 % with
`"distributed"` at $\rho_c$ = 50 … 5 000 Ω·m (−1.7 % without a shell, from
other findings). The 0.15.0 models rebuilt by hand in the release code
reproduce the 0.15.0 numbers to round-off with the point kernel, and the
boundary-element solution reproduces the changes to 0.08 percentage
points.

Affected are generated mesh foundations with `concrete_rho_ohm_m` set.
Worlds without a shell, ring and building-outline foundations — the AP1
catalogue among them — and the golden cases do not change by the brace
rule.

## Owners of a rectangle without cross-braces

A rectangle without cross-braces (`style="ring"`, or `"mesh"` with
`n_x = n_y = 1`) is one uniform-current owner per side: one strip per side,
bonded to the local $-y$ side (the anchor), at every orientation, bare or
encased (user decision of 2026.11, Q38). Up to that decision the default
`concrete_model="lumped"` made an axis-aligned one — also one turned by a
multiple of 90°, built as the axis-aligned rectangle with swapped sides — a
single owner: one `GridMeshElectrode`, the same current density on all four
sides; only `"distributed"` or a rotation by another angle split it (audit
D3B-04). `concrete_model` has had no effect
since (see [Deprecations](../reference/deprecations.md)). A building outline
(`polygon_xy_m`) stays one closed polyline; since the user decision Q44
(with Q30) the image family gives it one owner per side as well, so a
rectangular outline equals the rectangle given by its size. Up to then it
was one owner and a rectangular outline read like the single owner did.

The change of the input impedance, one owner per side over one owner minus
one, is never positive. The four bonded owners carry the split of the
current among the sides that dissipates least (their owner matrix is
symmetric and positive definite), and the single owner's split, in
proportion to the side lengths, is one of those they can take; for a square
the two splits coincide and nothing changes. The decrease is largest for
narrow rectangles, conductive upper layers over resistive ones, and thick,
wet shells. Over the 39 soils of the AP1 grid, for a 5 mm wire at 0.8 m (the
generator defaults) and a short side of at least 1 m, the largest decreases
found are:

| Scope | aspect ratio ≤ 4, sides ≤ 80 m | aspect ratio ≤ 50, sides ≤ 150 m |
|---|---|---|
| bare, homogeneous soil | −0.089 % (7 m × 1.75 m) | −0.330 % (25 m × 1 m) |
| bare or shell up to 5 cm (the default), $\rho_c$ = 30 … 1 000 Ω·m | −0.320 % (80 m × 20 m, 5 cm at 30 Ω·m, 30 / 200 Ω·m, $h_1$ = 5 m) | −0.907 % (150 m × 7.75 m, bare, same soil) |
| bare or shell up to 0.2 m, $\rho_c$ = 30 … 1 000 Ω·m | −0.686 % (4 m × 1 m, 0.2 m at 30 Ω·m, 1 000 / 30 Ω·m, $h_1$ = 5 m) | −1.529 % (16 m × 1 m, same shell, $h_1$ = 10 m) |

The field description and the generators' module docstring quote a bound
with a margin over these values: per side the input impedance is lower by
less than 1 % up to aspect ratio 4 (short side at least 1 m, sides up to
80 m) and 2 % up to aspect ratio 50 (short side at least 1 m, sides up to
150 m); bare or with the default 5 cm shell by less than 0.4 % and 1 %.
Until the decision the same numbers, as one owner over one per side minus
one (0.089 … 1.553 %), bounded the difference between the two values of
`concrete_model`. The values come from a coarse scan of the scope (30
rectangles with short sides of 1 to 37.5 m and aspect ratios 1 to 50, bare
and with 5 cm and 0.2 m shells, 61 values of $\rho_c$) refined around each
maximum, 13 524 owner matrices in all (audit R2B-01, re-run for the
decision); none of them gives a positive change beyond round-off
($4\cdot10^{-16}$, for a square). Every case decreases most for the wettest
concrete of the scope (30 Ω·m), and the largest cases grow with the shell
thickness; drier concrete brings the two models together (−0.04 % at
5 000 Ω·m for the 16 m × 1 m case), a perfectly conducting shell separates
them up to −1.66 %. A foundation at 0.5 … 1.5 m or a wire of 2.5 … 10 mm
deepens the largest decreases by at most 0.10 percentage points (−0.78 %
for 4 m × 1 m at 0.5 m, −0.99 % for 150 m × 7.75 m with a 10 mm wire).
Below a short side of 1 m the decrease keeps growing (2 m × 0.5 m: −1.01 %,
10 m × 0.5 m: −2.22 % with the 0.2 m shell; 1.2 m × 0.3 m: −0.33 % with the
default shell), which is why the scope names the short side.
`Engine.solve` of the generated rectangles (both styles) against one
`GridMeshElectrode` with the same radius and shell reproduces the scan to
the printed digit; the line kernel at 0.25 m or 1 m segments and the point
kernel stay within the bound (−0.51 … −0.69 % and −1.38 … −1.53 % for the
two largest cases).

At the AP1 station (the ring of 4 m × 4 m at 0.6 m with three bonded 4 m
rods) the change is −0.0009 … −0.0026 % over the 39 AP1 soils, at 2 m and at
0.5 m segments (`Engine.solve` and the owner algebra of the seven wires
agree to $10^{-15}$): the square ring alone would not change, the rods break
its symmetry. The houses of the AP1 catalogue are building outlines and did
not change by Q38; one owner per side of an outline (user decision Q44)
moves the twelve outlines of the tutorial T06 layout by at most −0.004 %
each and the layout with the AP1 station by −0.03 … −0.40 % (5 cm of
100 Ω·m concrete or bare, four soils; evidence of stream w1sol).

**Model uncertainty.** The largest values belong to the 0.2 m shell, whose
outer radius of 0.205 m is not thin against the segments: at the default
0.5 m, segment length over wire radius is 2.4, below the thin-wire limit of
5, and the engine warns. There the numbers depend on how the solver places
the radius: an independent thin-wire boundary-element solution (audit R2B)
that averages the potential over the wire surface gives −0.594 % and
−1.447 % instead of −0.686 % and −1.529 %, the same at 0.25, 0.125 and
0.083 m segments; for the bare and 5 cm cases of the table it agrees with
groundfield to 0.001 points. Against one current per segment, all segments
at one potential (the moment-method reference), one owner per side is the
closer model, which is why the decision kept it: at the largest decreases
of the table one owner was 0.55 % (80 m × 20 m, bare) to 2.16 % (16 m × 1 m)
high, one per side is 0.23 … 0.90 % high (groundfield's per-segment
assembly at 0.25 m segments; for the bare cases the boundary-element
solution agrees to 0.001 points). Over the whole scope (audit D3B-01: 80
rectangles with short sides of 1 to 150 m, the 39 AP1 soils, bare and with
5 cm and 0.2 m shells, 61 values of $\rho_c$, 9 360 segment matrices;
segments of 0.5 m up to sides of 50 m, 1 m beyond) one owner per side is at
most 1.4 % high, the single owner up to 2.2 % (2.16 % for 16 m × 1 m),
and one current per segment
is never above one owner per side nor one owner per side above one owner:

| Scope (short side ≥ 1 m) | aspect ratio ≤ 4, sides ≤ 80 m | aspect ratio ≤ 50, sides ≤ 150 m |
|---|---|---|
| bare | +0.230 % (80 m × 20 m, 30 / 200 Ω·m, $h_1$ = 5 m) | +1.336 % (150 m × 3 m, same soil) |
| bare or shell up to 5 cm, $\rho_c$ = 30 … 1 000 Ω·m | +0.231 % (80 m × 20 m, 5 cm of 30 Ω·m) | +1.339 % (150 m × 3 m, 5 cm of 30 Ω·m) |
| bare or shell up to 0.2 m, $\rho_c$ = 30 … 1 000 Ω·m | +0.253 % (6 m × 1.5 m, 0.2 m of 30 Ω·m, 1 000 / 30 Ω·m, $h_1$ = 5 m) | +1.347 % (150 m × 3 m, 0.2 m of 30 Ω·m) |

The largest values sit at the corner of the scope, the longest and
narrowest rectangle (145 m × 2.9 m: +1.335 %, 125 m × 2.5 m: +1.323 %;
150 m × 2.9 m, aspect ratio 51.7 and so outside the scope, +1.350 %, audit
R4B-06);
narrower ones than the scope allows lie higher (150 m × 1 m: +1.65 %,
audit D3B). The independent boundary-element solution of audit D3B gives
the same bare and 5 cm values to 0.001 points, at the scan's segments and
at half of them. For the 0.2 m shell the comparison is uncertain by about
0.2 points, because the outer radius is not thin against the segments: at
150 m × 3 m the two solutions agree (+1.346 / +1.347 %), at 6 m × 1.5 m
the boundary-element value grows from +0.253 % to +0.334 % when its
segments are halved to 0.25 m; for the cases of the table above it lies up
to 0.23 points lower at 0.25 m segments, and refining its segments to
0.083 m — shorter than the radius, which groundfield refuses — moves it by
up to 0.17 points (4 m × 1 m: one owner +0.85 → +1.01 %, one per side
+0.25 → +0.41 %).

## Converting between the conventions

The helpers in `groundfield.geometry.electrodes` convert between the
material description, the coefficient and a lumped resistance:

| Helper | Formula | Use |
|---|---|---|
| `concrete_shell_coefficient(rho_c, wire_radius, thickness)` | $C = \frac{\rho_c}{2\pi}\ln\frac{r_a + t}{r_a}$ | from concrete resistivity and cover |
| `shell_coefficient_from_resistance(electrode, R)` | $C = R\,L$ | from a lumped shell resistance of the whole electrode |
| `shell_resistance_from_coefficient(electrode, C=None)` | $R = C/L$ | the lumped resistance a coefficient represents |

$L$ is `electrode.total_length()`, all wires of the electrode. A lumped
resistance of a site belongs to its encased wires only — every wire of
the foundation, braces included, but not the bonded rods; spreading it
over the rods was the error behind the mixed-site deviations of finding
10.13. With $\rho_c = 100$ Ω·m, 5 cm cover and a
5 mm wire, $C = (100/2\pi)\ln 11 \approx 38.2$ Ω·m, i.e. 3.8 Ω for 10 m of
encased wire.

## Validity

| Condition | Statement |
|---|---|
| Geometry | thin shell around a thin wire; the shell must not overlap a neighbouring wire's shell along its length (at the corners and crossings of a foundation the shells of the meeting wires share a region of the size of $r_b$, which each wire counts as its own) |
| Current | radial through the shell; the series decomposition holds below 1 kHz for foundation sizes |
| Concrete | homogeneous $\rho_c$ per electrode; moisture states span about 30 Ω·m (wet) to tens of kΩ·m (dry), see ADR-0012 |
| Field quantities | the surface potential is that of the bare wire of radius $r_b$ fed through the shell; the shell resistance itself is a series element |
| Backends | `image`, `image_2layer`, `solve_mutual_*`, `mom`, `mom_sommerfeld` and `fem` (since 2026.11) |

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
