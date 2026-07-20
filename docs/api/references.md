# Analytical reference formulas

The :mod:`groundfield.references` subpackage contains analytical
reference formulas used in plausibility tests and as closed-form
sanity checks against the numerical backends.

## Dwight 1936 — straight rod in homogeneous soil

[`groundfield.references.dwight1936`](dwight1936) implements the
classical formula for the spreading resistance of a single vertical
rod electrode in homogeneous soil:

$$
R = \frac{\rho}{2\pi L}\,\left[\ln\!\frac{4 L}{a} - 1\right]
$$

with $\rho$ the soil resistivity, $L$ the rod length and $a$ the
rod radius. The formula is exercised by
``tests/test_dwight_references.py`` against the image-method backend.

## Carson 1926 — overhead-line earth-return impedance

[`groundfield.references.carson`](carson) implements Carson's
series expansion for the earth-return self/mutual impedance of an
overhead conductor over a homogeneous lossy half-space.

## Oeding — inductive coupling between buried conductors

[`groundfield.references.oeding`](oeding) implements the
closed-form expressions for the mutual inductance between buried
conductors in homogeneous soil from Oeding & Oswald,
*Elektrische Kraftwerke und Netze*.

## IEEE Std 80 — substation grounding-grid resistance

`groundfield.references.ieee80` implements **Sverak's equation**
(IEEE Std 80-2000 Eq. 57 / 2013 Eq. 53) for the grounding resistance of
a horizontal grid,

$$
R_g = \rho\left[\frac{1}{L_T}
      + \frac{1}{\sqrt{20\,A}}
        \left(1 + \frac{1}{1 + h\sqrt{20/A}}\right)\right],
$$

with total conductor length $L_T$, grid area $A$ and burial depth $h$.

Beyond the closed-form resistance, ``tests/test_ieee80_benchmark.py``
reproduces the standard's **Annex H benchmark** (IEEE Std 80-2013) for
all three benchmark grids: the uniform-soil Grid 1 (bare grid, Table H.5)
and Grid 2 (grid + twenty 7.5 m rods, Table H.6), and the two-layer
Grid 3 (Grid 2 geometry in a soil with $\rho_1 = 300$ Ω·m,
$\rho_2 = 100$ Ω·m and interface $h = 6.096$ m, so the rods cross the
layer boundary; Table H.7). Solved with the ``mom`` backend — which
recovers the current distribution as the reference programs
CDEGS / ETAP / WinIGS do — ``groundfield`` matches the published grid
resistance, GPR and touch voltage to about 1 % on every grid and the step
voltage to a few percent. The worked
``notebooks/43_ieee80_benchmark.ipynb`` lays the results side by side for
all three grids, tabulates the relative error and plots the
grid-resistance comparison and the touch-voltage surface.

## Carson — earth-return line impedance

`groundfield.references.earth_return` implements **Carson's** closed-form
series impedance per unit length of a conductor with earth return — the
engineering-level ($\Omega/\text{km}$) companion to the dimensionless
Carson integral in `groundfield.references.carson`:

$$
Z'_\text{self} = \frac{\omega\mu_0}{8}
    + j\,\frac{\omega\mu_0}{2\pi}\,\ln\frac{D_e}{\mathrm{GMR}},
\qquad
D_e = 658.87\,\sqrt{\rho/f}\ \text{[m]},
$$

with the soil-independent earth-return resistance $\omega\mu_0/8$ and the
equivalent-depth reactance ($D_e$ is the depth of the fictitious return
conductor). The mutual form replaces $\mathrm{GMR}$ by the
conductor–conductor separation.

``tests/test_earth_return_benchmark.py`` validates the assembled
Sommerfeld/Pollaczek earth-return stack
(``earth_inductive_model="sommerfeld"``) against it: the reactance $X'$ —
carrying the frequency- and soil-dependent depth $D_e$ rather than the
naive geometric image at $2h$ — matches Carson to about 1 %, while the
resistance $R'$ approaches $\omega\mu_0/8$ from below as the modelled line
lengthens (the return current closes over roughly one skin depth). The
worked ``notebooks/45_earth_return_line_impedance.ipynb`` plots $Z'(f)$
for the self and mutual terms and the $R'$ convergence, including the
normal-soil power-frequency corner (100 Ω·m, 50 Hz). A **two-layer**
section validates the layered earth-return reactance against the rigorous
closed form of Tsiamitros *et al.* (IEEE Trans. PWRD **20**(3), 2005,
Eq. 8) across an explicit **50 Hz – 1 MHz** frequency sweep — matching to a
few percent in $X'$ and to $\approx 0.003$ in the layer attribution, with
the low-frequency ($\le 1$ kHz) regime, where the inductive return is
deep-layer dominated, highlighted for low-frequency analysis. Since 0.13.0
``"sommerfeld"`` is the **default** ``earth_inductive_model``, so a plain
inductive solve already carries this earth-return physics;
``"perfect_mirror"`` (the former default, a fast electrostatic-image
baseline) is now opt-in for a quick frequency-independent estimate.

## Pollaczek/Sunde — layered buried earth-return line impedance

`groundfield.references.pollaczek` implements the rigorous **layered**,
buried-conductor earth-return **self** impedance per unit length — the
two-layer companion to the homogeneous Carson closed form above. Where
Carson gives the equivalent-depth reactance of an overhead conductor, this
gives the exact spectral integral for a conductor buried at depth $h$ in
the upper layer of a two-layer earth ($\rho_1$ over $\rho_2$, interface at
$h_1$):

$$
Z'_\text{self} = j\,\frac{\omega\mu_0}{2\pi}\left[
    \ln\!\frac{2h}{\mathrm{GMR}}
    + 2\int_0^\infty \frac{e^{-2h\lambda}}{\lambda + u_\text{eff}(\lambda)}
      \,d\lambda \right],
$$

with the **effective vertical wavenumber** of the two-layer earth from the
Tagg/Sunde/Wait tanh-recursion (exposed as `layered_effective_wavenumber`):

$$
u_\text{eff}(\lambda) = u_1\,
    \frac{u_2 + u_1\tanh(u_1 h_1)}{u_1 + u_2\tanh(u_1 h_1)},
\qquad
u_i = \sqrt{\lambda^2 + j\omega\mu_0/\rho_i}.
$$

### Mathematical / physical model

The formulation is **quasi-static** (displacement current neglected,
$u_i = \sqrt{\lambda^2 + j\omega\mu_0/\rho_i}$ carries only the conduction
term), valid for the power- to audio-frequency range that grounding
analysis works in. It returns the **earth-return** series impedance only
(external inductance; no internal-conductor $R'$ or internal inductance —
the same convention as `carson_self_impedance`), so a caller adds the
internal conductor term separately. The $\ln(2h/\mathrm{GMR})$ term is the
small-argument form of the perfect-mirror image
$K_0(\gamma\,\mathrm{GMR}) - K_0(2\gamma h)$; the integral is the
finite-conductivity earth-return correction, with the layering entering
through $u_\text{eff}$ alone. Limiting cases: $\rho_1 = \rho_2 \Rightarrow
u_\text{eff} = u_1$ (homogeneous Pollaczek, matching `carson_self_impedance`
in the reactance to $< 0.1\,\%$, the earth-return resistance sitting just
**below** $\omega\mu_0/8$ — Carson's small-argument limit, which the exact
Pollaczek integral recovers only as $f \to 0$, the deficit growing with
frequency); $h_1 \to \infty \Rightarrow$
upper layer only; $h_1 \to 0 \Rightarrow$ lower layer only. In between, the
reactance lies within the $[\text{Carson}(\rho_1), \text{Carson}(\rho_2)]$
bracket and is **deep-layer dominated at low frequency** — a thin resistive
topsoil is inductively "seen through".

Unlike the assembled Sommerfeld/Pollaczek stack
(`earth_inductive_model="sommerfeld"`), which builds a PEEC segment
inductance matrix from which a single per-unit-length impedance cannot be
read back without an ill-defined return-path assumption, this returns one
scalar series impedance $z'(f) = R' + jX'$ that a reduced nodal network of
an earthing system consumes directly, and that serves as an independent
analytic reference for the assembled stack's earth-return reactance.
``tests/test_pollaczek_reference.py`` (50 tests) pins the homogeneous
Carson limit, the resistance-from-below behaviour, both $h_1$ limits, the
Carson bracket across 50 Hz – 1 kHz, the low-frequency deep-layer regime,
the wavenumber recursion and the input guards; a loop-closure test in
``tests/test_earth_return_benchmark.py`` ties the assembled stack to this
reference and to the Tsiamitros *et al.* (2005, Eq. 8) integral (all three
within $\sim 1\,\%$ on the Case V two-layer reactance at 1 kHz). The worked
``notebooks/46_pollaczek_earth_return.ipynb`` plots $z'(f)$ against Carson,
the homogeneous convergence and the two-layer bracket across layer
thickness. References: Pollaczek (1926); Sunde (1949, *Earth Conduction
Effects in Transmission Systems*, §§ 3–4); Wait (1970); Tsiamitros *et al.*
(2005).

### Conductor–earth loop impedance

`loop_impedance_conductor_earth` builds the full **conductor–earth loop**
impedance on top of the earth-return term:

$$
z'_\text{loop}(f) = R'_\text{internal} + Z'_\text{earth-return},
$$

the series impedance of a current that flows out along a buried earthing
conductor — a cable PEN, a cable screen — and returns through the soil, with
the earthing electrodes at both ends idealised as $\approx 0\,\Omega$. The
earth-return term (`pollaczek_self_impedance`) carries the external and,
through the $\mathrm{GMR}$, the internal inductance of the conductor;
`r_internal` is its own IEC 60228 series resistance per unit length. For a
solid round conductor pass $\mathrm{GMR} = 0.7788\,a = a\,e^{-1/4}$ (internal
inductance $\mu_0/8$ included); for a thin tubular / wire screen pass the
screen radius (no internal inductance). With the ideal termination the loop
impedance of a line of length $\ell$ is $z'_\text{loop}\,\ell$; a real
termination earthing resistance $R_E$ adds in series on top. This is the loop
that governs earth-fault current sharing and the reach of an earthing system
tied together through NAYY PENs or NA2XS2Y screens — which the bare
earth-return impedance does not give on its own. Earth-return path only: no
mutual coupling to a parallel conductor and no internal skin-effect model
(pass a skin-corrected `r_internal` near the top of the band). The notebook
works it for both a 150 mm² Al PEN and a 25 mm² Cu screen.

### Coupling between two conductor–earth loops

`pollaczek_mutual_impedance` is the off-diagonal counterpart of the self
impedance — the earth-return **coupling** between two parallel buried
conductor–earth loops:

$$
Z'_\text{M} = j\,\frac{\omega\mu_0}{2\pi}\left[
    \ln\frac{D}{d}
    + 2\int_0^\infty
      \frac{e^{-(h_i+h_j)\lambda}}{\lambda + u_\text{eff}(\lambda)}
      \cos(\lambda x)\,d\lambda\right],
$$

with the horizontal separation $x$, the direct distance
$d=\sqrt{x^2+(h_i-h_j)^2}$ and the mirror distance
$D=\sqrt{x^2+(h_i+h_j)^2}$; the $\ln(D/d)$ term is the perfect-mirror image
pair (small-argument $K_0(\gamma d)-K_0(\gamma D)$) and the integral the
finite-conductivity correction, the layering again entering only through
$u_\text{eff}$. Unlike the self, it carries **no internal-conductor term** —
two galvanically separate loops couple purely inductively through the soil —
so with the ideal termination the coupling of a parallel run of length
$\ell$ is $Z'_\text{M}\,\ell$, and together with two self impedances it forms
the $2\times2$ loop impedance matrix of the coupled pair. It reduces to
`carson_mutual_impedance` (and to the exact homogeneous Pollaczek mutual
$K_0$ form) when $\rho_1=\rho_2$, its magnitude lies below the self and falls
off logarithmically with separation, and its two-layer reactance stays within
the Carson bracket, deep-layer dominated at low frequency. This is exactly
the coupling term an independent-loop reduction of a networked earthing
system drops; the notebook works it as a separation sweep and a $2\times2$
matrix for two parallel PENs.

## API reference

::: groundfield.references
