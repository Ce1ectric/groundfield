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

## API reference

::: groundfield.references
