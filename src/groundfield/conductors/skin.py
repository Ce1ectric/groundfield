r"""Internal impedance of a solid round conductor (skin effect).

The longitudinal current in a round conductor of radius $a$ and
resistivity $\rho$ crowds towards the surface as the frequency rises.
The internal impedance per unit length follows from the Bessel solution
of the diffusion equation inside the wire (Schelkunoff 1934),

$$
Z'_\text{int}(\omega) \;=\; \frac{k\,\rho}{2\pi a}\,
\frac{I_0(k a)}{I_1(k a)},
\qquad k = \sqrt{j\omega\mu_0/\rho},
$$

with the modified Bessel functions $I_0$, $I_1$ (equivalently the
Kelvin functions ber/bei of $\sqrt{2}\,a/\delta$). Its limits are the
DC resistance $\rho/(\pi a^2)$ plus the uniform-current internal
inductance $\mu_0/(8\pi)$ — exactly the "DC" convention of
:func:`groundfield.coupling.inductance.thin_wire_self_inductance`
(``include_internal=True``) — and, for $a \gg \delta$, the surface
impedance $(1 + j)\,\rho/(2\pi a\delta)$.

Examples (review pass 10, finding 10.37): 150 mm² Al has
$R_\text{ac}/R_\text{dc} = 1.23$ at 550 Hz and $1.55$ at 1 kHz, and its
internal inductance drops to $0.74\times$ the DC value at 1 kHz.

References
----------
- Schelkunoff, S. A. (1934). The electromagnetic theory of coaxial
  transmission lines and cylindrical shields. *Bell Syst. Tech. J.*
  **13**(4), 532–579.
- IEC 60287-1-1:2023, §5.1.2 (skin-effect factor of round conductors).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["MU_0", "solid_round_internal_impedance"]

MU_0 = 4.0e-7 * math.pi

# Below |k a| = 0.02 the Bessel ratio is evaluated from its series,
# I0(z)/I1(z) = (2/z)(1 + z^2/8 - z^4/192 + z^6/3072 + ...), which is
# exact to ~1e-16 there and avoids the cancellation that hides the
# O(z^2) internal reactance in the direct ratio.
_SERIES_LIMIT = 0.02


def solid_round_internal_impedance(
    frequency: float | np.ndarray,
    *,
    radius: float,
    resistivity: float,
) -> complex | np.ndarray:
    r"""Internal series impedance per unit length of a solid round wire.

    Parameters
    ----------
    frequency
        Frequency in Hz (scalar or array, ``>= 0``).
    radius
        Conductor radius $a$ in m (the radius of a solid conductor with
        the same conducting cross section, $a = \sqrt{A/\pi}$).
    resistivity
        Conductor resistivity $\rho$ in Ω·m.

    Returns
    -------
    complex or np.ndarray
        $Z'_\text{int} = R'(f) + j\omega L'_\text{int}(f)$ in Ω/m;
        at ``frequency == 0`` the DC resistance $\rho/(\pi a^2)$.
    """
    if not (radius > 0.0 and math.isfinite(radius)):
        raise ValueError(f"radius must be positive and finite, got {radius}")
    if not (resistivity > 0.0 and math.isfinite(resistivity)):
        raise ValueError(f"resistivity must be positive and finite, got {resistivity}")
    f = np.asarray(frequency, dtype=float)
    if np.any(f < 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("frequency must be finite and >= 0")
    r_dc = resistivity / (math.pi * radius * radius)
    omega = 2.0 * math.pi * f
    k = np.sqrt(1j * omega * MU_0 / resistivity)
    z = k * radius
    out = np.full(f.shape, r_dc, dtype=complex)
    small = np.abs(z) < _SERIES_LIMIT
    if np.any(small):
        z2 = z[small] ** 2
        # Z' = r_dc * (k a / 2) * I0/I1 = r_dc * (1 + z^2/8 - z^4/192 + z^6/3072)
        out[small] = r_dc * (1.0 + z2 / 8.0 - z2 * z2 / 192.0 + z2**3 / 3072.0)
    big = ~small
    if np.any(big):
        # Imported here: ``import groundfield`` must not load scipy.special.
        from scipy.special import ive

        zb = z[big]
        out[big] = (
            k[big] * resistivity / (2.0 * math.pi * radius) * ive(0, zb) / ive(1, zb)
        )
    if np.ndim(frequency) == 0:
        return complex(out[()])
    return out
