"""The earth part of a ``fem`` solve: electrode ports (private).

:class:`EarthPorts` is the interface between a finite-element model of the
soil and the shared network solve (:mod:`groundfield.solver.fem._network`).
A model — the axisymmetric one of 2026.11
(:mod:`groundfield.solver.fem._axisymmetric`) or a three-dimensional one —
delivers

* the **port conductance matrix** $G$ between the electrodes, each
  electrode one equipotential port (the metal of an encased electrode;
  its shell is part of the model): the currents $\\mathbf I = G\\mathbf U$
  the electrodes leak into the soil for the port potentials $\\mathbf U$;
* the **leakage per display segment** for 1 V on each port, so that the
  result carries point sources with the current distribution of the
  model (:meth:`FieldResult.potential
  <groundfield.solver.result.FieldResult.potential>`, touch voltages);
* the model's metadata (mesh sizes, error estimate).

The network solve adds conductors (lumped, isolated distributed chains,
inductive branches with their earth-return models), junctions and current
and voltage sources exactly as for the integral backends, through
:func:`groundfield.solver._pipeline.solve` with the port matrix
$R = G^{-1}$ as the multi-port matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__: list[str] = []


@dataclass(frozen=True)
class EarthPorts:
    """Port model of the soil between the electrodes of one world.

    Attributes
    ----------
    names
        Electrode names in port order (the order of ``world.electrodes``).
    conductance
        Port conductance matrix $G$ in S, shape ``(P, P)``, symmetric
        positive definite.
    segment_points
        Midpoints in m of the display segments, shape ``(M, 3)``, in the
        frame of the world that was solved.
    segment_lengths
        Lengths in m, shape ``(M,)``.
    segment_directions
        Unit tangents, shape ``(M, 3)``.
    segment_owner
        Port index of every display segment, shape ``(M,)``.
    segment_leakage
        Current in A that leaves the soil-side surface of each display
        segment for 1 V on port ``j`` and 0 V on the other ports, shape
        ``(M, P)``; the rows of one port sum to its row of
        ``conductance`` (up to the extrapolation, see
        :meth:`segment_currents`).
    metadata
        Model-specific metadata, merged into the result's metadata.
    """

    names: tuple[str, ...]
    conductance: np.ndarray
    segment_points: np.ndarray
    segment_lengths: np.ndarray
    segment_directions: np.ndarray
    segment_owner: np.ndarray
    segment_leakage: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def resistance(self) -> np.ndarray:
        """Port resistance matrix $R = G^{-1}$ in Ω, shape ``(P, P)``."""
        r = np.linalg.inv(self.conductance)
        return 0.5 * (r + r.T)

    def segment_currents(
        self, potentials: np.ndarray, currents: np.ndarray
    ) -> np.ndarray:
        """Display-segment currents in A for port potentials and currents.

        The distribution is that of the model, ``segment_leakage @ U``; the
        small difference between its sum over a port and the port current
        of the network solve (the conductance matrix may be extrapolated
        beyond the mesh the distribution comes from) is spread over the
        port's segments in proportion to their lengths, so the segments of
        every port carry exactly the port's current.

        Parameters
        ----------
        potentials
            Port potentials in V, shape ``(P,)`` (complex).
        currents
            Port currents in A, shape ``(P,)`` (complex).
        """
        sc = self.segment_leakage @ np.asarray(potentials, dtype=complex)
        owner = self.segment_owner
        lengths = self.segment_lengths
        for k in range(len(self.names)):
            mask = owner == k
            if not mask.any():
                continue
            residual = complex(currents[k]) - complex(sc[mask].sum())
            sc[mask] += residual * lengths[mask] / lengths[mask].sum()
        return sc
