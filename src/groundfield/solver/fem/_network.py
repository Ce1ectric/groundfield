"""Network solve of ``fem`` on the shared pipeline (private).

The finite-element model supplies the earth part as electrode ports
(:class:`~groundfield.solver.fem._ports.EarthPorts`); everything else is the
pipeline of the integral backends (:mod:`groundfield.solver._pipeline`):

1. :func:`~groundfield.solver._pipeline.inject` validates the world and
   sums the impressed currents per node (``return_to`` honoured);
   :func:`~groundfield.solver._pipeline.impressed_voltages` lists the
   voltage sources;
2. :func:`~groundfield.solver._pipeline.build_topology` builds clusters,
   lumped branches, junctions and the chains of isolated distributed
   conductors on a discretisation with **one pseudo-segment per port**;
3. :func:`~groundfield.solver._pipeline.assemble_branches` assembles the
   frequency-dependent branch block (inductance, Carson / Sommerfeld
   earth return, skin effect);
4. :func:`~groundfield.solver._pipeline.solve` solves the augmented nodal
   system per frequency with the multi-port matrix of the ports,
   $Z = R = G^{-1}$ (the reaction "kernel" of the port pseudo-segments);
   the segment potentials it returns are the port potentials.

The result carries the electrodes' potentials and currents, point
sources with the model's leakage distribution (display segments of the
engine's ``segment_length``) and the pipeline's metadata (junction
potentials, voltage-source currents).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from groundfield.solver._discretize import _Segment
from groundfield.solver._pipeline import (
    Discretisation,
    SegmentArrays,
    _junction_potentials,
    assemble_branches,
    build_topology,
    impressed_voltages,
    inject,
    solve,
    solved_source_records,
)
from groundfield.solver.result import FieldResult, PointSource

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.fem._ports import EarthPorts
    from groundfield.world import World

__all__: list[str] = []


class _PortKernel:
    """Reaction "kernel" of the port pseudo-segments: $\\varphi = R\\,I$.

    The pipeline passes the segment currents (one pseudo-segment of unit
    length per port, shape ``(P,)`` or ``(P, k)``) and reads the segment
    potentials; for ports these are the port potentials.
    """

    #: The pipeline may apply the operator again for the potentials.
    materialised = True

    def __init__(self, resistance: np.ndarray) -> None:
        self.resistance = resistance

    def __call__(self, points, lengths, radii, currents):  # noqa: ANN001, ANN204
        del points, lengths, radii
        return self.resistance @ currents


def solve_network(
    world: "World",
    engine: "Engine",
    earth: "EarthPorts",
    *,
    metadata: dict,
    kernel: str,
) -> FieldResult:
    """Solve ``world``'s network on the port model ``earth``.

    Parameters
    ----------
    world, engine
        The world (in the frame of ``earth``) and the engine.
    earth
        Port model of the soil.
    metadata
        Backend metadata of the result (``earth.metadata`` is merged in).
    kernel
        Source kernel of the point sources for the post-solve field
        evaluation (``"line"`` or ``"point"``).
    """
    node_currents = inject(world, where="fem", stacklevel=4)
    voltages = impressed_voltages(world)
    names = list(earth.names)
    connection = [np.asarray(e.connection_point, dtype=float) for e in world.electrodes]
    segments = [
        _Segment(
            midpoint=connection[k],
            length=1.0,
            electrode_name=name,
            wire_radius=float(world.electrodes[k].wire_radius),
        )
        for k, name in enumerate(names)
    ]
    disc = Discretisation(
        segments=segments, owner_segments={n: [k] for k, n in enumerate(names)}
    )
    topo = build_topology(world, disc)
    if len(disc.segments) != len(names):  # pragma: no cover - refused before
        raise AssertionError("fem: a galvanic conductor reached the network solve")
    branches = assemble_branches(world, engine, topo)
    n_ports = len(names)
    arrays = SegmentArrays(
        points=np.array(connection, dtype=float).reshape(n_ports, 3),
        lengths=np.ones(n_ports),
        radii=np.array([s.wire_radius for s in segments]),
        shell_coefficients=np.zeros(n_ports),
    )
    solution = solve(
        world,
        engine,
        disc=disc,
        node_currents=node_currents,
        topo=topo,
        branches=branches,
        arrays=arrays,
        kernel=_PortKernel(earth.resistance),
        voltages=voltages,
    )
    n_freq = len(engine.frequencies)
    potentials = {
        name: [complex(solution.segment_potentials[f][k]) for f in range(n_freq)]
        for k, name in enumerate(names)
    }
    currents = {
        name: [complex(solution.owner_currents[f].get(name, 0j)) for f in range(n_freq)]
        for name in names
    }
    per_freq = []
    for f in range(n_freq):
        u = np.array([potentials[n][f] for n in names], dtype=complex)
        i = np.array([currents[n][f] for n in names], dtype=complex)
        per_freq.append(earth.segment_currents(u, i))
    point_sources = [
        PointSource(
            position=tuple(float(v) for v in earth.segment_points[m]),
            current=[complex(per_freq[f][m]) for f in range(n_freq)],
            electrode_name=names[int(earth.segment_owner[m])],
            length=float(earth.segment_lengths[m]),
            direction=tuple(float(v) for v in earth.segment_directions[m]),
        )
        for m in range(earth.segment_lengths.size)
    ]
    clusters = {
        name: sorted(n for n in names if topo.cluster_id[n] == topo.cluster_id[name])
        for name in names
    }
    md = dict(metadata)
    md.update(earth.metadata)
    md["kernel"] = kernel
    md["n_segments"] = int(earth.segment_lengths.size)
    md["earth_inductive_model"] = branches.earth_inductive_model
    if branches.has_inductance:
        from groundfield.coupling.carson import skin_depth

        # The reference resistivity of the integral backends: rho_1.
        sigma = (
            branches.sigma_earth
            if branches.sigma_earth is not None
            else 1.0 / float(md["rhos"][0])
        )
        md["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma) for f in engine.frequencies
        }
    if world.junctions:
        md["junction_potentials"] = _junction_potentials(
            world, disc, arrays, solution, topo.cluster_id, n_freq
        )
    return FieldResult(
        backend="fem",
        frequencies=list(engine.frequencies),
        electrode_potentials=potentials,
        electrode_currents=currents,
        point_sources=point_sources,
        soil_resistivity=float(md["rhos"][0]),
        soil=world.soil,
        clusters=clusters,
        metadata=md,
        sources=solved_source_records(world, n_freq, solution),
    )
