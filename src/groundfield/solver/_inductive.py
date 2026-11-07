"""Inductive branch assembly of the conductor network (private).

Partial-inductance matrix (ADR-0004), the earth-return correction
(Carson series, ADR-0005; Sommerfeld, ADR-0006) and the opt-in
frequency-dependent internal impedance of the conductors (skin effect)
over the finite branches of the nodal-analysis system.

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.

Physics and conventions
-----------------------
The longitudinal impedance of the finite branches is

$$
Z_b(\\omega) \\;=\\; R \\;+\\; j\\omega\\,L \\;+\\; \\Delta Z(\\omega),
$$

with $R$ the DC branch resistances (diagonal), $L$ the
partial-inductance matrix of all inductive branches (Neumann double
line integrals, thin-wire self terms) and $\\Delta Z$ the
frequency-dependent correction built here. Review pass 10 fixed the
conventions the three terms are assembled with:

- **Image baseline per earth-return model** (findings 10.20 / F32 and
  10.46). An earth-return correction is only meaningful together with
  the image convention of the inductance matrix it is added to
  (:mod:`groundfield.coupling.inductance`):

  =========================  ==================  =======================
  ``earth_inductive_model``  image of $L$        $\\Delta Z$
  =========================  ==================  =======================
  ``"perfect_mirror"``       ``"additive"``      none (ADR-0004)
  ``"carson_series"``        ``"antiparallel"``  Carson $(\\omega\\mu_0/\\pi)(P + jQ)$
  ``"sommerfeld"``           ``"none"``          $\\Delta G = G - 1/R$ (exact)
  =========================  ==================  =======================

  Carson's correction is defined against the anti-parallel
  (perfect-electric-conductor) image; up to 0.15.0 the solver added it
  on top of the additive image, which gave 1.66–2.25 times the
  earth-return reactance on 100 m – 1 km lines (+41 % $|Z|$ at 1 kHz
  on a TN network). The Sommerfeld correction is built against free
  space: exact (no image to add and remove again; the same numbers as
  the additive pairing to round-off) and valid for conductors that
  cross the soil surface, which the pair integrals split at $z = 0$
  (with an image convention such a segment is singular and raises).
- **Touching segments** (10.46): chain neighbours and bends share an
  end point, where the Neumann integrand is log-singular; the closed
  forms (``touching="exact"``) replace the 16x16 Gauss–Legendre rule
  (-0.3 % ... +0.8 % for such pairs), so a straight conductor keeps the
  inductance of the unsplit wire for every discretisation.
- **Internal inductance**: every self term carries the DC internal
  inductance $\\mu_0\\ell/(8\\pi)$ of a solid round conductor
  (``include_internal=True``), for distributed and lumped branches
  alike (the package-wide convention of
  :mod:`groundfield.coupling.inductance`).
- **Lumped and single-segment conductors** (10.24): a conductor with an
  ``inductance_model`` that the topology keeps as one lumped branch
  (``discretize_segment_length=None``, or one sub-segment) is one
  inductive branch from its start to its end point; up to 0.15.0 its
  inductance was silently dropped.
- **Skin effect** (10.37, opt-in): a conductor with
  ``resistance_model="skin"`` adds
  $\\ell_b\\,\\Delta Z'_\\text{skin}(f)$
  (:meth:`~groundfield.conductors.Conductor.skin_correction_per_length`,
  the exact solid-round internal impedance relative to the DC
  convention) to the diagonal entry of each of its branches of length
  $\\ell_b$: the resistance becomes $R_\\text{ac}(f)$ and the internal
  inductance $L_\\text{int}(f)$. A conductor without an
  ``inductance_model`` has no inductance at all, so only the
  resistance part $\\operatorname{Re}\\Delta Z'_\\text{skin}$ applies to
  it.
- **Several frequencies** (10.60): the correction is built for the whole
  frequency set of the solve in one multi-frequency assembly (geometry,
  $\\lambda$ grid and Bessel tables once per segment pair); every
  frequency is bit-identical to a single-frequency assembly. The opt-in
  ``sommerfeld_method="tabulated"`` shares one table of the spectral
  integral per pair of depth groups instead (faster on large networks,
  agrees with the exact assembly to about $10^{-5}$ of the largest
  entry).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.conductors import Conductor
    from groundfield.solver._discretize import _DistributedBranch

__all__: list[str] = []

#: Image convention of the partial-inductance matrix per earth-return
#: model (review pass 10, findings 10.20 / F32 and 10.46).
IMAGE_OF_EARTH_MODEL: dict[str, str] = {
    "perfect_mirror": "additive",
    "carson_series": "antiparallel",
    "sommerfeld": "none",
}

#: Neumann rule for segments that share an end point or an axis
#: (review pass 10, 10.46): closed forms instead of the 16x16 rule.
SOLVER_TOUCHING = "exact"

#: Memory bound in bytes of the multi-frequency branch correction: matrices
#: of one assembly plus the cached ones (W3, 30 houses: about 2 MB per
#: frequency, so all 11 frequencies of a sweep fit many times over).
_CACHE_BYTES = 256 * 2**20


@dataclass(frozen=True)
class BranchGeometry:
    """Geometry of one finite branch for the inductive assembly.

    The attribute names match
    :class:`~groundfield.solver._discretize._DistributedBranch`, so both
    kinds of branch are assembled by the same code.

    Attributes
    ----------
    p_a, p_b
        End points of the branch in m, shape ``(3,)``; the positive
        branch current flows from ``p_a`` to ``p_b``.
    wire_radius
        Wire radius in m (thin-wire self term).
    inductive
        Whether the branch takes part in the inductance matrix.
    """

    p_a: np.ndarray
    p_b: np.ndarray
    wire_radius: float
    inductive: bool


@dataclass(frozen=True)
class SkinTerm:
    """Skin-effect correction of one branch (``resistance_model="skin"``).

    Attributes
    ----------
    index
        Position of the branch in ``finite_branches``.
    length
        Branch length in m.
    conductor
        Conductor the branch belongs to.
    inductive
        ``True``: the full internal-impedance correction (resistance and
        internal inductance; the inductance matrix carries the DC
        internal inductance). ``False``: the resistance part only (the
        branch has no inductance).
    """

    index: int
    length: float
    conductor: Conductor
    inductive: bool

    def correction(self, frequency: float) -> complex:
        """Correction of the branch impedance in Ω at ``frequency`` (Hz)."""
        dz = complex(self.conductor.skin_correction_per_length(frequency))
        if not self.inductive:
            dz = complex(dz.real, 0.0)
        return self.length * dz


class _BranchCorrection:
    """Frequency-dependent branch correction ``omega -> ΔZ(omega)`` in Ω.

    Earth-return correction of the inductive branches (Carson series or
    Sommerfeld, built against the image convention of the inductance
    matrix) plus the skin-effect correction on the diagonal of
    ``resistance_model="skin"`` branches, over **all** finite branches.

    The matrices are built for the whole frequency set ``omegas`` on the
    first call with one of those frequencies (one multi-frequency
    assembly, review pass 10, 10.60) and cached; any other angular
    frequency is built on demand. The returned matrices are read-only.

    Parameters
    ----------
    n_total
        Number of finite branches.
    index
        Positions of the inductive branches in ``finite_branches``.
    seg_endpoints
        End points of the inductive branches in m, shape ``(M, 2, 3)``.
    radii
        Wire radii of the inductive branches in m, shape ``(M,)``.
    earth_model
        ``"perfect_mirror"``, ``"carson_series"`` or ``"sommerfeld"``.
    sigma_earth
        Earth conductivity in S/m (``carson_series``).
    layered_earth
        :class:`~groundfield.coupling.sommerfeld_inductance.LayeredEarth`
        (``sommerfeld``).
    skin_terms
        Skin-effect corrections of single branches.
    omegas
        Angular frequencies in rad/s of the solve (``None``: build each
        frequency on demand).
    sommerfeld_method
        ``"exact"`` or ``"tabulated"`` assembly of the Sommerfeld
        correction (see
        :func:`~groundfield.coupling.sommerfeld_inductance.build_sommerfeld_correction_matrices`).
    """

    def __init__(
        self,
        n_total: int,
        *,
        index: np.ndarray,
        seg_endpoints: np.ndarray,
        radii: np.ndarray,
        earth_model: str,
        sigma_earth: float | None,
        layered_earth: object,
        skin_terms: Sequence[SkinTerm],
        omegas: Sequence[float] | None,
        sommerfeld_method: str = "exact",
    ) -> None:
        self._n = int(n_total)
        self._index = np.asarray(index, dtype=int)
        self._seg = np.array(seg_endpoints, dtype=float)
        self._radii = np.array(radii, dtype=float)
        self._model = earth_model
        self._image = IMAGE_OF_EARTH_MODEL[earth_model]
        self._sigma = sigma_earth
        self._earth = layered_earth
        self._skin = tuple(skin_terms)
        self._method = sommerfeld_method
        batch: list[float] = []
        for om in omegas or ():
            om = float(om)
            if om > 0.0 and om not in batch:
                batch.append(om)
        self._batch = tuple(batch)
        self._cache: dict[float, np.ndarray] = {}
        self._zero: np.ndarray | None = None
        self._n_built = 0
        # Frequencies per assembly and cached matrices are bounded so that
        # a long frequency sweep over a large network stays within
        # _CACHE_BYTES (the (n_f, M, M) earth matrices of one assembly plus
        # the cached full matrices); evicted frequencies are rebuilt on
        # demand, bit-identically.
        per_frequency = 16 * (self._n**2 + self._seg.shape[0] ** 2)
        self._max_batch = max(1, _CACHE_BYTES // max(per_frequency, 1))

    @property
    def n_assemblies(self) -> int:
        """Number of frequencies assembled so far (diagnostics)."""
        return self._n_built

    def __call__(self, omega: float) -> np.ndarray:
        """Correction matrix over all finite branches at ``omega`` (rad/s)."""
        omega = float(omega)
        if omega <= 0.0:
            if self._zero is None:
                self._zero = np.zeros((self._n, self._n), dtype=complex)
                self._zero.setflags(write=False)
            return self._zero
        out = self._cache.get(omega)
        if out is None:
            if omega in self._batch:
                # this frequency and the next uncached ones of the solve
                start = self._batch.index(omega)
                order = self._batch[start:] + self._batch[:start]
                pending = [om for om in order if om not in self._cache]
                chunk = pending[: self._max_batch]
            else:
                chunk = [omega]
            while self._cache and len(self._cache) + len(chunk) > self._max_batch:
                self._cache.pop(next(iter(self._cache)))  # oldest first
            self._build(chunk)
            out = self._cache[omega]
        return out

    def _earth_matrices(self, omegas: list[float]) -> np.ndarray | None:
        """Earth-return corrections of the inductive branches, (n_f, M, M)."""
        if self._seg.shape[0] == 0 or self._model == "perfect_mirror":
            return None
        if self._model == "carson_series":
            from groundfield.coupling.inductance import (
                build_carson_correction_matrices,
            )

            return build_carson_correction_matrices(
                self._seg,
                self._radii,
                omegas=omegas,
                sigma_earth=float(self._sigma),
                image=self._image,
            )
        from groundfield.coupling.sommerfeld_inductance import (
            build_sommerfeld_correction_matrices,
        )

        return build_sommerfeld_correction_matrices(
            self._seg,
            self._radii,
            omegas=omegas,
            earth=self._earth,
            image=self._image,
            method=self._method,
        )

    def _build(self, omegas: list[float]) -> None:
        earth = self._earth_matrices(omegas)
        ix = np.ix_(self._index, self._index)
        for k, omega in enumerate(omegas):
            full = np.zeros((self._n, self._n), dtype=complex)
            if earth is not None:
                full[ix] = earth[k]
            frequency = omega / (2.0 * math.pi)
            for term in self._skin:
                full[term.index, term.index] += term.correction(frequency)
            full.setflags(write=False)
            self._cache[omega] = full
        self._n_built += len(omegas)


def _assemble_inductance_matrix(
    distributed_branches: Sequence[_DistributedBranch],
    *,
    n_lumped_branches: int,
    n_total_branches: int,
    earth_model: str = "perfect_mirror",
    sigma_earth: float | None = None,
    layered_earth: object = None,
    lumped_branches: Sequence[BranchGeometry | None] | None = None,
    skin_terms: Sequence[SkinTerm] = (),
    omegas: Sequence[float] | None = None,
    sommerfeld_method: str = "exact",
) -> tuple[np.ndarray | None, bool, _BranchCorrection | None]:
    """Build the partial-inductance matrix and the branch correction.

    The output covers *all* finite branches (lumped + distributed) in
    the ordering of the solver's ``finite_branches`` list: lumped
    branches first, distributed branches after. A branch takes part in
    the inductance matrix when its conductor has an
    ``inductance_model``; the rest of the matrix stays zero. The image
    convention of the matrix follows the earth-return model
    (:data:`IMAGE_OF_EARTH_MODEL`, see the module docstring), touching
    segment pairs use the closed forms, and every self term carries the
    DC internal inductance.

    The returned correction evaluates, per angular frequency $\\omega$,

    $$
    Z_b(\\omega) \\;=\\; R \\;+\\; j\\omega\\,L
        \\;+\\; \\Delta Z(\\omega),
    $$

    i.e. $\\Delta Z$ = earth-return correction (Carson series ADR-0005,
    Sommerfeld ADR-0006) + skin-effect corrections; the solver adds it
    to the branch block once per frequency.

    Parameters
    ----------
    distributed_branches
        Branches produced by
        :func:`~groundfield.solver._topology._build_distributed_topology`;
        the ``inductive`` flag selects whether a branch contributes.
    n_lumped_branches
        Number of lumped finite-impedance branches that come before
        the distributed ones in ``finite_branches``.
    n_total_branches
        Total number of finite branches in ``finite_branches``.
    earth_model
        ``"perfect_mirror"`` (ADR-0004, no correction),
        ``"carson_series"`` (ADR-0005) or ``"sommerfeld"`` (ADR-0006).
    sigma_earth
        Earth conductivity in S/m; required for ``"carson_series"``.
    layered_earth
        :class:`~groundfield.coupling.sommerfeld_inductance.LayeredEarth`;
        required for ``"sommerfeld"``.
    lumped_branches
        Geometry of the lumped branches (one entry per lumped branch,
        ``None`` for a branch without known geometry); ``None`` keeps
        every lumped branch purely resistive (callers that do not map
        lumped branches to conductors).
    skin_terms
        Skin-effect corrections (:class:`SkinTerm`).
    omegas
        Angular frequencies in rad/s of the solve; the correction is
        built for all of them at once on first use.
    sommerfeld_method
        ``"exact"`` (default) or ``"tabulated"`` (opt-in) assembly of the
        Sommerfeld correction.

    Returns
    -------
    L : np.ndarray | None
        ``(n_total_branches, n_total_branches)`` partial-inductance
        matrix in H (all zeros when only skin-effect branches are
        present), or ``None`` when no branch has an inductance or a
        frequency-dependent impedance.
    has_inductance : bool
        ``L is not None``: the branch block depends on the frequency.
    correction : _BranchCorrection | None
        ``omega -> ΔZ(omega)`` over all finite branches in Ω, or
        ``None`` (``perfect_mirror`` without skin-effect branches).

    Raises
    ------
    ValueError
        For an unknown ``earth_model``, ``carson_series`` without a
        positive ``sigma_earth``, ``sommerfeld`` without
        ``layered_earth``, a ``lumped_branches`` list of the wrong
        length, or (image conventions ``additive`` / ``antiparallel``)
        a conductor that crosses or lies in the soil surface.
    """
    from groundfield.coupling.inductance import build_inductance_matrix

    if earth_model not in IMAGE_OF_EARTH_MODEL:
        raise ValueError(
            f"earth_model must be one of {tuple(IMAGE_OF_EARTH_MODEL)}, "
            f"got {earth_model!r}"
        )
    records: list[tuple[int, object]] = []
    if lumped_branches is not None:
        if len(lumped_branches) != n_lumped_branches:
            raise ValueError(
                f"lumped_branches has {len(lumped_branches)} entries for "
                f"{n_lumped_branches} lumped branches."
            )
        records += [
            (k, b)
            for k, b in enumerate(lumped_branches)
            if b is not None and b.inductive
        ]
    records += [
        (n_lumped_branches + offset, db)
        for offset, db in enumerate(distributed_branches)
        if db.inductive
    ]
    skin_terms = tuple(skin_terms)
    if not records and not skin_terms:
        return None, False, None

    L_full = np.zeros((n_total_branches, n_total_branches))
    idx = np.array([k for k, _ in records], dtype=int)
    if records:
        seg_endpoints = np.stack(
            [np.stack([b.p_a, b.p_b], axis=0) for _, b in records], axis=0
        )
        radii = np.array([b.wire_radius for _, b in records])
        L_sub = build_inductance_matrix(
            seg_endpoints,
            radii,
            image=IMAGE_OF_EARTH_MODEL[earth_model],
            touching=SOLVER_TOUCHING,
        )
        L_full[np.ix_(idx, idx)] = L_sub
    else:
        seg_endpoints = np.zeros((0, 2, 3))
        radii = np.zeros(0)

    if records and earth_model == "carson_series":
        if sigma_earth is None or sigma_earth <= 0.0:
            raise ValueError(
                "earth_model='carson_series' requires sigma_earth > 0; "
                f"got sigma_earth={sigma_earth!r}"
            )
    elif records and earth_model == "sommerfeld" and layered_earth is None:
        raise ValueError(
            "earth_model='sommerfeld' requires layered_earth "
            "(LayeredEarth instance); got None."
        )

    correction: _BranchCorrection | None = None
    if (records and earth_model != "perfect_mirror") or skin_terms:
        correction = _BranchCorrection(
            n_total_branches,
            index=idx,
            seg_endpoints=seg_endpoints,
            radii=radii,
            earth_model=earth_model,
            sigma_earth=sigma_earth,
            layered_earth=layered_earth,
            skin_terms=skin_terms,
            omegas=omegas,
            sommerfeld_method=sommerfeld_method,
        )
    return L_full, True, correction


# ---------------------------------------------------------------------
# Branches -> conductors
# ---------------------------------------------------------------------


def _is_chain(conductor: Conductor) -> bool:
    """Whether the topology routes ``conductor`` as a distributed chain."""
    return bool(conductor.is_distributed) and int(conductor.n_segments) > 1


def _lumped_branch_conductors(
    conductors: Sequence[Conductor],
    cluster_id: dict[str, str],
    lumped_branches: Sequence[tuple[str, str, float]],
) -> list[Conductor | None]:
    """Conductor behind every lumped finite branch, in branch order.

    The topology step builds one lumped branch ``(root(start),
    root(end), R)`` per finite, non-chain conductor, in world order. The
    same key identifies the conductor here (the first unused one in
    world order for equal keys); a branch without a matching conductor
    gets ``None`` and stays purely resistive.
    """
    candidates: dict[tuple[str, str, float], deque] = defaultdict(deque)
    for c in conductors:
        if _is_chain(c) or c.start_electrode is None or c.end_electrode is None:
            continue
        key = (
            cluster_id.get(c.start_electrode, c.start_electrode),
            cluster_id.get(c.end_electrode, c.end_electrode),
            float(c.series_resistance),
        )
        candidates[key].append(c)
    out: list[Conductor | None] = []
    for node_a, node_b, resistance in lumped_branches:
        queue = candidates.get((node_a, node_b, float(resistance)))
        out.append(queue.popleft() if queue else None)
    return out


def _chain_branch_key(branch) -> tuple[bytes, bytes, float]:
    return (
        np.asarray(branch.p_a, dtype=float).tobytes(),
        np.asarray(branch.p_b, dtype=float).tobytes(),
        float(branch.R),
    )


def _chain_branch_conductors(
    conductors: Sequence[Conductor],
    distributed_branches: Sequence[_DistributedBranch],
    wanted: Sequence[Conductor],
) -> list[Conductor | None]:
    """Conductor (out of ``wanted``) behind every distributed branch.

    The chains of the ``wanted`` conductors are rebuilt with the
    topology's own discretiser and matched by their exact end points and
    resistance; other branches get ``None``.
    """
    from groundfield.solver._discretize import _discretize_conductor

    owner: dict[tuple[bytes, bytes, float], Conductor] = {}
    for c in wanted:
        _, chain = _discretize_conductor(c)
        for branch in chain:
            owner.setdefault(_chain_branch_key(branch), c)
    return [owner.get(_chain_branch_key(db)) for db in distributed_branches]


def branch_records(
    conductors: Sequence[Conductor],
    *,
    cluster_id: dict[str, str],
    finite_branches: Sequence[tuple[str, str, float]],
    distributed_branches: Sequence[_DistributedBranch],
    n_lumped_branches: int,
) -> tuple[list[BranchGeometry | None], list[SkinTerm]]:
    """Lumped-branch geometry and skin-effect terms of a topology.

    Parameters
    ----------
    conductors
        Conductors of the world, in world order.
    cluster_id
        Cluster root per node name (topology step).
    finite_branches
        ``(node_a, node_b, R)`` per finite branch, lumped ones first.
    distributed_branches
        Distributed-conductor branches (topology step).
    n_lumped_branches
        Number of lumped branches at the head of ``finite_branches``.

    Returns
    -------
    lumped : list[BranchGeometry | None]
        One entry per lumped branch: start → end of its conductor,
        inductive when the conductor has an ``inductance_model``
        (finding 10.24).
    skin : list[SkinTerm]
        Skin-effect corrections of all branches of
        ``resistance_model="skin"`` conductors (finding 10.37).

    Warns
    -----
    UserWarning
        If an inductive or skin-effect conductor cannot be matched to the
        branches the topology built for it (its inductance or skin effect
        would otherwise be lost without a trace).
    """
    lumped_conductors = _lumped_branch_conductors(
        conductors, cluster_id, finite_branches[:n_lumped_branches]
    )
    lumped: list[BranchGeometry | None] = []
    skin: list[SkinTerm] = []
    for k, c in enumerate(lumped_conductors):
        if c is None:
            lumped.append(None)
            continue
        p_a = np.array(c.start, dtype=float)
        p_b = np.array(c.end, dtype=float)
        length = float(np.linalg.norm(p_b - p_a))
        inductive = c.inductance_model is not None and length > 0.0
        lumped.append(BranchGeometry(p_a, p_b, float(c.wire_radius), inductive))
        if c.resistance_model == "skin" and length > 0.0:
            skin.append(SkinTerm(k, length, c, inductive))

    chain_skin = [
        c for c in conductors if _is_chain(c) and c.resistance_model == "skin"
    ]
    if chain_skin:
        owners = _chain_branch_conductors(conductors, distributed_branches, chain_skin)
        for offset, (db, c) in enumerate(
            zip(distributed_branches, owners, strict=True)
        ):
            if c is None:
                continue
            length = float(np.linalg.norm(np.asarray(db.p_b) - np.asarray(db.p_a)))
            skin.append(
                SkinTerm(n_lumped_branches + offset, length, c, bool(db.inductive))
            )
        matched = {id(c) for c in owners if c is not None}
        _warn_unmatched([c for c in chain_skin if id(c) not in matched], "skin effect")

    if any(c is None for c in lumped_conductors):
        used = {id(c) for c in lumped_conductors if c is not None}
        _warn_unmatched(
            [
                c
                for c in conductors
                if not _is_chain(c)
                and id(c) not in used
                and (c.inductance_model is not None or c.resistance_model == "skin")
                and not c.is_ideal()
                and c.start_electrode is not None
                and c.end_electrode is not None
                and cluster_id.get(c.start_electrode, c.start_electrode)
                != cluster_id.get(c.end_electrode, c.end_electrode)
            ],
            "inductance / skin effect",
        )
    return lumped, skin


def _warn_unmatched(conductors: Sequence[Conductor], what: str) -> None:
    if not conductors:
        return
    names = ", ".join(repr(c.name) for c in conductors)
    warnings.warn(
        f"Conductor(s) {names} could not be matched to the branches of the "
        f"conductor network; their {what} is not applied (internal "
        "inconsistency between the topology and the inductive assembly — "
        "please report it).",
        UserWarning,
        stacklevel=4,
    )


#: Type of the correction returned by :func:`_assemble_inductance_matrix`
#: (historic name, re-exported by :mod:`groundfield.solver.image`).
_CarsonBuilder = _BranchCorrection
