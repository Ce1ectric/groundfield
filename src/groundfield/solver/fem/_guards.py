"""Refusals and the remedy text of the ``fem`` backend (private).

* :func:`reject_electrodes_out_of_the_soil` refuses what every backend
  refuses (audit D2A-06, R2A-07).
* :func:`remedy` names the backends whose own checks accept a world that
  ``fem`` refuses (audit D2A-06, R2A-03, R2A-04): every refusal of
  ``fem`` takes its remedy from here.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from groundfield.geometry.electrodes import RodElectrode
from groundfield.solver._layered import as_layer_stack

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__: list[str] = []


def reject_electrodes_out_of_the_soil(world: "World") -> None:
    """Refuse leakage paths at or above the soil surface, like every backend.

    The integral backends refuse a leakage segment whose mid-point lies at
    or above the surface, in the surface plane (``2 z <= wire_radius``: the
    mirror image lies inside the wire) or with a part in the air
    (:func:`groundfield.solver._guards._check_segment_depths`). The
    equivalent-hemisphere reduction of ``fem`` (up to 2026.11) solved such
    worlds: its formulas clamped the depth at 1 mm, so a ring 0.5 m in the
    air read like a ring 1 mm deep (audit D2A-06). The conforming model
    refuses them as well, with the same three conditions, on the
    electrodes (every one, driven or
    passive, as the integral backends discretise every one): a rod whose
    head lies above the surface (to round-off — a head at ``z = 0`` is the
    ordinary driven rod), any other electrode — horizontal, its end
    depths equal to the 1e-9 m the geometry model accepts — with its
    shallowest point at ``z <= 0`` or with ``2 z <= wire_radius``. Only
    :class:`RodElectrode` is non-horizontal: a strip whose end depths
    differ by round-off is a horizontal wire, not a rod (audit R2A-01).
    The leakage segments of a galvanic distributed conductor are checked
    as the integral backends discretise them (audit R2A-07: a bare
    conductor in the air was solved, ``fem`` treating it as a lumped
    branch); an isolated conductor has no leakage segment and may run
    anywhere.

    Raises
    ------
    ValueError
        For the first such electrode or conductor (no other backend solves
        the world either, so not a :class:`BackendCapabilityError`).
    """
    from groundfield.solver._discretize import _discretize_conductor
    from groundfield.solver._guards import _SURFACE_TOL

    remedy = (
        "Bury it (e.g. at z = 0.5 ... 0.8 m), or start a rod at the "
        "surface (z = 0), which is the ordinary driven rod."
    )
    for e in world.electrodes:
        z_top = float(e.bounds()[0][2])  # the shallowest point
        a = float(e.wire_radius)
        if isinstance(e, RodElectrode):  # its head is its shallowest point
            if z_top >= -_SURFACE_TOL:
                continue
            what = f"reaches above the soil surface (its head is at z = {z_top:.4g} m)"
        elif z_top <= 0.0:
            what = f"lies at or above the soil surface (z = {z_top:.4g} m)"
        elif 2.0 * z_top <= a:
            what = (
                f"lies in the soil-surface plane (z = {z_top:.4g} m <= "
                f"wire_radius/2 = {0.5 * a:.4g} m), partly in the air"
            )
        else:
            continue
        raise ValueError(
            f"Backend 'fem': electrode '{e.name}' {what}. Every backend refuses "
            f"an electrode out of the soil: z is a depth, positive downwards. {remedy}"
        )
    for c in world.conductors:
        if not getattr(c, "is_distributed", False):
            continue
        if getattr(c, "coupling_to_soil", "isolated") != "galvanic":
            continue
        segments, _ = _discretize_conductor(c)
        if not segments:
            continue
        z = np.array([float(seg.midpoint[2]) for seg in segments])
        a = float(c.wire_radius)
        k = int(np.argmin(z))
        z_ends = (float(c.start[2]), float(c.end[2]))
        if z[k] <= 0.0:
            what = (
                f"has a leakage segment at or above the soil surface (z = {z[k]:.4g} m)"
            )
        elif 2.0 * z[k] <= a:
            what = (
                f"has a leakage segment in the soil-surface plane (z = {z[k]:.4g} m "
                f"<= wire_radius/2 = {0.5 * a:.4g} m), partly in the air"
            )
        elif min(z_ends) < -_SURFACE_TOL:
            what = f"reaches above the soil surface (an end at z = {min(z_ends):.4g} m)"
        else:
            continue
        raise ValueError(
            f"Backend 'fem': conductor '{c.name}' (coupling_to_soil='galvanic', "
            f"distributed) {what}. Every backend refuses a leaking conductor out "
            "of the soil: z is a depth, positive downwards. Bury it, or make it "
            "coupling_to_soil='isolated' if it does not leak into the soil."
        )


def _first_clause(text: str, backend: str) -> str:
    """The first sentence of a backend's error message, without its prefix."""
    if "cannot solve this world: " in text:
        text = text.split("cannot solve this world: ", 1)[1]
    for prefix in (f"{backend}: ", f"solve_{backend}: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = re.split(r"\.\s+(?=[A-Z])", text, maxsplit=1)[0].rstrip(".")
    return text if len(text) <= 240 else text[:237] + "..."


class _Refusal(NamedTuple):
    """Why a candidate backend refuses a world (:func:`_backend_refusal`)."""

    #: ``"world"`` (no engine setting lifts it), ``"limit"`` (the engine's
    #: resource limit, ``Engine.max_segments``; the later checks were not
    #: run) or ``"segment_length"`` (the point form at this segment length).
    kind: str
    #: The backend's own reason: the first clause of its message.
    reason: str
    #: For ``"segment_length"``: the segment length in m below which the
    #: point form accepts every horizontal segment (the hint of its guard).
    finer: float = math.nan


def _below_the_top_layer(world: "World", h_1: float) -> tuple[str, float] | None:
    """The deepest leakage path that ``mom_sommerfeld`` finds below ``h_1``.

    ``mom_sommerfeld`` refuses three or more layers when a segment midpoint
    lies at or below the first interface. Decided here from the geometry,
    independent of the engine's segment length and resource limit (audit
    R2A-04): a rod reaching below ``h_1`` (it is cut there, so a tip on the
    interface keeps every midpoint above it), any other electrode at or
    below ``h_1``, a midpoint of a galvanic distributed conductor at or
    below it. Returns the owner and its depth in m, or ``None``.
    """
    from groundfield.solver._discretize import _discretize_conductor

    found: list[tuple[float, str]] = []
    for e in world.electrodes:
        z_deep = float(e.bounds()[1][2])
        if (z_deep > h_1) if isinstance(e, RodElectrode) else (z_deep >= h_1):
            found.append((z_deep, f"electrode '{e.name}'"))
    for c in world.conductors:
        if not getattr(c, "is_distributed", False):
            continue
        if getattr(c, "coupling_to_soil", "isolated") != "galvanic":
            continue
        segments, _ = _discretize_conductor(c)
        z_deep = max((float(seg.midpoint[2]) for seg in segments), default=-math.inf)
        if z_deep >= h_1:
            found.append((z_deep, f"conductor '{c.name}'"))
    if not found:
        return None
    z_deep, owner = max(found)
    return owner, z_deep


def _backend_refusal(world: "World", engine: "Engine", backend: str) -> _Refusal | None:
    """Why ``backend`` would refuse ``world``, or ``None`` if it accepts it.

    Runs the backend's own checks without solving (audit D2A-06): the
    capability check of the registry, the cross-layer refusal of
    ``mom_sommerfeld`` for three or more layers (from the geometry, before
    anything that depends on the engine's settings, audit R2A-04), the
    discretisation with its resource limit, the burial-depth guard of every
    segment (galvanic conductor segments included) and the point form of
    the reaction matrix at the first interface (``mom``, ``mom_sommerfeld``,
    ``image_2layer`` with ``kernel='point'``). It mirrors ``prepare`` of
    those backends (``solver/mom.py``, ``solver/mom_sommerfeld.py``,
    ``solver/image_2layer.py``) and resolves the kernel like
    :meth:`Engine.solve <groundfield.solver.engine.Engine.solve>`.

    A refusal by the resource limit is a setting of the engine (kind
    ``"limit"``; the later checks are not run). A refusal by the point form
    names, as its guard does, the segment length below which every
    horizontal segment passes (kind ``"segment_length"``); whether that
    length lifts the refusal is checked by :func:`remedy`. The reason is
    the backend's own (the first clause of its message).
    """
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver._guards import (
        CROSS_LAYER_DIAGONAL_SPREAD,
        CROSS_LAYER_WARN_DISTANCE,
        _check_point_interface_diagonal,
        _check_segment_depths,
    )
    from groundfield.solver._kernels import DEFAULT_KERNEL
    from groundfield.solver._pipeline import (
        SegmentArrays,
        build_topology,
        discretise,
    )
    from groundfield.solver.registry import (
        check_capabilities,
        effective_kernel,
        resolve_backend,
    )

    stack = as_layer_stack(world.soil)
    h_1 = float(stack.h[0]) if stack.n_layers >= 2 else None
    requested = engine.kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            resolved = resolve_backend(backend, world)
            check_capabilities(
                resolved, world, kernel=requested, kernel_explicit=requested is not None
            )
            kernel = effective_kernel(
                resolved.spec, requested if requested is not None else DEFAULT_KERNEL
            )
        except (ValueError, NotImplementedError) as exc:
            return _Refusal("world", _first_clause(str(exc), backend))
        if backend == "mom_sommerfeld" and h_1 is not None and stack.n_layers >= 3:
            below = _below_the_top_layer(world, h_1)
            if below is not None:
                return _Refusal(
                    "world",
                    f"{below[0]} reaches z = {below[1]:.3g} m, at or below the first "
                    f"interface at h_1 = {h_1:.3g} m, and it solves three or more "
                    "layers only with every segment in the top layer",
                )
        try:
            disc = discretise(world, engine, None if h_1 is None else (h_1,))
            build_topology(world, disc)  # adds the galvanic conductor segments
            _check_segment_depths(disc.segments, where=backend, kernel=kernel)
        except ResourceLimitError as exc:
            return _Refusal("limit", _first_clause(str(exc), backend))
        except (ValueError, NotImplementedError) as exc:
            return _Refusal("world", _first_clause(str(exc), backend))
        if h_1 is None or (backend == "image_2layer" and kernel == "line"):
            return None
        arrays = SegmentArrays.from_segments(disc.segments)
        z = arrays.points[:, 2]
        cross_layer = bool(z.max() >= h_1)
        try:
            _check_point_interface_diagonal(
                disc.segments,
                arrays.points,
                arrays.lengths,
                arrays.directions,
                h_1,
                backend=f"'{backend}'",
                remedy="",
                cross_layer=cross_layer,
            )
        except (ValueError, NotImplementedError) as exc:
            horizontal = (
                np.ones(z.shape, dtype=bool)
                if arrays.directions is None
                else np.abs(arrays.directions[:, 2]) < 1.0 - 1e-9
            )
            finer = np.abs(z - h_1) / CROSS_LAYER_WARN_DISTANCE
            if cross_layer:
                finer = np.minimum(finer, z / CROSS_LAYER_DIAGONAL_SPREAD)
            finer = finer[horizontal]
            return _Refusal(
                "segment_length",
                _first_clause(str(exc), backend),
                float(finer.min()) if finer.size else math.nan,
            )
    return None


def remedy(world: "World", engine: "Engine") -> str:
    """The backend(s) that solve ``world`` per their own checks.

    Names the candidates of the soil (``image`` and ``mom`` for homogeneous
    soil, ``mom`` and ``image_2layer`` for two layers, ``mom_sommerfeld``
    for three or more) that accept the world at this engine's settings, or
    that a setting lifts (audit D2A-06, R2A-04): the resource limit (raise
    ``Engine.max_segments``) and the point form at the interface (a finer
    segment length, checked by running the backend's checks again at it).
    Backends that refuse the world itself are named with their own reason;
    with none left the text says that no backend solves the world yet.
    Every error message of fem takes its remedy from here (audit R2A-03).
    """
    n_layers = as_layer_stack(world.soil).n_layers
    candidates = {1: ("image", "mom"), 2: ("mom", "image_2layer")}.get(
        n_layers, ("mom_sommerfeld",)
    )
    refusals = {b: _backend_refusal(world, engine, b) for b in candidates}
    # A finer segment length lifts a point-form refusal only if the backend
    # accepts the world at it (or stops at the resource limit there) and it
    # keeps the thin-wire limit (segments of at least ds/2 longer than the
    # wire radius).
    widest = max(float(e.wire_radius) for e in world.electrodes)
    needs_limit: set[str] = set()
    for b, r in refusals.items():
        if r is None or r.kind != "segment_length":
            continue
        ds = 0.99 * r.finer
        if math.isfinite(ds) and ds > 2.0 * widest:
            again = _backend_refusal(
                world, engine.model_copy(update={"segment_length": ds}), b
            )
            if again is None or again.kind == "limit":
                if again is not None:
                    needs_limit.add(b)
                continue
        refusals[b] = _Refusal("world", r.reason)
    named = [
        b for b in candidates if refusals[b] is None or refusals[b].kind != "world"
    ]
    limited = {
        b: refusals[b].reason
        for b in named
        if refusals[b] is not None and refusals[b].kind == "limit"
    }
    finer = {
        b: refusals[b]
        for b in named
        if refusals[b] is not None and refusals[b].kind == "segment_length"
    }
    by_reason: dict[str, list[str]] = {}
    for b in candidates:
        if refusals[b] is not None and refusals[b].kind == "world":
            by_reason.setdefault(refusals[b].reason, []).append(f"'{b}'")
    refused = " ".join(
        f"{' and '.join(names)} refuse{'s' if len(names) == 1 else ''} this "
        f"world: {reason}."
        for reason, names in by_reason.items()
    )
    if not named:
        return f"No backend solves this world yet. {refused}"
    if named == ["mom_sommerfeld"]:
        text = (
            "Use backend='mom_sommerfeld', which solves three or more layers "
            "when every segment lies in the top layer, as here."
        )
    elif named == ["mom", "image_2layer"]:
        text = "Use backend='mom' (one current per segment) or 'image_2layer'."
    else:
        text = "Use backend=" + " or ".join(f"'{b}'" for b in named) + "."
    if limited:
        names = " and ".join(f"'{b}'" for b in limited)
        text += (
            f" This engine's resource limit refuses the world for {names} "
            f"({next(iter(limited.values()))}): raise Engine.max_segments or "
            "coarsen segment_length (their other checks were not run)."
        )
    for b, r in finer.items():
        text += (
            f" '{b}' refuses this world at segment_length = "
            f"{engine.segment_length:g} m ({r.reason}) but accepts it below "
            f"{r.finer:.3g} m"
        )
        if b in needs_limit:
            text += (
                ", with a larger Engine.max_segments (its other checks were not "
                "run at that length)"
            )
        text += "."
    return f"{text} {refused}" if refused else text
