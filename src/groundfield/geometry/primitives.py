"""Geometric primitives for grounding systems.

Each primitive provides a factory function that returns a list of
``Segment`` objects. Complex earth electrodes (foundation electrode +
ring conductor + driven rod) are then assembled as the union of
several primitives.
"""

from __future__ import annotations

__all__: list[str] = []
