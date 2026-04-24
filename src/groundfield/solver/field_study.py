"""Oberste Koordinationsklasse ``FieldStudy``.

Eine ``FieldStudy`` bündelt alle Eingangsdaten zu einem Rechenlauf:

* das Bodenmodell (``groundfield.soil``),
* die Geometrie der Erdungsanlage (``groundfield.geometry``),
* optionale Leiter (``groundfield.conductors``),
* die Frequenzliste,
* die Randbedingungen (Einspeisung, Hilfselektrode, Erdschluss).

Nach ``solve()`` stehen die Ergebnisse als ``FieldResult`` bereit und
können über ``groundfield.postprocess`` und ``groundfield.io`` weiter
verarbeitet werden.

Todo
----
* Konstruktor mit Pydantic-Validierung.
* ``solve(backend="mom" | "fem")`` mit Fortschrittsanzeige und Caching.
* Einheitliches Ergebnisobjekt ``FieldResult``.
"""

from __future__ import annotations

__all__: list[str] = []
