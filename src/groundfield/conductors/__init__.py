"""Leiter, PEN und Kabelschirme.

Hier werden alle stromführenden Elemente oberhalb und innerhalb des
Erdreichs modelliert, die nicht selbst Erder sind aber dennoch zur
Stromverteilung beitragen:

- Freileitungen (Phase + Erdseil)
- Kabel (Phase + Schirm, konzentrisch oder drei-adrig)
- PEN-Leiter im Niederspannungsnetz
- Hilfselektroden bei Erdungsmessungen

Contents
--------
Line
    Freileitungsabschnitt mit Koordinaten und Leitertyp.
Cable
    Kabelabschnitt mit Phasen- und Schirmstruktur.
PENConductor
    PEN-Leiter im TN-Netz mit zugeordneten Erdungsknoten.
ConductorLibrary
    Bibliothek von Leitertypen (Querschnitt, Material, Frequenzbereich).

Notes
-----
Die Carson-Korrekturen für den Erdrückstrompfad werden hier nicht
ausgerechnet; sie fließen über ``groundfield.coupling`` ein. Dadurch bleibt
das Conductor-Modell rein geometrisch/materialbasiert und lässt sich mit
unterschiedlichen Rückleiter-Modellen kombinieren.
"""

from __future__ import annotations

__all__: list[str] = []
