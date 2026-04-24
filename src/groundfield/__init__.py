"""groundfield — numerische Feldberechnung von Erdungsanlagen.

``groundfield`` ist ein Open-Source-Python-Paket für die physikalische
Referenzmodellierung vernetzter Erdungssysteme. Es bildet innerhalb der
``groundmeas`` / ``groundinsight`` / ``groundfield`` Softwarefamilie die
feldtheoretische Seite ab: Bodenmodelle, Erdergeometrien, Leiter und deren
Kopplungen werden als 3D-Problem im Erdreich formuliert und numerisch gelöst.
Die Ergebnisse sind Feldverläufe, Potenzialkurven, Stromverteilungen und
reduzierte Ersatzmodelle (``rho-f``-Modell) für die Weiterverarbeitung in
``groundinsight``.

Das Paket ist bewusst so gestaltet, dass es die in Arbeitspaket 1 der
Dissertation zu vernetzten Erdungssystemen geforderten Referenzrechnungen
tragen kann: TN-Ortsnetz mit Ortsnetzstation, Hausanschlüssen, Kabelverteilern
und geschichtetem Erdreich.

Subpackages
-----------
soil
    Bodenmodelle (homogen, Mehrschicht, frequenzabhängig).
geometry
    Erder- und Leitergeometrien, Meshgenerierung.
conductors
    Leiter, PEN, Kabelschirme und ihre Eigen- und Koppelimpedanzen.
solver
    Numerischer Feldlöser (PDE/MoM) im Frequenzbereich.
coupling
    Induktive und galvanische Kopplung zwischen Leitern, Schirmen und Erdreich.
postprocess
    Auswertung: Potenzialverläufe, Berührungsspannungen, Stromdichte.
io
    Datei-I/O und Austauschformate, insbesondere Export reduzierter
    ``rho-f``-Modelle für ``groundinsight``.
utils
    Hilfsfunktionen für Koordinaten, Einheiten, Logging und Validierung.

Examples
--------
>>> import groundfield as gf
>>> soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
>>> geom = gf.HorizontalRingElectrode(radius=5.0, depth=0.8, wire_radius=0.005)
>>> study = gf.FieldStudy(soil=soil, geometry=geom, frequencies=[50.0])
>>> # result = study.solve()  # noqa: E800 — placeholder, implementation follows
"""

from __future__ import annotations

# Re-exports werden mit dem Aufbau der einzelnen Module sukzessive ergänzt.
# Bis dahin bleibt die Top-Level-API minimal, damit der Import auch bei einer
# leeren Skelettstruktur funktioniert und von den Tests geprüft werden kann.

__all__: list[str] = [
    "__version__",
]

# Version
__version__ = "0.1.0"
