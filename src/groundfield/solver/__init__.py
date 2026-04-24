"""Numerischer Feldlöser.

Dieses Subpackage bildet den rechnerischen Kern von ``groundfield``. Die
Leitgröße ist das komplexe Potential ``phi(r, f)`` im Erdreich und auf den
Leiterflächen, ausgewertet per Frequenz im Phasor-Bereich. Als
Lösungsansatz ist die Momentenmethode (MoM) mit Bildladungen im
geschichteten Halbraum vorgesehen; ein alternatives FEM-Backend über
``scikit-fem`` oder ``fenics`` kann später angebunden werden.

Contents
--------
FieldStudy
    Obere API: bündelt Bodenmodell, Geometrie, Leiter und Frequenzliste,
    orchestriert den Aufbau der Systemmatrix und den Solve-Schritt.
SystemMatrix
    Assemblierung der Admittanzmatrix / MoM-Matrix pro Frequenz.
Solver
    Dünnbesetzte LU-Zerlegung und iterative Verfahren (GMRES) für große
    Geometrien.
Backend
    Abstraktion für austauschbare numerische Backends (MoM, FEM).

Leitprinzip
-----------
Das PDE- bzw. Feldmodell ist Referenz, nicht Endprodukt. Der Löser muss
so instrumentiert sein, dass aus jeder Lösung die für ``groundinsight``
benötigten Reduktions-Größen (Eingangsimpedanz, Transferimpedanzen,
``rho-f``-Kurve) extrahierbar sind.
"""

from __future__ import annotations

__all__: list[str] = []
