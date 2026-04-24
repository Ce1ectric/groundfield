"""Datei-I/O und Export an die Schwesterprojekte.

Dieses Subpackage bildet die Schnittstelle von ``groundfield`` zur
übrigen Softwarefamilie:

* ``io.groundinsight``: Export eines reduzierten ``rho-f``-Modells
  (Zweipol-Impedanz bzw. Mehrtor als Funktion von Bodenparameter und
  Frequenz) in die von ``groundinsight`` erwarteten Impedanzformeln.
* ``io.json``: Ein- und Auslesen einer gesamten ``FieldStudy`` als
  JSON-Beschreibung für Reproduzierbarkeit und Regressionstests.
* ``io.vtk``: Export von 3D-Feldverläufen nach VTK/ParaView.
* ``io.csv``: Schlanker Export von Pfad- und Punktergebnissen.

Leitprinzip
-----------
Der Export an ``groundinsight`` ist der Dreh- und Angelpunkt der
Dissertation: aus dem PDE-/Feldmodell entsteht das für Planung und
Typisierung taugliche reduzierte Ersatzmodell. Dieses Modul stellt den
entsprechenden Brücken-Kopf bereit.
"""

from __future__ import annotations

__all__: list[str] = []
