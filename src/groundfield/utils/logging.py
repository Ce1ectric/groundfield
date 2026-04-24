"""Einheitliche Log-Konfiguration.

``groundfield`` verwendet den ``logging``-Modul von Python mit einem
einheitlichen Logger-Namen (``groundfield.<subpackage>.<modul>``). Dieses
Modul stellt eine kleine ``configure`` Funktion bereit, die in Notebooks
und Skripten aufgerufen werden kann, um eine Standard-Konfiguration zu
setzen.
"""

from __future__ import annotations

__all__: list[str] = []
