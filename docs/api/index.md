# API-Referenz

Die API-Referenz wird direkt aus den Docstrings des Quelltextes
erzeugt — via
[`mkdocstrings`](https://mkdocstrings.github.io/). Jede Unterseite
entspricht einem Subpackage.

- [Soil](soil.md) — Bodenmodelle.
- [Geometry](geometry.md) — Erder- und Leitergeometrien.
- [Conductors](conductors.md) — Leiter, PEN, Kabelschirme.
- [Solver](solver.md) — numerischer Feldlöser.
- [Coupling](coupling.md) — galvanische und induktive Kopplung.
- [Postprocess](postprocess.md) — Potential, Spannungen, Ströme.
- [IO](io.md) — JSON/VTK-Export und die Brücke zu `groundinsight`.

## Top-Level-Paket

Die wichtigsten Klassen werden perspektivisch auf der Paket-Ebene
re-exportiert:

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
geom = gf.HorizontalRingElectrode(radius=5.0, depth=0.8, wire_radius=0.005)
study = gf.FieldStudy(soil=soil, geometry=geom, frequencies=[50.0])
```

Bis die jeweiligen Implementierungen stehen, sind die Re-Exports in
`src/groundfield/__init__.py` bewusst leer gehalten.
