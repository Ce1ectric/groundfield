# groundfield

**Numerische Feldberechnung von Erdungsanlagen.**

[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`groundfield` ist ein Open-Source-Python-Paket für die physikalische
Referenzmodellierung vernetzter Erdungssysteme. Innerhalb der
`groundmeas` / `groundinsight` / `groundfield` Softwarefamilie bildet
`groundfield` die feldtheoretische Seite ab: Bodenmodelle, Erder-
geometrien, Leiter und deren Kopplungen werden als 3D-Problem im
Erdreich formuliert und numerisch gelöst. Die Ergebnisse sind
Feldverläufe, Potenzialkurven, Stromverteilungen und — zentral —
reduzierte Ersatzmodelle (`rho-f`-Modell), die an `groundinsight`
übergeben werden können.

- **Dokumentation**: <https://ce1ectric.github.io/groundfield/>
- **Quellcode**: <https://github.com/Ce1ectric/groundfield>
- **Issue-Tracker**: <https://github.com/Ce1ectric/groundfield/issues>

## Einordnung in die Softwarefamilie

```
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (Messung)         (reduziertes                (Feldmodell,
                     Netzmodell)                 PDE-Referenz)
```

`groundfield` liefert das physikalisch fundierte Referenzmodell, aus
dem reduzierte Impedanz- und Mehrtor-Darstellungen abgeleitet werden.
Diese wandern als `BusType`-/`BranchType`-Formeln nach `groundinsight`
und können dort mit Messdaten aus `groundmeas` in Einklang gebracht
werden.

## Zielbild

`groundfield` entsteht als Werkzeug für **Arbeitspaket 1** der
Dissertation zu vernetzten Erdungssystemen. Untersucht werden unter
anderem:

- geschichtetes Erdreich (Zwei- und Mehrschicht-Modelle)
- beliebige Erdergeometrien (Ring-, Band-, Staberder, Fundamenterder,
  Maschen)
- Leiter, Kabelschirme und PEN mit gegenseitiger Kopplung
- Strom- und Potentialverteilung im Erdreich
- Einfluss der Stromeinspeisung und der Messgeometrie auf das
  Erdungsmessergebnis
- Ableitung reduzierter `rho-f`-Modelle für `groundinsight`

## Installation

`groundfield` benötigt **Python 3.12 oder neuer**.

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

Die Dokumentations-Extras liegen in einer optionalen Poetry-Gruppe:

```bash
poetry install --with docs
```

## Schnelleinstieg

```python
import groundfield as gf

# 1. Bodenmodell (z.B. Zwei-Schicht-Modell des Parameterraums AP1)
soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)

# 2. Geometrie der Erdungsanlage
geom = gf.HorizontalRingElectrode(
    radius=5.0, depth=0.8, wire_radius=0.005,
)

# 3. Feldstudie über einen Frequenzbereich
study = gf.FieldStudy(
    soil=soil, geometry=geom, frequencies=[50.0, 150.0, 250.0],
)
result = study.solve()

# 4. Auswertung
result.plot_potential_profile(along="x", y=0.0, z=0.0)
print(result.grounding_impedance())

# 5. Export eines reduzierten rho-f-Modells an groundinsight
bustype = gf.export_rho_f_bustype(
    result,
    name="TN-Ortsnetz-Ring",
    description="AP1 Referenzfall Ringleiter, Zweischicht",
)
```

Die API ist zu Beginn noch im Aufbau — viele Klassen sind bewusst als
Stubs angelegt und werden zusammen mit dem Arbeitspaket 1
ausgebaut.

## Leitprinzipien

- **Das PDE-/Feldmodell ist Referenz, nicht Endprodukt.** Der Löser
  muss so instrumentiert sein, dass sich aus jeder Lösung eine
  identifikations-taugliche Reduktion ableiten lässt.
- **Messbarkeit vor Genauigkeit.** Der relevante Frequenzbereich ist
  < 1 kHz; das erlaubt vereinfachte Bodenmodelle und schnelle Löser.
- **grey-box statt black-box.** Die geometrischen und materialseitigen
  Eingangsgrößen bleiben sichtbar; identifiziert wird nur das, was
  nicht ohnehin physikalisch vorgegeben ist.

## Entwicklung

```bash
# Tests mit Coverage
poetry run pytest --cov=groundfield

# Formatierung
poetry run black src tests scripts

# Doku lokal
poetry install --with docs
poetry run mkdocs serve
```

Ein Release wird über das Poetry-Skript angestoßen. Es aktualisiert die
Version in `pyproject.toml`, `src/groundfield/__init__.py` und
`CITATION.cff`, rollt den `[Unreleased]`-Block der `CHANGELOG.md` in
einen neuen Abschnitt und erzeugt einen annotierten Tag.

```bash
poetry run release patch
poetry run release minor
poetry run release major
poetry run release set 1.2.3
```

## Zitieren

Wenn du `groundfield` in wissenschaftlicher Arbeit verwendest, nutze
bitte die Angaben in `CITATION.cff`.

## Lizenz

`groundfield` wird unter der [MIT-Lizenz](LICENSE) veröffentlicht.
