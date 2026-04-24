# Schnelleinstieg

> **Hinweis:** Dieses Beispiel zeigt die angestrebte API. Einige Klassen
> sind in Version `0.1.0` bewusst als Stubs angelegt und werden im
> Verlauf von Arbeitspaket 1 ausgebaut.

```python
import groundfield as gf

# 1) Bodenmodell: typischer Zwei-Schicht-Boden aus dem AP1-Parameterraum
soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)

# 2) Geometrie: Ringerder um eine Ortsnetzstation
geom = gf.HorizontalRingElectrode(
    radius=5.0,
    depth=0.8,
    wire_radius=0.005,
)

# 3) Frequenzbereich < 1 kHz
frequencies = [50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0]

# 4) Feldstudie aufbauen und lösen
study = gf.FieldStudy(soil=soil, geometry=geom, frequencies=frequencies)
result = study.solve()

# 5) Auswertung
Z_in = result.grounding_impedance()          # Eingangsimpedanz je Frequenz
phi = result.potential_along(
    start=(0.0, 0.0, 0.0),
    end=(30.0, 0.0, 0.0),
    n=120,
)                                              # Potentialverlauf
U_T = result.touch_voltage(point=(1.0, 0.0, 0.0))

# 6) Reduktion auf ein rho-f-Modell für groundinsight
bustype = gf.export_rho_f_bustype(
    result,
    name="TN-Ortsnetz-Ring",
    description="AP1 Referenzfall Ringleiter, Zweischicht",
)
```

## Einordnung in Arbeitspaket 1

- Das obige Skript entspricht einem *Referenzfall der Erdungsmessung*.
- Durch Variation von `soil.rho_1`, `soil.rho_2`, `soil.h_1` und der
  Hilfselektroden-Position wird der Parameterraum aufgespannt.
- Die resultierenden `rho-f`-Kurven sind Eingang für die Modell-
  reduktion in `groundinsight`.
