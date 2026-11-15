# Install the optional extras

The core package needs only numpy, scipy, pydantic, pandas, matplotlib and
sympy. Four optional *extras* add features; the
[installation guide](../getting-started/installation.md) covers Python
versions, Poetry and the development setup.

| Extra | Installs | Needed for |
|---|---|---|
| `geo` | requests, shapely, pyproj | OpenStreetMap building footprints (`gf.geo`, `LvNetworkLayout.from_osm`, OSM placement of `TnNetworkGenerator`) |
| `groundinsight` | groundinsight | live `BusType` objects (`gf.to_bustype`) and the groundinsight database (`save_bustype_to_db`); the JSON export works without it |
| `fast` | numba | compiled kernels for the two-layer image series and the line-source kernel; results are identical, solves are faster |
| `fem3d` | scikit-fem (BSD-3), pyamg (MIT) | the three-dimensional model of the `fem` backend: every world its axisymmetric model cannot hold (strips, meshes, rods side by side, several clusters, galvanic distributed conductors, resolved concrete) |

## Install

```bash
pip install "groundfield[geo]"
pip install "groundfield[fem3d]"
pip install "groundfield[geo,groundinsight,fast,fem3d]"   # all extras
```

From a checkout with Poetry:

```bash
poetry install --extras "geo fast fem3d"
poetry install --all-extras
```

## Check what is available

A missing extra is not an error until you call a function that needs it;
then groundfield raises an `ImportError` that names the extra to install.
To see what your environment has:

```python
import importlib.util

import groundfield as gf

print("groundfield", gf.__version__)
extras = {
    "shapely": "geo",
    "pyproj": "geo",
    "requests": "geo",
    "groundinsight": "groundinsight",
    "numba": "fast",
    "skfem": "fem3d",
    "pyamg": "fem3d",
}
available = {module: importlib.util.find_spec(module) is not None for module in extras}
for module, extra in extras.items():
    print(f"{module:14s} [{extra}]: {'installed' if available[module] else 'missing'}")
```

<!-- doctest-check
assert gf.__version__
assert set(available) == {"shapely", "pyproj", "requests", "groundinsight", "numba", "skfem", "pyamg"}
-->

## Is the compiled kernel used?

With `numba` installed the compiled kernels are picked up automatically;
there is nothing to configure. They return the same matrices bit for bit
as the numpy path, so results do not depend on the extra — only the run
time does. Compare both on your machine with the timing recipe in
[Speed up large studies](speed-up-large-studies.md).

## Without the `fem3d` extra

`fem` keeps its axisymmetric model (rods on one axis, rings around it) with
the core dependencies. A world that needs the three-dimensional model
raises an `ImportError` that names the extra; `FemOptions(model="axisymmetric")`
turns that into the axisymmetric model's own refusal, which names the
backends that solve the world. The extra changes no other backend and no
result of the axisymmetric model.
