# Install the optional extras

This guide installs the optional parts of groundfield — OpenStreetMap
footprints, the groundinsight bridge, compiled kernels and the
three-dimensional `fem` model — and checks which of them your environment
has. Read it when a function asks for an extra, or before you set up a
machine for large studies.

**Prerequisites:** groundfield itself, installed as in the
[installation guide](../getting-started/installation.md), which also
covers Python versions, Poetry and the development setup. The core package
needs only numpy, scipy, pydantic, pandas, matplotlib and sympy.

## The four extras

| Extra | Installs | Needed for |
|---|---|---|
| `geo` | requests, shapely, pyproj | OpenStreetMap building footprints |
| `groundinsight` | groundinsight | live `BusType` objects and the groundinsight database |
| `fast` | numba | compiled kernels (same results, faster solves) |
| `fem3d` | scikit-fem (BSD-3), pyamg (MIT) | the three-dimensional model of `fem` |

- `geo` serves `gf.geo`, `LvNetworkLayout.from_osm` and the OSM placement
  of `TnNetworkGenerator`.
- `groundinsight` serves `gf.to_bustype` (a live `BusType`) and
  `save_bustype_to_db`; the JSON export works without it.
- `fast` compiles the kernels of the two-layer image series and of the
  line-source kernel; results are identical, solves are faster.
- `fem3d` adds the three-dimensional model of the `fem` backend: every
  world its axisymmetric model cannot hold (strips, meshes, rods side by
  side, several clusters, galvanic distributed conductors, resolved
  concrete).

## 1. Install

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

## 2. Check what is available

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

**What you get:** one line per module, `installed` or `missing`, with the
extra that provides it.

## 3. The compiled kernels need no setting

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

## See also

- [Installation](../getting-started/installation.md): Python versions,
  Poetry, the development setup.
- [`fem` engine](../explanation/engines/fem.md): the axisymmetric and the
  three-dimensional model.
- [Performance and scaling](../explanation/performance.md): measured run
  times with and without numba.
- [Work with OpenStreetMap offline](osm-offline.md): the `geo` extra in use.
