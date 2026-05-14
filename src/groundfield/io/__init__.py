"""File I/O and export to the sister projects.

This subpackage forms the interface between ``groundfield`` and the
rest of the software family:

* ``io.groundinsight``: export of a reduced ``rho-f`` model
  (``RhoFStandardFit`` or ``VectorFitResult``) into the
  ``groundinsight`` ``BusType`` schema. Two equally supported
  transports — JSON file (neutral, schema-versioned) and a live
  Python ``BusType`` instance via lazy import. See
  ``docs/adr/0008-groundinsight-bridge.md``.
* ``io.json``: serialise and deserialise a complete ``FieldStudy`` to
  / from JSON for reproducibility and regression tests. *Reserved.*
* ``io.vtk``: export of 3-D field results to VTK / ParaView.
  *Reserved.*
* ``io.csv``: lightweight export of path and point results.
  *Reserved.*

Guiding principle
-----------------
The export to ``groundinsight`` closes the pipeline: the PDE /
field model produces the reduced equivalent model used for
planning and type studies. ``io.groundinsight`` provides the
bridge.
"""

from __future__ import annotations

from groundfield.io.csv import (
    save_cluster_impedances_csv,
    save_electrode_table_csv,
    save_potential_path_csv,
)
from groundfield.io.groundinsight import (
    BusTypeSpec,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    evaluate_spec,
    fit_quality_summary,
    load_bustype_json,
    save_bustype_json,
    save_bustype_to_db,
    to_bustype,
    to_bustype_dict,
)
from groundfield.io.vtk import (
    export_field_vtk,
    export_geometry_vtk,
)

__all__ = [
    "BusTypeSpec",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "to_bustype_dict",
    "to_bustype",
    "save_bustype_json",
    "load_bustype_json",
    "save_bustype_to_db",
    "evaluate_spec",
    "fit_quality_summary",
    "save_potential_path_csv",
    "save_electrode_table_csv",
    "save_cluster_impedances_csv",
    "export_geometry_vtk",
    "export_field_vtk",
]
