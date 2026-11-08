# Result files

`save_result(result, path)` writes a `FieldResult` to a versioned JSON
file and `load_result(path)` reads it back; `world_fingerprint(world)`
is the hash of a world that every solve records in its provenance. A
parameter study can therefore store each solve and later check which
world, engine and package versions produced it.

## Format

One UTF-8 JSON document (standard JSON: no `NaN` / `Infinity` tokens,
no pickle) with an envelope:

```json
{
  "format": "groundfield.result",
  "schema_version": 1,
  "written_by": "groundfield <version>",
  "provenance": {"groundfield_version": "...", "engine": {...}, "world_fingerprint": "...", "solved_at_utc": "..."},
  "result": {...}
}
```

`result` is the lossless JSON of the `FieldResult`: complex numbers are
strings, non-finite floats of typed fields are `"Infinity"` / `"NaN"`
strings, and metadata values JSON cannot express are tagged objects
(arrays with dtype and shape, tuples, complex numbers). `provenance`
repeats the solve provenance (groundfield, numpy, scipy, pydantic and
Python versions, platform, engine configuration, world fingerprint, UTC
time stamp) so it can be read without parsing the result.

**Compatibility.** `load_result` reads every schema version up to
`RESULT_SCHEMA_VERSION` and raises `ResultFormatError` for a newer one
instead of guessing. The loader reads only what the writer writes
(plain numeric and string arrays with bounded size and nesting), so a
file from an untrusted source cannot make it allocate arbitrary memory
or construct objects. Files are written atomically; `overwrite=False`
(default) refuses to replace an existing file.

The world is not part of the file. Save it alongside with
`world.model_dump_json()` (exact round trip) and compare it with
`world_fingerprint(world)` against `provenance["world_fingerprint"]`.

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_source(world, attached_to="g1", magnitude=1.0)
result = gf.create_engine(frequencies=[50.0, 1000.0]).solve(world)

path = gf.save_result(result, "rod.result.json")
loaded = gf.load_result(path)

assert loaded.input_impedance() == result.input_impedance()        # bit-identical
assert loaded.metadata["provenance"]["world_fingerprint"] == gf.world_fingerprint(world)
```

## API reference

::: groundfield.io.results
