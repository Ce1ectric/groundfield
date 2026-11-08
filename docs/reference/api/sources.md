# Sources

The :mod:`groundfield.sources` module exposes the source types that
drive a :class:`World`: :class:`CurrentSource` (impressed current
into an electrode cluster) and :class:`VoltageSource` (impressed
voltage between an injection electrode and an explicit return
electrode).

Since 0.5.0 the public :data:`Source` annotation is a Pydantic
discriminated union (``Annotated[Union[...], Discriminator("kind")]``)
with a companion :data:`SourceAdapter` for stand-alone source-dict
validation. JSON / dict round-trips report errors against the
selected sub-class instead of dumping the whole union's validator
chain.

## Round-tripping a source list

```python
from groundfield.sources import SourceAdapter

dicts = [
    {"kind": "current", "name": "inj_1",
     "attached_to": "g1", "magnitude": 1.0},
    {"kind": "voltage", "name": "v_aux",
     "attached_to": "g1", "return_to": "g_aux", "magnitude": 100.0},
]
sources = [SourceAdapter.validate_python(d) for d in dicts]
```

The discriminator surface (Pass 5) and the typed validation surface
keep error messages local to the offending source.

## API reference

::: groundfield.sources
