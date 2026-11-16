# Utilities

`groundfield.utils` holds the helpers that the other subpackages share:
the logger names, the "did you mean …?" suggestions in error messages,
and the two warning helpers of the public API — one for deprecated names
and one for settings that no backend reads. Read this page when you
configure logging or want to know where a warning comes from. What each
warning means and what to do about it is on
[Warnings and errors](../warnings.md); the deprecated names and their
replacements are listed on [Deprecations](../deprecations.md).

| Helper | Purpose |
|---|---|
| `get_logger`, `configure` | loggers below `groundfield`; a default stream handler |
| `suggest`, `did_you_mean` | closest valid names for an unknown name |
| `warn_deprecated`, `REMOVAL_VERSION`, `DEPRECATED_SINCE` | the deprecation policy of the calendar releases |
| `warn_inert`, `InertConfigurationWarning` | a setting is stored, but no backend reads it |

## Logging

Every module logs through `get_logger(__name__)`, so all messages sit
below the logger `groundfield`. The library installs no handler;
`configure(level)` adds one stream handler to `groundfield` when it has
none and sets its level. Any standard `logging` set-up works as well,
for example `logging.getLogger("groundfield.solver").setLevel(logging.DEBUG)`.

## Name suggestions

The factories reject an unknown name with the closest valid one:

```python
import groundfield as gf
from groundfield.utils import did_you_mean, suggest

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
try:
    gf.create_electrode(world, "rnig", name="a", center=(0.0, 0.0, 0.5), radius=1.0)
except ValueError as error:
    message = str(error)   # "Unknown electrode kind 'rnig'. Did you mean 'ring'? ..."

assert "Did you mean 'ring'?" in message
assert suggest("rnig", ["rod", "ring", "strip", "star"]) == ["ring"]
assert did_you_mean("rnig", ["rod", "ring"]) == " Did you mean 'ring'?"
```

## Deprecation policy

A deprecated name keeps working, emits a `DeprecationWarning` that names
its replacement and is removed no earlier than two releases and six
months after the announcement: for the names deprecated in
`DEPRECATED_SINCE` (2026.11) that is `REMOVAL_VERSION` (2027.5) at the
earliest. Both helpers point the warning at the first caller outside
groundfield and pydantic, the line the user can change, and stay silent
when groundfield's own code is the caller.

## API reference

::: groundfield.utils

`groundfield.utils.suggest` is also the name of the submodule that
defines the function, so the function is rendered here separately:

::: groundfield.utils.suggest.suggest
    options:
      heading_level: 3
