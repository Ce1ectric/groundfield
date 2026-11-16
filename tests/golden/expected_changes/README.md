# Expected golden-value changes

One TOML file per work stream (`core2.toml`, `coup.toml`, …). Every
entry names the finding that justifies the change. See
`../README.md` for the format.

At a release the baseline is regenerated and the files move to
`released/<version>/`, where the harness no longer reads them.
