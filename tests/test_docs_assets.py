"""Documentation assets: no third-party JavaScript at page load (finding 10.08).

v0.15.0 loaded MathJax from ``cdn.jsdelivr.net/npm/mathjax@3`` - a
floating major version without subresource integrity. MathJax 3.2.2 is
now vendored under ``docs/javascripts/vendor/mathjax/``: the combined
component ``tex-mml-chtml.js`` plus the files it loads lazily (CHTML web
fonts, autoloaded TeX extensions) and the Apache-2.0 license, byte-identical
to the npm tarball ``mathjax-3.2.2.tgz`` whose registry integrity
(SHA-512) was verified when the files were imported. ``SHA256SUMS`` pins
every vendored file, so an accidental edit (for example by an
end-of-file fixer) is caught here.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
VENDOR = DOCS / "javascripts" / "vendor" / "mathjax"
# SHA-384 of es5/tex-mml-chtml.js in the npm tarball mathjax-3.2.2.tgz
# (registry integrity sha512-Bt+SSVU8eBG27zChVewOicYs7Xsdt40qm4+UpHyX7k0/
# O9NliPc+x77k1/FEsPsjKPZGJvtRZM1vO+geW0OhGw==).
MATHJAX_322_SRI = (
    "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"
)


def _extra_javascript() -> list[str]:
    """Entries of ``extra_javascript`` in ``mkdocs.yml`` (plain or ``path:``)."""
    lines = (REPO / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
    start = lines.index("extra_javascript:") + 1
    entries: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break  # next top-level key
        match = re.match(r"^\s*-\s+(?:path:\s*)?(?P<value>[^#\s]+)", line)
        if match is not None:
            entries.append(match["value"].strip("'\""))
    return entries


@pytest.mark.finding("10.08")
def test_docs_load_no_remote_javascript() -> None:
    entries = _extra_javascript()
    assert entries, "extra_javascript is empty"
    remote = [e for e in entries if re.match(r"^(?:https?:)?//", e)]
    assert remote == [], "third-party script without integrity protection"
    for entry in entries:
        assert (DOCS / entry).is_file(), entry
    assert "javascripts/vendor/mathjax/tex-mml-chtml.js" in entries
    # the configuration must be defined before MathJax starts
    assert entries.index("javascripts/mathjax.js") < entries.index(
        "javascripts/vendor/mathjax/tex-mml-chtml.js"
    )


@pytest.mark.finding("10.08")
def test_vendored_mathjax_is_upstream_3_2_2() -> None:
    main = (VENDOR / "tex-mml-chtml.js").read_bytes()
    sri = "sha384-" + base64.b64encode(hashlib.sha384(main).digest()).decode()
    assert sri == MATHJAX_322_SRI


@pytest.mark.finding("10.08")
def test_vendored_files_match_their_checksums() -> None:
    listed: dict[str, str] = {}
    for line in (VENDOR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.lstrip("*")] = digest
    present = {
        p.relative_to(VENDOR).as_posix()
        for p in VENDOR.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "PROVENANCE.txt"}
    }
    assert present == set(listed)
    for name, digest in listed.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    # what the combined component loads lazily is present
    assert len([n for n in listed if n.startswith("output/chtml/fonts/woff-v2/")]) == 23
    for extension in ("boldsymbol", "cancel", "color", "mathtools", "physics"):
        assert f"input/tex/extensions/{extension}.js" in listed
    assert "LICENSE" in listed
