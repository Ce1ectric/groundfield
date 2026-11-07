"""Release tool for groundfield (calendar versioning).

groundfield uses calendar versioning ``YYYY.MM.MICRO`` in its PEP 440
form: the month is not zero-padded and there is no hyphen (``2026.11.0``,
``2027.1.0``), because PEP 440 drops leading zeros and reads ``2026-11``
as a post-release. ``MICRO`` counts the earlier releases of the same
month, so ``2026.11.1`` is the second release of November 2026. Git tags
carry a ``v`` prefix (``v2026.11.0``).

Usage
-----
The tool is not installed with the package (no console script and no
top-level ``scripts`` package in the wheel, finding 10.02). Run it with
the project's Python from any directory; it always operates on the
repository that contains it::

    python scripts/release.py calver                    # version from today
    python scripts/release.py calver --date 2026-11-07  # explicit date
    python scripts/release.py set 2026.11.0rc1          # explicit version
    python scripts/release.py calver --dry-run          # validate, show plan

``calver``
    Derive the version from the release date (default: today) and the
    existing ``vYYYY.M.MICRO`` tags.
``set VERSION``
    Use an explicit version: canonical PEP 440, of the form
    ``YYYY.M.MICRO`` with an optional pre-, post- or dev-release suffix
    (``2026.11.0rc1``), sorting above the current version. This also
    finalises a pre-release (``2026.11.0rc1`` -> ``2026.11.0``).
``--date YYYY-MM-DD``
    Release date for the version (``calver``), the changelog heading
    and ``date-released`` in ``CITATION.cff``.
``--dry-run``
    Run every check and print the plan; write nothing, call no git
    command that changes the repository.

Procedure
---------
1. Validate everything before anything is written (finding 10.09): the
   repository root is the git top level, the working tree is clean
   (untracked files included), ``HEAD`` is on a branch, the version in
   ``pyproject.toml`` (``[project] version``),
   ``src/groundfield/__init__.py`` (``__version__``) and
   ``CITATION.cff`` (``version``) agree, the new version is valid and
   sorts above the current one, its tag does not exist, ``CHANGELOG.md``
   has a non-empty ``## [Unreleased]`` block and no section for the new
   version, ``changelog.d/`` holds no unmerged fragments, and
   ``CLAUDE.md`` does not hard-code a version (review pass 5).
2. Write the new version into the three files (and ``date-released``),
   and turn ``## [Unreleased]`` into ``## [X] - YYYY-MM-DD`` below a
   fresh empty ``## [Unreleased]``; the link references at the end of the
   changelog are updated.
3. Commit the four files as ``chore(release): vX`` and create the
   annotated tag ``vX``. Git always runs in the repository root. If the
   commit fails (for example a rejecting pre-commit hook), the files are
   restored and the index is reset.

The tool never pushes. Publish with ``git push origin <branch>`` and
``git push origin vX``; the tag starts the release workflow (tests,
tag/version check, build, PyPI trusted publishing).

Requirements: Python >= 3.12 and the ``packaging`` distribution (a
dependency of pytest, present in the development environment).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "scripts/release.py needs the 'packaging' distribution (pip install packaging)"
    ) from exc

logger = logging.getLogger("groundfield.release")

# --- constants ---------------------------------------------------------------

PYPROJECT = "pyproject.toml"
INIT_FILE = "src/groundfield/__init__.py"
CITATION = "CITATION.cff"
CHANGELOG = "CHANGELOG.md"
FRAGMENT_DIR = "changelog.d"

TAG_PREFIX = "v"
COMMIT_MESSAGE = "chore(release): {tag}"
TAG_MESSAGE = "groundfield {version}"
# Keep a Changelog heading form ``## [X] - YYYY-MM-DD``.
HEADING_SEPARATOR = " - "

# Canonical PEP 440 calendar version: YYYY.M.MICRO plus an optional
# pre-, post- or dev-release suffix. No leading zeros, no local part.
_NUMBER = r"(?:0|[1-9]\d*)"
CALVER_RE = re.compile(
    rf"^(?P<year>[1-9]\d{{3}})\.(?P<month>[1-9]|1[0-2])\.(?P<micro>{_NUMBER})"
    rf"(?:(?:a|b|rc){_NUMBER})?(?:\.post{_NUMBER})?(?:\.dev{_NUMBER})?$"
)

# Lower-cased placeholder lines that do not count as changelog content.
_PLACEHOLDERS = frozenset(
    {"_no changes yet._", "_nothing yet._", "nothing yet.", "nothing yet", "---"}
)

_UNRELEASED_HEADING_RE = re.compile(r"^## \[Unreleased\][^\n]*\n", re.MULTILINE)
_LEVEL2_HEADING_RE = re.compile(r"^## ", re.MULTILINE)
_LINK_REFERENCE_RE = re.compile(r"^\[[^\]\n]+\]:[ \t]", re.MULTILINE)
_UNRELEASED_LINK_RE = re.compile(
    r"^\[Unreleased\]:[ \t]+(?P<base>\S+/compare/)\S+?\.\.\.HEAD[ \t]*$",
    re.MULTILINE,
)
_INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"(?P<version>[^"]*)"', re.M)
_CITATION_VERSION_RE = re.compile(
    r"^version:[ \t]*(?P<quote>[\"']?)(?P<version>[^\"'\s#]+)(?P=quote)"
    r"[ \t]*(?:#[^\n]*)?$",
    re.MULTILINE,
)
_CITATION_DATE_RE = re.compile(r"^date-released:[^\n]*$", re.MULTILINE)
_TOML_TABLE_RE = re.compile(r"^\[(?P<name>[^\[\]\n]+)\][ \t]*(?:#[^\n]*)?$", re.M)
_TOML_VERSION_RE = re.compile(r'^version[ \t]*=[ \t]*"(?P<version>[^"]*)"', re.M)

# CLAUDE.md must not hard-code a version literal (review pass 5): flag any
# line that looks like ``__version__ = "<X.Y.Z>"`` or ``version: X.Y.Z``.
_CLAUDE_HARDCODED_VERSION_RE = re.compile(
    r"""(?mx)
    ^[^\n]*                              # leading prose / list marker
    (?:__version__|version)              # the field name
    \s*[:=]\s*                           # ':' or '='
    ['"]?
    (?P<version>\d+\.\d+\.\d+(?:[-.]?[0-9A-Za-z.-]+)?)
    ['"]?
    """
)


class ReleaseError(Exception):
    """A check failed; nothing has been written."""


@dataclass
class TextFile:
    r"""A text file of the release with its original and new content.

    Parameters
    ----------
    rel : str
        Path relative to the repository root (POSIX separators).
    original : str
        Content with ``\n`` line endings, as read.
    newline : str
        Line ending found in the file (``"\n"`` or ``"\r\n"``); it is
        kept when the file is written back.
    updated : str
        New content with ``\n`` line endings.
    """

    rel: str
    original: str
    newline: str
    updated: str = ""


@dataclass(frozen=True)
class ReleasePlan:
    """Everything the release will change, computed before any write.

    Parameters
    ----------
    root : Path
        Repository root.
    current : str
        Version recorded in the three version files.
    version : str
        New version (canonical PEP 440).
    date : datetime.date
        Release date.
    branch : str
        Branch that receives the release commit.
    files : tuple[TextFile, ...]
        Files to rewrite, in commit order.
    """

    root: Path
    current: str
    version: str
    date: dt.date
    branch: str
    files: tuple[TextFile, ...]

    @property
    def tag(self) -> str:
        """Git tag of the release (``vX``)."""
        return f"{TAG_PREFIX}{self.version}"


# --- versions ----------------------------------------------------------------


def validate_release_version(value: str) -> str:
    """Validate a version for a new release.

    Parameters
    ----------
    value : str
        Candidate version, e.g. ``"2026.11.0"`` or ``"2026.11.0rc1"``.

    Returns
    -------
    str
        ``value`` unchanged.

    Raises
    ------
    ReleaseError
        If ``value`` is not valid PEP 440, not in canonical form, carries
        a local part, or is not of the form ``YYYY.M.MICRO`` with an
        optional pre-, post- or dev-release suffix.
    """
    try:
        parsed = Version(value)
    except InvalidVersion as exc:
        raise ReleaseError(f"{value!r} is not a valid PEP 440 version") from exc
    if parsed.local is not None:
        raise ReleaseError(f"{value!r} has a local version part; PyPI rejects it")
    canonical = str(parsed)
    if canonical != value:
        raise ReleaseError(
            f"{value!r} is not the canonical PEP 440 form; use {canonical!r}"
        )
    if CALVER_RE.match(value) is None:
        raise ReleaseError(
            f"{value!r} is not a calendar version YYYY.M.MICRO "
            "(month without leading zero, optional aN/bN/rcN, .postN, .devN)"
        )
    return value


def calver_version(date: dt.date, tags: Iterable[str]) -> str:
    """Return the calendar version of a release on ``date``.

    Parameters
    ----------
    date : datetime.date
        Release date.
    tags : Iterable[str]
        Existing git tags. Only final releases ``vYYYY.M.MICRO`` of the
        same month count; pre-, post- and dev-releases are ignored.

    Returns
    -------
    str
        ``YYYY.M.MICRO`` where ``MICRO`` is the number of earlier releases
        in that month (one more than the highest existing ``MICRO``, so a
        deleted tag never makes a number be reused).
    """
    micros: list[int] = []
    for tag in tags:
        if not tag.startswith(TAG_PREFIX):
            continue
        try:
            parsed = Version(tag[len(TAG_PREFIX) :])
        except InvalidVersion:
            continue
        if parsed.is_prerelease or parsed.is_postrelease or parsed.local:
            continue
        if len(parsed.release) == 3 and parsed.release[:2] == (date.year, date.month):
            micros.append(parsed.release[2])
    micro = max(micros) + 1 if micros else 0
    return f"{date.year}.{date.month}.{micro}"


# --- file transformations ----------------------------------------------------


def _load_toml(text: str) -> dict[str, object]:
    """Parse ``pyproject.toml``, turning syntax errors into release errors."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseError(f"{PYPROJECT} is not valid TOML: {exc}") from exc


def _pyproject_version_table(text: str) -> str:
    """Return the table that holds the project version in ``pyproject.toml``."""
    data: dict = _load_toml(text)
    if "version" in data.get("project", {}):
        return "project"
    if "version" in data.get("tool", {}).get("poetry", {}):
        return "tool.poetry"
    raise ReleaseError(f"{PYPROJECT}: no [project] version found")


def read_pyproject_version(text: str) -> str:
    """Return the project version recorded in ``pyproject.toml``.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    str
        ``[project] version`` (or ``[tool.poetry] version`` for legacy
        metadata).
    """
    data: dict = _load_toml(text)
    table = _pyproject_version_table(text)
    if table == "project":
        return str(data["project"]["version"])
    return str(data["tool"]["poetry"]["version"])


def bump_pyproject(text: str, version: str) -> str:
    """Replace the project version in ``pyproject.toml``.

    Only the ``version`` key of the ``[project]`` table (or of
    ``[tool.poetry]`` for legacy metadata) changes; ``version`` keys of
    other tables stay untouched.

    Parameters
    ----------
    text : str
        File content.
    version : str
        New version.

    Returns
    -------
    str
        Updated content.
    """
    table = _pyproject_version_table(text)
    headers = list(_TOML_TABLE_RE.finditer(text))
    start = next(
        (h.end() for h in headers if h["name"].replace(" ", "") == table), None
    )
    if start is None:
        raise ReleaseError(f"{PYPROJECT}: table [{table}] not found")
    end = next((h.start() for h in headers if h.start() > start), len(text))
    body, count = _TOML_VERSION_RE.subn(
        f'version = "{version}"', text[start:end], count=1
    )
    if count != 1:
        raise ReleaseError(f"{PYPROJECT}: no version line in [{table}]")
    updated = text[:start] + body + text[end:]
    if read_pyproject_version(updated) != version:  # pragma: no cover - guard
        raise ReleaseError(f"{PYPROJECT}: version rewrite failed")
    return updated


def read_init_version(text: str) -> str:
    """Return ``__version__`` from ``src/groundfield/__init__.py``.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    str
        The version literal.
    """
    matches = _INIT_VERSION_RE.findall(text)
    if len(matches) != 1:
        raise ReleaseError(f"{INIT_FILE}: expected one __version__ line")
    return str(matches[0])


def bump_init(text: str, version: str) -> str:
    """Replace ``__version__`` in ``src/groundfield/__init__.py``.

    Parameters
    ----------
    text : str
        File content.
    version : str
        New version.

    Returns
    -------
    str
        Updated content.
    """
    read_init_version(text)
    return _INIT_VERSION_RE.sub(f'__version__ = "{version}"', text, count=1)


def read_citation_version(text: str) -> str:
    """Return the top-level ``version`` of ``CITATION.cff``.

    Parameters
    ----------
    text : str
        File content.

    Returns
    -------
    str
        The version string.
    """
    matches = list(_CITATION_VERSION_RE.finditer(text))
    if len(matches) != 1:
        raise ReleaseError(f"{CITATION}: expected one top-level 'version:' line")
    return matches[0]["version"]


def bump_citation(text: str, version: str, date: dt.date) -> str:
    """Set ``version`` and ``date-released`` in ``CITATION.cff``.

    Parameters
    ----------
    text : str
        File content.
    version : str
        New version.
    date : datetime.date
        Release date; ``date-released`` is inserted after ``version`` if
        it is missing.

    Returns
    -------
    str
        Updated content.
    """
    read_citation_version(text)
    version_line = f'version: "{version}"'
    date_line = f'date-released: "{date.isoformat()}"'
    updated = _CITATION_VERSION_RE.sub(version_line, text, count=1)
    if _CITATION_DATE_RE.search(updated):
        return _CITATION_DATE_RE.sub(date_line, updated, count=1)
    return updated.replace(version_line, f"{version_line}\n{date_line}", 1)


def _has_content(body: str) -> bool:
    """Return ``True`` if a changelog block holds more than placeholders."""
    return any(
        line.strip() and line.strip().lower() not in _PLACEHOLDERS
        for line in body.splitlines()
    )


def roll_changelog(text: str, version: str, date: dt.date) -> str:
    """Turn ``## [Unreleased]`` into the section of a new release.

    The body of ``## [Unreleased]`` (up to the next level-2 heading, or
    the link references at the end) becomes ``## [version] - date`` below
    a fresh, empty ``## [Unreleased]``; sections are separated by ``---``.
    The ``[Unreleased]: .../compare/vOLD...HEAD`` link reference is moved
    to the new tag and a ``[version]: .../releases/tag/vX`` reference is
    added below it.

    Parameters
    ----------
    text : str
        Content of ``CHANGELOG.md``.
    version : str
        New version.
    date : datetime.date
        Release date.

    Returns
    -------
    str
        Updated content.

    Raises
    ------
    ReleaseError
        If there is no ``## [Unreleased]`` heading, the block is empty, or
        a section for ``version`` already exists.
    """
    heading = _UNRELEASED_HEADING_RE.search(text)
    if heading is None:
        raise ReleaseError(f"{CHANGELOG}: no '## [Unreleased]' heading")
    if re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE):
        raise ReleaseError(f"{CHANGELOG}: a section [{version}] already exists")
    start = heading.end()
    candidates = [len(text)]
    for pattern in (_LEVEL2_HEADING_RE, _LINK_REFERENCE_RE):
        match = pattern.search(text, start)
        if match is not None:
            candidates.append(match.start())
    end = min(candidates)
    body = text[start:end].strip("\n")
    if not _has_content(body):
        raise ReleaseError(
            f"{CHANGELOG}: the '[Unreleased]' block is empty; add release "
            "notes before cutting the release"
        )
    body = re.sub(r"(?:\n[ \t]*)*\n---[ \t]*$", "", body).strip("\n")
    section = (
        "## [Unreleased]\n\n---\n\n"
        f"## [{version}]{HEADING_SEPARATOR}{date.isoformat()}\n\n"
        f"{body}\n\n---\n\n"
    )
    updated = text[: heading.start()] + section + text[end:]

    link = _UNRELEASED_LINK_RE.search(updated)
    if link is None:
        logger.warning(
            "%s: no '[Unreleased]: .../compare/...HEAD' link reference; "
            "link references not updated",
            CHANGELOG,
        )
        return updated
    base = link["base"]
    tag_url = base.replace("/compare/", "/releases/tag/")
    replacement = (
        f"[Unreleased]: {base}{TAG_PREFIX}{version}...HEAD\n"
        f"[{version}]: {tag_url}{TAG_PREFIX}{version}"
    )
    return updated[: link.start()] + replacement + updated[link.end() :]


# --- repository checks -------------------------------------------------------


def _check_claude_md_no_hardcoded_version(root: Path) -> None:
    """Refuse the release if ``CLAUDE.md`` hard-codes a version literal.

    ``CLAUDE.md`` (local, not committed) must defer to the canonical
    version sources. Lines inside fenced code blocks and the "do not
    hard-code" reminder itself are ignored.

    Parameters
    ----------
    root : Path
        Repository root.

    Raises
    ------
    RuntimeError
        If ``CLAUDE.md`` contains a hard-coded version literal.
    """
    path = root / "CLAUDE.md"
    if not path.is_file():
        return
    in_fence = False
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or "do not hard-code" in line.lower():
            continue
        if _CLAUDE_HARDCODED_VERSION_RE.search(line) is not None:
            offenders.append((lineno, line.rstrip()))
    if offenders:
        lines = "\n".join(f"  CLAUDE.md:{n}: {text}" for n, text in offenders)
        raise RuntimeError(
            "CLAUDE.md hard-codes a version literal:\n"
            f"{lines}\n"
            "Remove the literal - the canonical sources are pyproject.toml, "
            "src/groundfield/__init__.py:__version__ and CITATION.cff, which "
            "scripts/release.py keeps in sync; CLAUDE.md must defer to them."
        )


def _git(root: Path, *args: str, check: bool = True) -> str:
    """Run git in the repository root and return its standard output.

    Parameters
    ----------
    root : Path
        Repository root; always the working directory of git (10.09).
    *args : str
        Arguments after ``git``.
    check : bool
        Raise :class:`ReleaseError` on a non-zero exit status.

    Returns
    -------
    str
        Standard output without surrounding whitespace.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _git_ok(root: Path, *args: str) -> bool:
    """Return ``True`` if a git command exits with status 0."""
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.returncode == 0


def _check_repository(root: Path) -> str:
    """Check the git state and return the current branch.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    str
        Name of the checked-out branch.

    Raises
    ------
    ReleaseError
        If ``root`` is not the top level of a git work tree, the tree is
        not clean (untracked files included) or ``HEAD`` is detached.
    """
    try:
        toplevel = _git(root, "rev-parse", "--show-toplevel")
    except (ReleaseError, OSError) as exc:
        raise ReleaseError(f"{root} is not a git work tree: {exc}") from exc
    if Path(toplevel).resolve() != root.resolve():
        raise ReleaseError(f"{root} is not the top level of its git repository")
    status = _git(root, "status", "--porcelain")
    if status:
        shown = "\n".join(f"  {line}" for line in status.splitlines()[:20])
        raise ReleaseError(
            "the working tree is not clean; commit or stash first:\n" + shown
        )
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if not branch:
        raise ReleaseError("HEAD is detached; check out the release branch first")
    return branch


def _check_fragments(root: Path) -> None:
    """Refuse the release while ``changelog.d/`` holds unmerged fragments."""
    folder = root / FRAGMENT_DIR
    if not folder.is_dir():
        return
    fragments = sorted(
        p.relative_to(root).as_posix()
        for p in folder.glob("*.md")
        if p.name.lower() != "readme.md"
    )
    if fragments:
        raise ReleaseError(
            "unmerged changelog fragments; merge them into [Unreleased] of "
            f"{CHANGELOG} and delete them: " + ", ".join(fragments)
        )


def _read(root: Path, rel: str) -> TextFile:
    r"""Read a release file, normalising line endings to ``\n``."""
    path = root / rel
    if not path.is_file():
        raise ReleaseError(f"{rel} not found")
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return TextFile(rel=rel, original=raw.replace("\r\n", "\n"), newline=newline)


def _write(root: Path, text_file: TextFile, content: str) -> None:
    """Write a release file atomically, keeping its line endings."""
    path = root / text_file.rel
    tmp = path.with_name(path.name + ".release-tmp")
    with open(tmp, "w", encoding="utf-8", newline=text_file.newline) as handle:
        handle.write(content)
    os.replace(tmp, path)


# --- plan and apply ----------------------------------------------------------


def plan_release(
    root: Path, command: str, version: str | None, date: dt.date
) -> ReleasePlan:
    """Run every check and compute the new file contents.

    Nothing is written. The order of the checks puts the cheap,
    user-actionable ones first.

    Parameters
    ----------
    root : Path
        Repository root.
    command : {"calver", "set"}
        How the new version is determined.
    version : str or None
        Explicit version for ``"set"``.
    date : datetime.date
        Release date.

    Returns
    -------
    ReleasePlan
        The validated plan.

    Raises
    ------
    ReleaseError
        If any check fails.
    """
    branch = _check_repository(root)
    try:
        _check_claude_md_no_hardcoded_version(root)
    except RuntimeError as exc:
        raise ReleaseError(str(exc)) from exc
    _check_fragments(root)

    pyproject = _read(root, PYPROJECT)
    init = _read(root, INIT_FILE)
    citation = _read(root, CITATION)
    changelog = _read(root, CHANGELOG)
    found = {
        PYPROJECT: read_pyproject_version(pyproject.original),
        INIT_FILE: read_init_version(init.original),
        CITATION: read_citation_version(citation.original),
    }
    if len(set(found.values())) != 1:
        listing = "\n".join(f"  {rel}: {value}" for rel, value in found.items())
        raise ReleaseError(f"the version files disagree:\n{listing}")
    current = found[PYPROJECT]
    try:
        current_version = Version(current)
    except InvalidVersion as exc:
        raise ReleaseError(f"current version {current!r} is not PEP 440") from exc

    tags = _git(root, "tag", "--list").split()
    if command == "calver":
        new = calver_version(date, tags)
    elif version is not None:
        new = version
    else:  # pragma: no cover - argparse requires VERSION for 'set'
        raise ReleaseError("'set' needs a VERSION")
    validate_release_version(new)
    if Version(new) <= current_version:
        raise ReleaseError(
            f"the new version {new} does not sort above the current version {current}"
        )
    tag = f"{TAG_PREFIX}{new}"
    if _git_ok(root, "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}"):
        raise ReleaseError(f"tag {tag} already exists")

    pyproject.updated = bump_pyproject(pyproject.original, new)
    init.updated = bump_init(init.original, new)
    citation.updated = bump_citation(citation.original, new, date)
    changelog.updated = roll_changelog(changelog.original, new, date)
    return ReleasePlan(
        root=root,
        current=current,
        version=new,
        date=date,
        branch=branch,
        files=(pyproject, init, citation, changelog),
    )


def apply_release(plan: ReleasePlan) -> None:
    """Write the files, commit them and create the annotated tag.

    Parameters
    ----------
    plan : ReleasePlan
        Validated plan from :func:`plan_release`.

    Raises
    ------
    ReleaseError
        If writing, committing or tagging fails. When the commit fails,
        the files are restored and the index is reset first.
    """
    rels = [f.rel for f in plan.files]
    try:
        for text_file in plan.files:
            _write(plan.root, text_file, text_file.updated)
        _git(plan.root, "add", "--", *rels)
        _git(plan.root, "commit", "-m", COMMIT_MESSAGE.format(tag=plan.tag))
    except (ReleaseError, OSError) as exc:
        for text_file in plan.files:
            _write(plan.root, text_file, text_file.original)
        _git(plan.root, "reset", "--quiet", "--", *rels, check=False)
        raise ReleaseError(f"{exc}\nthe release files were restored") from exc
    try:
        _git(
            plan.root,
            "tag",
            "--annotate",
            plan.tag,
            "-m",
            TAG_MESSAGE.format(version=plan.version),
        )
    except ReleaseError as exc:
        raise ReleaseError(
            f"{exc}\nthe release commit exists but the tag is missing; create "
            f"it with: git tag --annotate {plan.tag} -m "
            f"'{TAG_MESSAGE.format(version=plan.version)}'"
        ) from exc


def _repo_root() -> Path:
    """Return the repository that contains this script."""
    root = Path(__file__).resolve().parent.parent
    if not (root / PYPROJECT).is_file():
        raise ReleaseError(f"{root} has no {PYPROJECT}; is the script in scripts/?")
    return root


def _parse_date(value: str) -> dt.date:
    """Parse ``--date`` (``YYYY-MM-DD``)."""
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/release.py",
        description=(
            "Cut a calendar-versioned groundfield release: validate, bump the "
            "version files, roll the changelog, commit and tag. Never pushes."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calver = commands.add_parser(
        "calver", help="version YYYY.M.MICRO from the release date and the tags"
    )
    explicit = commands.add_parser(
        "set", help="explicit PEP 440 calendar version, e.g. 2026.11.0rc1"
    )
    explicit.add_argument("version", help="new version")
    for sub in (calver, explicit):
        sub.add_argument(
            "--date",
            type=_parse_date,
            default=None,
            metavar="YYYY-MM-DD",
            help="release date (default: today)",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="run every check and print the plan without changing anything",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release tool.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if a check or a git step failed
        (argparse exits with 2 on usage errors).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    date = args.date if args.date is not None else dt.date.today()
    try:
        root = _repo_root()
        plan = plan_release(root, args.command, getattr(args, "version", None), date)
    except ReleaseError as exc:
        logger.error("release aborted, nothing was changed: %s", exc)
        return 1

    heading = f"## [{plan.version}]{HEADING_SEPARATOR}{plan.date.isoformat()}"
    logger.info("repository:      %s (branch %s)", plan.root, plan.branch)
    logger.info("current version: %s", plan.current)
    logger.info("new version:     %s (%s)", plan.version, args.command)
    logger.info("files:           %s", ", ".join(f.rel for f in plan.files))
    logger.info("changelog:       '## [Unreleased]' becomes '%s'", heading)
    if args.dry_run:
        logger.info(
            "[dry-run] would commit '%s' and tag '%s'; nothing was changed",
            COMMIT_MESSAGE.format(tag=plan.tag),
            plan.tag,
        )
        return 0
    try:
        apply_release(plan)
    except ReleaseError as exc:
        logger.error("release failed: %s", exc)
        return 1
    logger.info(
        "release %s committed and tagged on branch %s. Nothing was pushed; "
        "publish with:\n    git push origin %s\n    git push origin %s",
        plan.tag,
        plan.branch,
        plan.branch,
        plan.tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
