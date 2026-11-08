"""Tests for the release tool ``scripts/release.py`` (calendar versioning).

The tool is exercised the way it is used: as ``python scripts/release.py``
in a subprocess, on a throw-away git repository that contains a copy of
the script and minimal (or real) project files. No network access is
needed; nothing is ever pushed.

Finding 10.09 (review pass 10): the v0.15.0 script ran git in the process
working directory, rewrote the version files before it validated the
changelog, had no branch / tag-exists checks and could not finalise a
pre-release. The tests marked ``finding("10.09")`` fail on v0.15.0 for
exactly these reasons.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release.py"

RELEASE_FILES = (
    "pyproject.toml",
    "src/groundfield/__init__.py",
    "CITATION.cff",
    "CHANGELOG.md",
)

PYPROJECT = """\
[project]
name = "groundfield"
version = "{version}"
requires-python = ">=3.12,<4.0"
dependencies = ["numpy (>=2.1.0,<3.0.0)"]

[tool.poetry]
packages = [{{ include = "groundfield", from = "src" }}]

[tool.poetry.group.docs.dependencies]
mkdocstrings = {{ extras = ["python"], version = "^0.26.1" }}

[tool.mypy]
python_version = "3.12"
"""

INIT = '''\
"""groundfield (test fixture)."""

__version__ = "{version}"
'''

CITATION = """\
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "groundfield"
type: software
authors:
  - family-names: Ehlert
    given-names: Christian
version: "{version}"
date-released: "2026-07-31"
license: MIT
"""

CHANGELOG = """\
# Changelog

All notable changes are documented in this file.

---

## [Unreleased]

{unreleased}
---

## [0.15.0] - 2026-07-31

### Fixed

- An old fix.

---

## Roadmap

- Something for later.

[Unreleased]: https://github.com/Ce1ectric/groundfield/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.15.0
"""

UNRELEASED_BODY = """\
### Added

- **Calendar versioning:** a new feature.

### Fixed

- **Release tool (10.09):** a fix.
"""


# ---------------------------------------------------------------- helpers


def _load_release_module() -> ModuleType:
    """Import ``scripts/release.py`` as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("gf_release_tool", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release() -> ModuleType:
    return _load_release_module()


def _git_env(home: Path) -> dict[str, str]:
    """Environment isolated from the user's global and system git config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    config = home / "gitconfig"
    if not config.exists():
        config.write_text(
            "[user]\n\tname = Release Test\n\temail = release-test@example.org\n"
            "[init]\n\tdefaultBranch = main\n"
            "[commit]\n\tgpgsign = false\n"
            "[tag]\n\tgpgsign = false\n",
            encoding="utf-8",
        )
    env["GIT_CONFIG_GLOBAL"] = str(config)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


class Repo:
    """A throw-away git repository with a copy of the release script."""

    def __init__(self, path: Path, env: dict[str, str]) -> None:
        self.path = path
        self.env = env

    def git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
        return result.stdout.strip()

    def write(self, rel: str, text: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def commit_all(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def snapshot(self) -> dict[str, str]:
        return {rel: self.read(rel) for rel in RELEASE_FILES}

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def tags(self) -> set[str]:
        return set(self.git("tag", "--list").split())

    def run_release(
        self, *args: str, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run ``python <repo>/scripts/release.py ARGS`` from ``cwd``."""
        return subprocess.run(
            [sys.executable, str(self.path / "scripts" / "release.py"), *args],
            cwd=cwd if cwd is not None else self.path,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )


def _make_repo(
    tmp_path: Path,
    *,
    version: str = "0.15.0",
    unreleased: str = UNRELEASED_BODY,
    tags: tuple[str, ...] = ("v0.15.0",),
) -> Repo:
    path = tmp_path / "repo"
    path.mkdir()
    repo = Repo(path, _git_env(tmp_path))
    repo.git("init", "-q")
    repo.write("pyproject.toml", PYPROJECT.format(version=version))
    repo.write("src/groundfield/__init__.py", INIT.format(version=version))
    repo.write("CITATION.cff", CITATION.format(version=version))
    repo.write("CHANGELOG.md", CHANGELOG.format(unreleased=unreleased))
    repo.write(".gitignore", "CLAUDE.md\n")
    repo.write("docs/index.md", "# docs\n")
    (path / "scripts").mkdir()
    shutil.copy(SCRIPT, path / "scripts" / "release.py")
    repo.commit_all("initial")
    for tag in tags:
        repo.git("tag", "-a", tag, "-m", tag)
    return repo


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return _make_repo(tmp_path)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _assert_untouched(
    repo: Repo, before: dict[str, str], head: str, tags: set[str]
) -> None:
    assert repo.snapshot() == before, "release files were modified"
    assert repo.head() == head, "a commit was created"
    assert repo.tags() == tags, "a tag was created"
    assert repo.git("status", "--porcelain") == "", "working tree is dirty"


# ---------------------------------------------------------------- CalVer core


@pytest.mark.parametrize(
    ("date", "tags", "expected"),
    [
        ("2026-11-07", [], "2026.11.0"),
        ("2026-11-07", ["v0.15.0", "v0.14.1"], "2026.11.0"),
        ("2026-11-30", ["v2026.11.0"], "2026.11.1"),
        ("2026-11-30", ["v2026.11.0", "v2026.11.1", "v2026.10.0"], "2026.11.2"),
        # pre-releases, post-releases and foreign tags are not releases
        ("2026-11-30", ["v2026.11.0rc1", "v2026.11.0.post1", "nightly"], "2026.11.0"),
        # a deleted tag never makes the next release reuse a number
        ("2026-11-30", ["v2026.11.0", "v2026.11.2"], "2026.11.3"),
        # month unpadded (PEP 440 normalises 01 to 1), no hyphen
        ("2027-01-02", ["v2026.12.4"], "2027.1.0"),
        ("2027-01-02", ["v2027.01.0"], "2027.1.1"),
    ],
)
def test_calver_version_from_date_and_tags(release, date, tags, expected) -> None:
    assert release.calver_version(dt.date.fromisoformat(date), tags) == expected


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ("2026.11.0-rc1", "use '2026.11.0rc1'"),  # valid PEP 440, not canonical
        ("v2026.11.0", "use '2026.11.0'"),
        ("2026.11", "YYYY.M.MICRO"),
        ("2026.13.0", "YYYY.M.MICRO"),
        ("2026.11.0+local", "local version"),
        ("not-a-version", "not a valid PEP 440 version"),
    ],
)
def test_release_version_validation(release, value, fragment) -> None:
    with pytest.raises(release.ReleaseError, match=re.escape(fragment)):
        release.validate_release_version(value)


@pytest.mark.parametrize(
    "value",
    ["2026.11.0", "2026.11.1", "2027.1.0", "2026.11.0rc1", "2026.11.0.post1"],
)
def test_release_version_validation_accepts_calver(release, value) -> None:
    assert release.validate_release_version(value) == value


def test_roll_changelog_structure(release) -> None:
    text = CHANGELOG.format(unreleased=UNRELEASED_BODY)
    new = release.roll_changelog(text, "2026.11.0", dt.date(2026, 11, 7))
    head, rest = new.split("## [Unreleased]\n", 1)
    assert head == text.split("## [Unreleased]\n", 1)[0]
    # fresh, empty [Unreleased] directly followed by the new section
    assert rest.startswith("\n---\n\n## [2026.11.0] - 2026-11-07\n\n### Added\n")
    section = rest.split("## [2026.11.0] - 2026-11-07\n", 1)[1]
    section = section.split("## [0.15.0]", 1)[0]
    assert section.strip().endswith("---")
    assert section.count("---") == 1, "exactly one divider before the old release"
    assert "**Release tool (10.09):** a fix." in section
    assert (
        "[Unreleased]: https://github.com/Ce1ectric/groundfield/compare/"
        "v2026.11.0...HEAD\n"
        "[2026.11.0]: https://github.com/Ce1ectric/groundfield/releases/tag/"
        "v2026.11.0\n"
        "[0.15.0]: "
    ) in new
    # the old release sections and the roadmap are preserved verbatim
    old_sections = text.split("## [0.15.0] - 2026-07-31", 1)[1]
    assert old_sections.split("[Unreleased]: ", 1)[0] in new


@pytest.mark.parametrize(
    "body", ["", "\n\n", "_No changes yet._\n", "---\n", "_Nothing yet._\n\n---\n"]
)
def test_roll_changelog_rejects_empty_unreleased(release, body) -> None:
    text = CHANGELOG.format(unreleased=body)
    with pytest.raises(release.ReleaseError, match=r"\[Unreleased\].*empty"):
        release.roll_changelog(text, "2026.11.0", dt.date(2026, 11, 7))


def test_roll_changelog_rejects_missing_unreleased(release) -> None:
    text = CHANGELOG.format(unreleased=UNRELEASED_BODY).replace(
        "## [Unreleased]", "## Unreleased"
    )
    with pytest.raises(release.ReleaseError, match=r"\[Unreleased\]"):
        release.roll_changelog(text, "2026.11.0", dt.date(2026, 11, 7))


def test_roll_changelog_rejects_existing_section(release) -> None:
    text = CHANGELOG.format(unreleased=UNRELEASED_BODY)
    with pytest.raises(release.ReleaseError, match="already"):
        release.roll_changelog(text, "0.15.0", dt.date(2026, 11, 7))


def test_bump_pyproject_changes_only_the_project_version(release) -> None:
    text = PYPROJECT.format(version="0.15.0")
    new = release.bump_pyproject(text, "2026.11.0")
    old_data, new_data = tomllib.loads(text), tomllib.loads(new)
    assert new_data["project"].pop("version") == "2026.11.0"
    assert old_data["project"].pop("version") == "0.15.0"
    assert new_data == old_data
    assert [
        a for a, b in zip(text.splitlines(), new.splitlines(), strict=True) if a != b
    ] == ['version = "0.15.0"']


def test_bump_citation_sets_version_and_date(release) -> None:
    text = CITATION.format(version="0.15.0")
    new = release.bump_citation(text, "2026.11.0", dt.date(2026, 11, 7))
    assert 'version: "2026.11.0"\n' in new
    assert 'date-released: "2026-11-07"\n' in new
    assert "2026-07-31" not in new
    # date-released is inserted after the version when it is missing
    no_date = text.replace('date-released: "2026-07-31"\n', "")
    new = release.bump_citation(no_date, "2026.11.0", dt.date(2026, 11, 7))
    assert 'version: "2026.11.0"\ndate-released: "2026-11-07"\n' in new


# ---------------------------------------------------------------- end to end


def test_calver_release_end_to_end(repo: Repo, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    before_head = repo.head()

    result = repo.run_release("calver", "--date", "2026-11-07", cwd=elsewhere)

    assert result.returncode == 0, _output(result)
    pyproject = tomllib.loads(repo.read("pyproject.toml"))
    assert pyproject["project"]["version"] == "2026.11.0"
    assert '__version__ = "2026.11.0"' in repo.read("src/groundfield/__init__.py")
    citation = repo.read("CITATION.cff")
    assert 'version: "2026.11.0"' in citation
    assert 'date-released: "2026-11-07"' in citation
    changelog = repo.read("CHANGELOG.md")
    assert "## [Unreleased]\n\n---\n\n## [2026.11.0] - 2026-11-07\n" in changelog
    assert "compare/v2026.11.0...HEAD" in changelog
    # one commit with exactly the four release files, an annotated tag on it
    assert repo.git("rev-parse", "HEAD~1") == before_head
    assert repo.git("log", "-1", "--format=%s") == "chore(release): v2026.11.0"
    changed = set(repo.git("diff", "--name-only", "HEAD~1", "HEAD").split())
    assert changed == set(RELEASE_FILES)
    assert repo.git("cat-file", "-t", "v2026.11.0") == "tag"
    assert repo.git("rev-parse", "v2026.11.0^{commit}") == repo.head()
    assert repo.git("status", "--porcelain") == ""
    # the script tells the user how to publish, because it does not push
    assert "git push origin v2026.11.0" in _output(result)


def test_second_release_in_a_month_increments_micro(repo: Repo) -> None:
    first = repo.run_release("calver", "--date", "2026-11-07")
    assert first.returncode == 0, _output(first)
    repo.write(
        "CHANGELOG.md",
        repo.read("CHANGELOG.md").replace(
            "## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n\n- A hotfix.\n", 1
        ),
    )
    repo.commit_all("docs: changelog for the hotfix")

    second = repo.run_release("calver", "--date", "2026-11-20")

    assert second.returncode == 0, _output(second)
    assert '__version__ = "2026.11.1"' in repo.read("src/groundfield/__init__.py")
    assert {"v2026.11.0", "v2026.11.1"} <= repo.tags()


@pytest.mark.finding("10.09")
def test_set_from_outside_the_repository(repo: Repo, tmp_path: Path) -> None:
    """git must run in the repository root, not in the caller's directory."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07", cwd=elsewhere)

    assert result.returncode == 0, _output(result)
    assert repo.git("log", "-1", "--format=%s") == "chore(release): v2026.11.0"
    assert "v2026.11.0" in repo.tags()
    assert repo.git("status", "--porcelain") == ""


@pytest.mark.finding("10.09")
def test_set_from_a_subdirectory(repo: Repo) -> None:
    result = repo.run_release(
        "set", "2026.11.0", "--date", "2026-11-07", cwd=repo.path / "docs"
    )

    assert result.returncode == 0, _output(result)
    assert "v2026.11.0" in repo.tags()
    assert repo.git("status", "--porcelain") == ""


@pytest.mark.finding("10.09")
def test_empty_changelog_aborts_before_any_write(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, unreleased="_No changes yet._\n\n")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "[Unreleased]" in _output(result)
    _assert_untouched(repo, before, head, tags)


@pytest.mark.finding("10.09")
def test_existing_tag_aborts_before_any_write(repo: Repo) -> None:
    repo.git("tag", "-a", "v2026.11.0", "-m", "pre-existing")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "v2026.11.0" in _output(result)
    assert "exists" in _output(result)
    _assert_untouched(repo, before, head, tags)


@pytest.mark.finding("10.09")
def test_detached_head_aborts_before_any_write(repo: Repo) -> None:
    repo.git("checkout", "-q", "--detach")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "branch" in _output(result)
    _assert_untouched(repo, before, head, tags)


@pytest.mark.finding("10.09")
def test_prerelease_can_be_finalised(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, version="2026.11.0rc1", tags=("v2026.11.0rc1",))

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    assert '__version__ = "2026.11.0"' in repo.read("src/groundfield/__init__.py")
    assert "v2026.11.0" in repo.tags()


def test_calver_after_a_prerelease_of_the_same_month(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, version="2026.11.0rc1", tags=("v2026.11.0rc1",))

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    assert "v2026.11.0" in repo.tags()


def test_never_pushes(repo: Repo, tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(remote)],
        env=repo.env,
        check=True,
    )
    repo.git("remote", "add", "origin", str(remote))
    repo.git("push", "-q", "-u", "origin", "main")
    remote_head = repo.git("ls-remote", "origin", "refs/heads/main").split()[0]

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    assert repo.git("ls-remote", "--tags", "origin") == ""
    assert repo.git("ls-remote", "origin", "refs/heads/main").split()[0] == (
        remote_head
    )


def test_dry_run_changes_nothing(repo: Repo) -> None:
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07", "--dry-run")

    assert result.returncode == 0, _output(result)
    out = _output(result)
    assert "2026.11.0" in out
    assert "v2026.11.0" in out
    assert "## [2026.11.0] - 2026-11-07" in out
    _assert_untouched(repo, before, head, tags)


def test_dry_run_still_validates(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, unreleased="\n")

    result = repo.run_release("calver", "--date", "2026-11-07", "--dry-run")

    assert result.returncode != 0
    assert "[Unreleased]" in _output(result)


@pytest.mark.parametrize("change", ["modified", "staged", "untracked"])
def test_dirty_tree_aborts(repo: Repo, change: str) -> None:
    if change == "untracked":
        repo.write("src/groundfield/new_module.py", "x = 1\n")
    else:
        repo.write("docs/index.md", "# changed\n")
        if change == "staged":
            repo.git("add", "docs/index.md")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "clean" in _output(result)
    assert repo.snapshot() == before
    assert (repo.head(), repo.tags()) == (head, tags)


def test_version_drift_aborts(repo: Repo) -> None:
    repo.write("CITATION.cff", CITATION.format(version="0.14.1"))
    repo.commit_all("drift")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode != 0
    out = _output(result)
    assert "CITATION.cff" in out and "0.14.1" in out
    _assert_untouched(repo, before, head, tags)


@pytest.mark.parametrize(
    "version", ["0.14.0", "0.15.0", "2026.11", "2026.11.0-rc1", "banana"]
)
def test_set_rejects_invalid_or_lower_versions(repo: Repo, version: str) -> None:
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("set", version, "--date", "2026-11-07")

    assert result.returncode != 0
    _assert_untouched(repo, before, head, tags)


def test_calver_must_sort_above_the_current_version(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, version="2026.12.0", tags=("v2026.12.0",))
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "2026.12.0" in _output(result)
    _assert_untouched(repo, before, head, tags)


def test_hardcoded_version_in_claude_md_aborts(repo: Repo) -> None:
    repo.write("CLAUDE.md", '# notes\n__version__ = "0.15.0"\n')  # gitignored
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "hard-code" in _output(result)
    _assert_untouched(repo, before, head, tags)


def test_unmerged_changelog_fragments_abort(repo: Repo) -> None:
    repo.write("changelog.d/README.md", "# fragments\n")
    repo.write("changelog.d/core1.md", "### Added\n\n- Something.\n")
    repo.commit_all("fragment")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "changelog.d/core1.md" in _output(result)
    _assert_untouched(repo, before, head, tags)


def test_failed_commit_restores_the_files(repo: Repo) -> None:
    hook = repo.path / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'hook says no' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode != 0
    assert "hook says no" in _output(result)
    _assert_untouched(repo, before, head, tags)


def test_crlf_files_keep_their_line_endings(tmp_path: Path) -> None:
    """A Windows checkout (CRLF) is released without rewriting line endings."""
    repo = _make_repo(tmp_path)
    for rel in RELEASE_FILES:
        text = repo.read(rel)
        (repo.path / rel).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    repo.commit_all("crlf")

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    for rel in RELEASE_FILES:
        raw = (repo.path / rel).read_bytes()
        assert b"\n" not in raw.replace(b"\r\n", b""), f"{rel} has LF endings"
    assert (
        b'__version__ = "2026.11.0"\r\n'
        in (repo.path / "src/groundfield/__init__.py").read_bytes()
    )
    changelog = (repo.path / "CHANGELOG.md").read_bytes()
    assert b"## [Unreleased]\r\n\r\n---\r\n\r\n## [2026.11.0] - 2026-11-07\r\n" in (
        changelog
    )


@pytest.mark.parametrize("args", [["calver", "--date", "2026-13-01"], [], ["bump"]])
def test_usage_errors_exit_with_status_2(repo: Repo, args: list[str]) -> None:
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release(*args)

    assert result.returncode == 2
    _assert_untouched(repo, before, head, tags)


def test_release_of_the_real_project_files(tmp_path: Path) -> None:
    """The tool handles the repository's own pyproject, CITATION and CHANGELOG."""
    repo = _make_repo(tmp_path)
    for rel in RELEASE_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        if rel == "CHANGELOG.md":
            # Right after a release the real [Unreleased] block is empty and
            # the tool refuses it; give the copy one entry so that the test
            # runs on the tag as well (the release workflow tests the tag).
            head, marker, tail = text.partition("\n## [Unreleased]\n")
            if "\n### " not in tail.split("\n## [", 1)[0]:
                tail = "\n### Fixed\n\n- A placeholder entry of this test.\n" + tail
            text = head + marker + tail
        repo.write(rel, text)
    repo.commit_all("real files")
    old_pyproject = tomllib.loads(repo.read("pyproject.toml"))
    old_changelog = repo.read("CHANGELOG.md")
    current = old_pyproject["project"]["version"]

    result = repo.run_release("calver", "--date", "2099-12-31")

    assert result.returncode == 0, _output(result)
    new_pyproject = tomllib.loads(repo.read("pyproject.toml"))
    assert new_pyproject["project"].pop("version") == "2099.12.0"
    assert old_pyproject["project"].pop("version") == current
    assert new_pyproject == old_pyproject
    assert '__version__ = "2099.12.0"' in repo.read("src/groundfield/__init__.py")
    assert 'version: "2099.12.0"' in repo.read("CITATION.cff")
    new_changelog = repo.read("CHANGELOG.md")
    assert "## [2099.12.0] - 2099-12-31\n" in new_changelog
    # the old release sections and the roadmap are untouched
    old_tail = old_changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[1]
    assert old_tail.split("\n[Unreleased]: ", 1)[0] in new_changelog
    assert repo.git("status", "--porcelain") == ""


# ---------------------------------------------------------------- remote state


def _with_origin(repo: Repo, tmp_path: Path, *, push: bool = True) -> Path:
    """Give ``repo`` a local bare ``origin`` (no network) and push main to it."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)],
        env=repo.env,
        check=True,
    )
    repo.git("remote", "add", "origin", str(origin))
    if push:
        repo.git("push", "-q", "-u", "origin", "main", "--tags")
    return origin


def _clone(origin: Path, path: Path, env: dict[str, str]) -> Repo:
    subprocess.run(["git", "clone", "-q", str(origin), str(path)], env=env, check=True)
    return Repo(path, env)


@pytest.mark.finding("A1-11", "10.09")
def test_a_clone_behind_its_upstream_is_refused(repo: Repo, tmp_path: Path) -> None:
    """The release was cut and pushed from another clone: the stale clone
    must not compute the same version again and create a second, different
    tag (the push would only fail later)."""
    origin = _with_origin(repo, tmp_path)
    laptop = _clone(origin, tmp_path / "laptop", repo.env)
    done = repo.run_release("calver", "--date", "2026-11-07")
    assert done.returncode == 0, _output(done)
    repo.git("push", "-q", "origin", "main", "v2026.11.0")

    before, head, tags = laptop.snapshot(), laptop.head(), laptop.tags()
    stale = laptop.run_release("calver", "--date", "2026-11-20")

    assert stale.returncode == 1, _output(stale)
    assert "behind" in _output(stale) and "origin/main" in _output(stale)
    _assert_untouched(laptop, before, head, tags)


@pytest.mark.finding("A1-11", "10.09")
def test_a_diverged_branch_is_refused(repo: Repo, tmp_path: Path) -> None:
    origin = _with_origin(repo, tmp_path)
    other = _clone(origin, tmp_path / "other", repo.env)
    other.write("docs/other.md", "# other\n")
    other.commit_all("other work")
    other.git("push", "-q", "origin", "main")
    repo.write("docs/local.md", "# local\n")
    repo.commit_all("local work")

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 1
    assert "behind" in _output(result)


@pytest.mark.finding("A1-11", "10.09")
def test_a_tag_on_the_remote_counts_like_a_local_one(
    repo: Repo, tmp_path: Path
) -> None:
    """A tag that exists only on the remote (pushed from another clone)."""
    origin = _with_origin(repo, tmp_path)
    other = _clone(origin, tmp_path / "other", repo.env)
    other.git("tag", "-a", "v2026.11.0", "-m", "v2026.11.0")
    other.git("push", "-q", "origin", "v2026.11.0")
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    explicit = repo.run_release("set", "2026.11.0", "--date", "2026-11-07")
    assert explicit.returncode == 1
    assert "v2026.11.0" in _output(explicit) and "remote" in _output(explicit)
    _assert_untouched(repo, before, head, tags)

    derived = repo.run_release("calver", "--date", "2026-11-07")
    assert derived.returncode == 0, _output(derived)
    assert "v2026.11.1" in repo.tags()


@pytest.mark.finding("A1-11")
def test_an_unreachable_remote_aborts(repo: Repo, tmp_path: Path) -> None:
    origin = _with_origin(repo, tmp_path)
    shutil.rmtree(origin)
    before, head, tags = repo.snapshot(), repo.head(), repo.tags()

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 1
    assert "origin" in _output(result)
    _assert_untouched(repo, before, head, tags)


@pytest.mark.finding("A1-11")
def test_a_branch_without_upstream_is_refused_when_a_remote_exists(
    repo: Repo, tmp_path: Path
) -> None:
    _with_origin(repo, tmp_path, push=False)

    result = repo.run_release("calver", "--date", "2026-11-07", "--dry-run")

    assert result.returncode == 1
    assert "upstream" in _output(result)


def test_a_branch_ahead_of_its_upstream_is_released(repo: Repo, tmp_path: Path) -> None:
    _with_origin(repo, tmp_path)
    repo.write("docs/more.md", "# not pushed yet\n")
    repo.commit_all("local work")

    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    assert "v2026.11.0" in repo.tags()


def test_the_remote_checks_change_nothing(repo: Repo, tmp_path: Path) -> None:
    """No fetch: remote-tracking refs and FETCH_HEAD stay as they were."""
    origin = _with_origin(repo, tmp_path)
    other = _clone(origin, tmp_path / "other", repo.env)
    other.write("docs/other.md", "# other\n")
    other.commit_all("other work")
    other.git("push", "-q", "origin", "main")
    tracking = repo.git("rev-parse", "refs/remotes/origin/main")

    repo.run_release("calver", "--date", "2026-11-07", "--dry-run")

    assert repo.git("rev-parse", "refs/remotes/origin/main") == tracking
    assert not (repo.path / ".git" / "FETCH_HEAD").exists()


def test_a_repository_without_remote_releases_with_a_note(repo: Repo) -> None:
    result = repo.run_release("calver", "--date", "2026-11-07")

    assert result.returncode == 0, _output(result)
    assert "no git remote" in _output(result)


@pytest.mark.finding("A1-11")
def test_an_upstream_branch_missing_on_the_remote_aborts(
    repo: Repo, tmp_path: Path
) -> None:
    origin = _with_origin(repo, tmp_path)
    subprocess.run(
        ["git", "update-ref", "-d", "refs/heads/main"],
        cwd=origin,
        env=repo.env,
        check=True,
    )

    result = repo.run_release("calver", "--date", "2026-11-07", "--dry-run")

    assert result.returncode == 1
    assert "does not exist on the remote" in _output(result)
