# Knowledge graph for AI coding assistants (graphify)

`groundfield` documents an optional, hand-driven workflow for
turning the source tree, ADRs and Markdown docs into a queryable
knowledge graph using
[`graphify`](https://github.com/safishamsi/graphify) (PyPI:
`graphifyy`, MIT). The graph is consumed by AI coding assistants
(Claude Code, Cursor, Codex, …) so they can answer structural
questions over the codebase without reading every file into
context.

## Why this exists

The `groundfield` source tree spans eight solver backends
(`image`, `image_2layer`, `image_nlayer`, `cim`, `mom`,
`mom_sommerfeld`, `bem`, `fem`), four ADRs, a notebook suite and
a growing test suite. Asking an AI assistant a structural
question — *"which backends consume the distributed-conductor
topology?"* or *"where is the rho-f fit produced?"* — used to
require loading hundreds of kilobytes of source into the model
context.

`graphify` extracts a static graph

```text
nodes  = symbols (functions, classes, modules, ADR sections, …)
edges  = call / import / reference / cites / …
```

once via Tree-sitter (locally, AST only) plus one optional
LLM-driven semantic pass. Queries are answered with a bounded
breadth-first traversal over the graph instead of full-corpus
search. The published benchmark figure on mixed corpora is on
the order of a 70× per-query token reduction at comparable
answer quality.

This is **tooling**, not a runtime feature of `groundfield`. It
does not change the public API or the solver behaviour.

## Installation

`graphifyy` pulls in roughly two dozen Tree-sitter language
packages (C, Go, Java, Rust, …) that have no place in
`groundfield`'s lock file. It is therefore installed
**independently of the project venv** via
[`pipx`](https://pipx.pypa.io/):

```bash
pipx install graphifyy
graphify --help
```

The default LLM backend is auto-selected from whichever provider
environment variable is set (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …). The recommended setup for this project is
the Anthropic API for the build phase only:

```bash
# Set ANTHROPIC_API_KEY only for the build / update commands.
# Do NOT export it globally in your shell rc, otherwise Claude
# Code will start charging your API credits instead of using your
# Pro subscription quota for normal coding sessions.
ANTHROPIC_API_KEY=sk-ant-... graphify extract . --backend claude
```

For a fully offline alternative, use a local Ollama model
(`--backend ollama --model qwen2.5-coder:14b`) — no API key, no
billing, slightly weaker semantic clustering.

## Build vs. query — two phases with different billing

Graphify has two operational phases that are billed very
differently. Understanding the split is what turns this from an
expensive tool into a near-free one for Pro / Max subscribers:

| Phase | When | Cost |
| --- | --- | --- |
| **Build** (`graphify extract`) | once, plus after larger refactors | one-shot LLM tokens for the semantic pass; ≈ 0.02–0.50 USD on `groundfield` (Haiku 4.5 → Sonnet 4.6) |
| **Update** (`graphify update`) | after small code changes | **0** — AST only, no LLM call |
| **Query** (via Claude Code with the installed hook) | every day while coding | counts against your **Claude Pro / Max subscription**, not API credits |

The trick is that the daily querying does **not** go through
`graphify query` (which would re-bill API tokens). Instead, the
`graphify claude install` step (see below) wires Claude Code
itself to read the pre-built graph through its own `Read` tool,
so the assistant's reasoning is paid by the subscription quota
and `graphify-out/graph.json` is just a static input.

## Manual on-demand workflow

`groundfield` does not install git hooks or a `watch` daemon —
the graph is rebuilt by hand when you need it.

### Initial extraction (full, with semantic LLM pass)

```bash
cd /path/to/groundfield
graphify extract . --backend claude
poetry run python scripts/generate_graphify_report.py
```

The `extract` call writes

```text
graphify-out/graph.json                  # the queryable graph
graphify-out/.graphify_analysis.json     # communities, gods, surprises
graphify-out/manifest.json               # per-file index
graphify-out/cache/ast/                  # AST cache (per-file content hash)
```

`graphify` 0.7.x no longer writes the human-readable
`GRAPH_REPORT.md` itself, even though the
`graphify claude install` hook and the auto-installed `CLAUDE.md`
section both reference it. The companion script
`scripts/generate_graphify_report.py` reconstructs the report from
`.graphify_analysis.json` and `graph.json` (no network, no API call)
so that AI coding assistants find the expected entry point. Run it
once after every `graphify extract`.

The `graphify-out/` directory is in `.gitignore`; it is local to
your checkout and depends on the chosen backend / model.

### Incremental refresh after code changes (cheap, AST only)

```bash
graphify update .
poetry run python scripts/generate_graphify_report.py
```

`update` re-extracts the AST without calling the LLM, which is
sufficient when only function bodies or signatures changed.
After larger refactors (new modules, removed code) re-run
`extract`. Re-run the report script in either case so that
`GRAPH_REPORT.md` stays in sync with the new graph.

After refactors that **delete** code, `graphify extract` may
keep the previous (larger) graph cached. Force a clean rebuild
with

```bash
rm -rf graphify-out/
ANTHROPIC_API_KEY=sk-ant-... GRAPHIFY_FORCE=1 \
    graphify extract . --backend claude --model claude-haiku-4-5-20251001
poetry run python scripts/generate_graphify_report.py
```

### Querying

```bash
graphify query "what backends consume a distributed-conductor topology?"
graphify query "where is the rho-f fit produced?"
graphify query "how does image_2layer handle the K-series truncation?"
graphify path "TwoLayerSoil" "compare_engines"
graphify explain "image_2layer.solve"
```

Useful flags:

- `--budget N` &mdash; cap the answer at `N` tokens (default 2000)
- `--graph PATH` &mdash; query a graph other than `./graphify-out/graph.json`
- `--dfs` &mdash; depth-first instead of breadth-first traversal

### Token-reduction benchmark

```bash
graphify benchmark
```

Prints the per-query token cost against a naive full-corpus
baseline, on this very repository.

## Recommended: Claude Code integration via `graphify claude install`

This is the step that pays for itself within a few sessions.
After the initial `extract`, run

```bash
graphify claude install
```

once per checkout. It does two things:

1. Appends a graphify section to your repo-local `CLAUDE.md`
   instructing Claude Code to consult `graphify-out/GRAPH_REPORT.md`
   before answering architecture-level questions.
2. Registers a `PreToolUse` hook in `.claude/settings.json` that
   fires before every `Glob` and `Grep` call and reminds Claude to
   navigate via the graph's "god nodes" and cluster structure
   instead of grepping raw files.

From this point on, **stop calling `graphify query` interactively**.
Just talk to Claude Code as usual — it will read
`graphify-out/GRAPH_REPORT.md` and `graph.json` as ordinary file
inputs through its own `Read` tool. That puts the cost of every
day-to-day question on your Claude Pro / Max subscription, not on
your API credit.

Reverse the integration at any time with

```bash
graphify claude uninstall
```

`groundfield`'s repo-local `CLAUDE.md` is in `.gitignore`, so the
hook configuration stays per machine and does not propagate to
other contributors. The same family of installers exists for
non-Anthropic clients: `graphify cursor install`,
`graphify codex install`, `graphify gemini install`, etc.

## Cross-repo graphs (optional)

`groundfield`, `groundinsight` and `groundmeas` share a real
interface (rho-f export, measurement-geometry import). A single
combined graph makes cross-repo queries possible:

```bash
graphify extract ../groundinsight --backend claude --global --as groundinsight
graphify extract ../groundmeas    --backend claude --global --as groundmeas
graphify extract .                --backend claude --global --as groundfield
graphify global list
```

The combined graph lives at `~/.graphify/global-graph.json` and
is independent of any single repository.

## Limitations

- The semantic LLM pass costs API tokens proportional to the
  corpus size. For `groundfield` today (≈ 10 kLoC plus docs and
  ADRs) the initial `extract` is on the order of a few cents at
  current Anthropic pricing.
- LaTeX and Beamer slides are **not** parsed by the bundled
  Tree-sitter grammars. Markdown ADRs in `docs/adr/` are picked
  up; the LaTeX dissertation deck in
  `~/Nextcloud/Forschung/Dissertation/arbeitsordner/` is not.
- Jupyter notebooks (`.ipynb`) are JSON; their code cells are
  visible to the Python grammar via the JSON wrapper, but the
  Markdown narrative is treated as plain text.
- The graph is a snapshot. After non-trivial refactors re-run
  `extract` rather than `update`, otherwise stale nodes can
  survive the AST pass.

## Why it is not in `pyproject.toml`

The same separation as for `black` / `ruff` / `mypy` in many
open-source Python projects: tools that operate on the source
tree are kept out of the library's dependency graph so that
downstream users of `groundfield` are not forced to install
them. `graphifyy` belongs to the developer's environment, not
to the library's runtime.
