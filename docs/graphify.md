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

The `groundfield` source tree spans the solver backends and their
shared pipeline, the coupling and post-processing layers, the
generators, the ADRs, the tutorials and a large test suite. Asking an
AI assistant a structural
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
search, which needs far fewer tokens per question than reading the
files (`graphify benchmark` measures it on this repository).

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

The LLM backend of the semantic pass is selected from whichever
provider environment variable is set (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …). Set the key only for the build command, not
globally in the shell profile, so that other tools do not pick it up:

```bash
ANTHROPIC_API_KEY=sk-ant-... graphify extract . --backend claude
```

For a fully offline alternative, use a local Ollama model
(`--backend ollama --model qwen2.5-coder:14b`) — no API key, no
billing, slightly weaker semantic clustering.

## Build, update, query

| Phase | When | Cost |
| --- | --- | --- |
| **Build** (`graphify extract`) | once, and after larger refactors | LLM tokens for the semantic pass (one-off) |
| **Update** (`graphify update`) | after small code changes | none: AST only, no LLM call |
| **Query** | while coding | the assistant reads the pre-built graph as ordinary files |

Day-to-day questions do not need `graphify query` (which calls the
LLM again): after `graphify claude install` (below) the coding
assistant reads `graphify-out/GRAPH_REPORT.md` and `graph.json`
directly.

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

From this point on the assistant reads
`graphify-out/GRAPH_REPORT.md` and `graph.json` as ordinary files;
`graphify query` is only needed outside the assistant.

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
  corpus size.
- LaTeX and Beamer slides are **not** parsed by the bundled
  Tree-sitter grammars; the Markdown pages and ADRs in `docs/` are.
- Jupyter notebooks (`.ipynb`) are JSON; their code cells are
  visible to the Python grammar via the JSON wrapper, but the
  Markdown narrative is treated as plain text.
- The graph is a snapshot. After non-trivial refactors re-run
  `extract` rather than `update`, otherwise stale nodes can
  survive the AST pass.

## Why it is not in `pyproject.toml`

`graphifyy` pulls in many Tree-sitter grammars and is not needed to
build, test or document the package, so it stays out of the lock file
and the dependency groups. It belongs to the developer's environment,
installed with pipx.
