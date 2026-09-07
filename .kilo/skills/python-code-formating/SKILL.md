---
name: python-code-formating
description: Python code formatting with uvx ruff@0.15.20. Fast, Black-compatible formatter. Use after any Python modification to format and lint.
---

# uvx ruff formatting

Format and lint Python code with `uvx ruff@0.15.20`. Use after any Python file
modification (see `Apply contribution rules` in `AGENTS.md`).

## Why uvx and not uv run

We use `uvx` instead of `uv run` to invoke ruff. `uvx` is the `uv` equivalent of
`pipx run`: it runs a tool inside an isolated, ephemeral environment created on the fly
and does not touch the project's own dependency resolution.

`uv run ruff` resolves and installs the project's full dependency tree first, which can
fail loudly even when all we want is the formatter. In this repo resolution errors out for
unsupported Python versions (see the unsatisfiable `tensorflow` marker in
`pyproject.toml`), so `uv run ruff` is unreliable here. `uvx` downloads ruff once (cached
afterwards) and runs it in the project directory so your `pyproject.toml` ruff config is
picked up.

Prerequisite: `uvx` ships with `uv`, so `uv` must be installed.

The version is pinned to `0.15.20` so formatting stays reproducible.

## Formatting workflow

Run these two commands in order, then fix any errors the check raises, and repeat until
the check passes with no errors:

```bash
uvx ruff@0.15.20 format
uvx ruff@0.15.20 check --fix
```

After a Python file modification, you may run these commands on the changed files only:

```bash
uvx ruff@0.15.20 format <changed file(s)>
uvx ruff@0.15.20 check --fix <changed file(s)>
```

`check --fix` fixes what it can automatically; fix the rest by hand, then re-run both
commands until the check reports no errors. That is the whole workflow.
