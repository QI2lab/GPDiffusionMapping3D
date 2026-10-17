# Contributing

Keep changes consistent with the physical model and support numerical changes
with independent mathematical expectations or physical simulations. Preserve
the original license, copyright and attribution notices.

## Scientific checks

```bash
uv sync --group dev
uv run --no-sync python -m pytest tests/cpu --strict-markers -q \
  -k 'not test_independent_3d_chains_converge'
```

The suite uses only `unit` and `integration` scientific categories. Shared
fixtures and simulation helpers live under `tests/`; application code lives
under `src/gpdiffusionmapping/`.

The shorter CI suite runs on PRs and pushes to `main`/`master`. Full convergence
is requested by adding `run-convergence` to a non-draft PR ready to merge. It
has no nightly schedule. See the [CI guide](github-ci.md) for the full protocol
and how to request a fresh run on the final revision.

## Documentation

Write topic pages in Markdown, with clear headings, relative links, fenced
code blocks and tables for comparisons. Add new pages to `mkdocs.yml` navigation.
Keep user-facing instructions separate from dated validation evidence.

```bash
uv sync --group dev --group docs
uv run --no-sync mkdocs build --strict
uv run --no-sync mkdocs serve
```

The Material theme provides light/dark palettes, search and code-copy buttons,
following the documentation style of `merfish3d-analysis`. JSON measurements
remain machine-readable evidence linked from the validation index.
