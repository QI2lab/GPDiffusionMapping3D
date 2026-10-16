# Contributing

Keep changes consistent with the physical model and include independent
mathematical or physical checks. Preserve the original license, copyright and
attribution notices.

## Tests

```bash
uv sync --group dev
uv run --no-sync python -m pytest tests/cpu --strict-markers -q \
  -k 'not test_independent_3d_chains_converge'
```

Tests use `unit` and `integration` scientific categories, with shared fixtures
under `tests/`. Deprecation warnings fail tests and must be fixed. Full
convergence runs are requested on PRs ready to merge; see [GitHub CI](github-ci.md).

## Documentation

Write concise Markdown pages with current usage, relative links, fenced code
blocks and tables where useful. Add pages to the navigation in `mkdocs.yml`.

```bash
uv sync --group dev --group docs
uv run --no-sync mkdocs build --strict
uv run --no-sync mkdocs serve
```

Open `http://127.0.0.1:8000/` for the local preview.
