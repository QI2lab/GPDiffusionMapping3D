# Scientific tests and validation tools

`cpu/` contains the collected unit and integration tests. Shared fixtures live
in `conftest.py`; `support/` contains their simulation oracles, standalone
validation protocols, benchmarks and historical diagnostic plots.

Install the package with `uv sync --group dev` before running these tools.
Application code lives in `src/gpdiffusionmapping/`; tests import that package.

Run the shorter scientific suite from the repository root:

```bash
uv run --no-sync python -m pytest tests/cpu --strict-markers -q \
  -k 'not test_independent_3d_chains_converge'
```

Full convergence remains an explicit pre-merge request via the
`run-convergence` PR label. See [the CI guide](../docs/github-ci.md).

Standalone development tools can run as modules, for example:

```bash
uv run --no-sync python -m tests.support.profile_cpu --help
uv run --no-sync python -m tests.support.run_sampling_repair --help
uv run --no-sync python -m tests.support.validate_migration
```

The former `scripts/*.py` tools now have the same filenames under
`tests/support/`. Older reports retain their original command paths as a
record of execution; substitute `tests/support/` when reproducing a retained
tool. Historical CPU/GPU plots consume archived results and do not run CUDA.
For end-user inference, use `gpdiffusion-map`; no development tools are needed.
