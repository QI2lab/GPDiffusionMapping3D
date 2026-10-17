# GPDiffusionMapping

## Introduction

Infer spatial diffusion maps from linked particle trajectories using Gaussian
processes. This project extends the method described in the
[original paper](https://doi.org/10.1016/j.bpr.2024.100194) and implemented in
[LabPresse/GPDiffusionMapping](https://github.com/LabPresse/GPDiffusionMapping).
Example and experimental datasets are in `data/`.

Original software: Copyright (c) 2024 Pressé Lab, under the [MIT license](LICENSE).
See [NOTICE](NOTICE) for attribution and dataset provenance.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dpshepherd/GPDiffusionMapping.git
cd GPDiffusionMapping
uv sync
```

Inference uses CPU float64 with NumPy/Numba; no GPU installation is needed.

## Usage

Provide linked tracks as CSV or Parquet. For 3D, the default columns are
`particle #`, `z`, `y`, `x`, and `timestamp`, with positions in nanometers and
timestamps in seconds.

```bash
uv run --no-sync gpdiffusion-map tracks.csv
uv run --no-sync gpdiffusion-map tracks.parquet --output-dir results
uv run --no-sync python -m gpdiffusionmapping --help
```

The command saves the inferred map, samples and metadata. Use `--help` for
column overrides, unit conversion and legacy 2D input options.

## Changes from Original

- Timestamped 3D inference alongside the original 2D method.
- Improved 3D sampling with blocked Metropolis proposals and longer warmup.
- Optimized CPU calculations and modern dependency management with uv.
- A command-line interface for CSV and Parquet input.
- Scientific unit/integration tests and GitHub Actions CI.

## Contributing

Open issues and pull requests against
[dpshepherd/GPDiffusionMapping](https://github.com/dpshepherd/GPDiffusionMapping),
with PRs targeting this repository's `main` branch.

```bash
uv sync --group dev
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```

Keep changes consistent with the physical model and include scientific checks.
Preserve the original copyright, license and attribution notices.
The shorter CI suite runs on PRs and pushes to `main`/`master`. Add
`run-convergence` to a non-draft PR ready to merge to request full convergence
checks. Further validation and
implementation details are in [`docs/`](docs/).
