# GPDiffusionMapping

## Introduction

Infer spatial diffusion maps from linked particle trajectories using Gaussian
processes. This project extends the method described in the
[DiffMAP-GP: Continuous 2D diffusion maps from particle trajectories without data binning using Gaussian processes](https://doi.org/10.1016/j.bpr.2024.100194)
by Kumar et al., implemented in
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

Provide linked tracks as CSV or Parquet. Default columns are `particle_id`,
`x`, `y`, and `timestamp`, plus `z` for 3D. Positions are in nanometers and
timestamps in seconds. Each cell is inferred separately with its own GP grid.

- **Directory:** each CSV/Parquet file directly inside it represents one cell.
- **Single file:** a `cell_id` column is required; the CLI automatically groups
  observations by cell before linking within-particle displacements.

```bash
# 2D: one file per cell
uv run --no-sync gpdiffusion-map cells_2d/ --dimensions 2 --output-dir maps_2d

# 3D: all cells in one CSV (Parquet also works)
uv run --no-sync gpdiffusion-map all_cells.csv --dimensions 3 --output-dir maps_3d

# Legacy 2D files without timestamps: provide seconds per frame
uv run --no-sync gpdiffusion-map legacy_cells/ --dimensions 2 \
  --x-col xPos --y-col yPos --delta-t 0.03333333333333333
```

Each cell gets a `cell_<identifier>/` output folder containing its map,
samples, metadata and sampling log. Directory inputs use filename stems as
cell identifiers; single-file inputs use `cell_id` values. Special characters
are URL-encoded in folder names. Particle IDs may repeat across cells.

Use `uv run --no-sync gpdiffusion-map --help` for column overrides, including
`--cell-col` and `--particle-col`, and unit conversion. Existing files using
`particle #` can use `--particle-col 'particle #'`.

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
checks. Further validation and implementation details are in [`docs/`](docs/).
