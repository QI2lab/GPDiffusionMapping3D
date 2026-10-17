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
`x`, `y`, and `timestamp`, plus `z` for 3D. Positions are in microns (µm),
timestamps in seconds, and diffusion coefficients in µm²/s. Each cell is
inferred separately with its own GP grid.

- **Directory:** each CSV/Parquet file directly inside it represents one cell.
- **Single file:** a `cell_id` column is required; the CLI automatically groups
  observations by cell before linking within-particle displacements.

Example single-file **2D** input (`cells_2d.csv`):

```csv
cell_id,particle_id,x,y,timestamp
cell_a,1,0.10,0.20,0.00
cell_a,1,0.12,0.23,0.01
cell_b,1,1.10,1.20,0.00
cell_b,1,1.13,1.22,0.01
```

Example single-file **3D** input (`cells_3d.csv`):

```csv
cell_id,particle_id,z,y,x,timestamp
cell_a,1,0.30,0.20,0.10,0.00
cell_a,1,0.32,0.23,0.12,0.01
cell_b,1,1.30,1.20,1.10,0.00
cell_b,1,1.33,1.22,1.13,0.01
```

These illustrate the columns; inference needs many more observations.
Parquet inputs use the same columns.

```bash
# 2D: one file per cell
uv run --no-sync gpdiffusion-map cells_2d/ --dimensions 2 --output-dir maps_2d

# 3D: all cells in one CSV (Parquet also works)
uv run --no-sync gpdiffusion-map cells_3d.csv --dimensions 3 --output-dir maps_3d

# 2D: all cells in one CSV
uv run --no-sync gpdiffusion-map cells_2d.csv --dimensions 2 --output-dir maps_2d

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
For input positions in nanometers, use `--scale 0.001` to convert to microns.
Saved arrays and figures both use µm and µm²/s; array keys use the `um` suffix.

## Changes from Original

- Timestamped 3D inference alongside the original 2D method.
- Micron-based inputs, numerical outputs and figures.
- Improved 3D sampling with blocked Metropolis proposals and longer warmup.
- Optimized CPU calculations and modern dependency management with uv.
- A command-line interface for CSV and Parquet input.
- Scientific unit/integration tests and GitHub Actions CI.

## Contributing

```bash
uv sync --group dev
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```

Keep changes consistent with the physical model and include scientific checks.
Preserve the original copyright, license and attribution notices.
The shorter CI suite runs on PRs and pushes to `main`/`master`. Add
`run-convergence` to a non-draft PR ready to merge to request full convergence
checks. Further validation and implementation details are in [`docs/`](docs/).
