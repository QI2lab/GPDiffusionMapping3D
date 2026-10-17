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

Provide linked tracks as CSV or Parquet. For 2D, use columns `particle #`,
`x`, `y`, and `timestamp`; for 3D, also include `z`. Positions are in nanometers
and timestamps in seconds. Both formats work with either dimension.

```bash
# 2D inference
uv run --no-sync gpdiffusion-map tracks_2d.csv --dimensions 2 --output-dir results/2d

# 3D inference
uv run --no-sync gpdiffusion-map tracks_3d.parquet --dimensions 3 --output-dir results/3d

# Legacy 2D input without timestamps: specify seconds per frame
uv run --no-sync gpdiffusion-map legacy.csv --dimensions 2 \
  --x-col xPos --y-col yPos --delta-t 0.03333333333333333
```

Each output directory contains the inferred map, samples and metadata.
Use `uv run --no-sync gpdiffusion-map --help` for column overrides and unit
conversion.

**One file per cell:** put the cell files in `cells/` and run this Bash loop.
Use unique filename stems and change `--dimensions 3` to `2` for 2D data.

```bash
for file in cells/*.csv cells/*.parquet; do
  [ -f "$file" ] || continue
  name="${file##*/}"
  uv run --no-sync gpdiffusion-map "$file" --dimensions 3 \
    --output-dir "maps/${name%.*}" || break
done
```

**One file containing many cells:** include a `cell_id` column and split it
before inference. Save this example as `batch_cells.py`, set the input path
and dimensions, then run `uv run --no-sync python batch_cells.py`:

```python
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote

import pandas as pd

source = Path("all_cells.parquet")  # .csv also works
dimensions = "3"  # "2" for 2D
tracks = (
    pd.read_parquet(source)
    if source.suffix == ".parquet"
    else pd.read_csv(source, dtype={"particle #": "string", "cell_id": "string"})
)
if tracks["cell_id"].isna().any():
    raise ValueError("Every observation must have a cell_id")

with TemporaryDirectory() as temporary:
    for cell_id, rows in tracks.groupby("cell_id", sort=False):
        name = "cell_" + quote(str(cell_id), safe="")
        path = Path(temporary) / f"{name}.csv"
        rows.to_csv(path, index=False)
        subprocess.run([
            "uv", "run", "--no-sync", "gpdiffusion-map", str(path),
            "--dimensions", dimensions, "--output-dir", str(Path("maps") / name),
        ], check=True)
```

This produces one map per cell under `maps/`. Particle IDs may repeat between
cells: grouping by `cell_id` keeps their tracks separate. Cells are inferred
independently, with a separate spatial grid for each cell.

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
