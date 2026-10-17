# Inference workflow

## Prepare linked tracks

Input observations must already be linked into particle trajectories. Default
columns are `particle_id`, `x`, `y`, and `timestamp`, with `z` required for 3D.
Positions are in µm and timestamps are numeric seconds. Each displacement uses
its actual positive elapsed time. Timestamp is not a fourth spatial coordinate:
the inferred field is static, `D(x, y)` or `D(z, y, x)`.

### One file per cell

Put CSV or Parquet files directly inside a directory:

```text
cells/
├── cell_0000.parquet
├── cell_0001.parquet
└── cell_0002.csv
```

The integer after the underscore identifies the cell. IDs must be unique,
nonnegative 64-bit integers. Files can mix formats; traversal is not recursive.
For longer names, the last underscore-number token supplies the ID. A
`cell_id` column is optional in this mode, but must match the filename if present.

### One combined file

A combined CSV or Parquet file requires an integer `cell_id` column. Particle
IDs may repeat across cells: the CLI separates cells before preparing tracks.

2D example (`cells_2d.csv`):

```csv
cell_id,particle_id,x,y,timestamp
0,1,0.10,0.20,0.00
0,1,0.12,0.23,0.01
1,1,1.10,1.20,0.00
1,1,1.13,1.22,0.01
```

3D example (`cells_3d.csv`):

```csv
cell_id,particle_id,z,y,x,timestamp
0,1,0.30,0.20,0.10,0.00
0,1,0.32,0.23,0.12,0.01
1,1,1.30,1.20,1.10,0.00
1,1,1.33,1.22,1.13,0.01
```

These examples illustrate columns. Inference requires many more observations
and adequate sampling of the spatial region of interest.

## Run inference

```bash
# 2D: a directory with one file per cell
uv run --no-sync gpdiffusion-map cells/ --dimensions 2 --output-dir maps_2d

# 2D: a combined file
uv run --no-sync gpdiffusion-map cells_2d.csv --dimensions 2 --output-dir maps_2d

# 3D: a combined file (Parquet also works)
uv run --no-sync gpdiffusion-map cells_3d.csv --dimensions 3 --output-dir maps_3d
```

Cells are processed in numeric order. Each has its own GP grid, warmup and
production chain; observations from different cells are not pooled. The default
is 1,000 production sweeps after warmup. Assess convergence before interpreting
results; the full 3D scientific validation used 20,000 sweeps per chain.

## Column names and units

Use exact column overrides with either format:

```bash
uv run --no-sync gpdiffusion-map tracks.parquet \
  --cell-col cell_label --particle-col track_id \
  --z-col z_nm --y-col y_nm --x-col x_nm --time-col time_s --scale 0.001
```

`--scale` converts input positions to microns. The example converts nanometer
input. Existing files with `particle #` can use `--particle-col 'particle #'`.

Legacy 2D input without timestamps requires an explicit time per frame:

```bash
uv run --no-sync gpdiffusion-map legacy_cells/ --dimensions 2 \
  --x-col xPos --y-col yPos --delta-t 0.03333333333333333
```

Legacy row order within a particle defines frame order. Three-dimensional input
always requires timestamps; `--delta-t` cannot override an existing timestamp.

## Read the results

Each cell produces a [map table and calibrated OME-TIFF](map-export.md), plus
samples, metadata, a preview and a sampling log. Parquet tables are the default;
use `--output-format csv` for CSV. Use `gpdiffusion-map --help` for all options.
