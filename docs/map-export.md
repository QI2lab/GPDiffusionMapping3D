# Per-cell MAP export

## Cell names and ordering

Cell identifiers are nonnegative int64 integers. Directory inputs use the last
underscore-number token in each filename stem (`cell_0000.csv`, `cell_0001.parquet`).
Duplicate IDs are rejected; an optional `cell_id` column must match the filename.
Single-file input requires `cell_id`. Processing order is numeric. Subdirectories,
map tables, TIFFs and previews share `cell_<ID>` names padded to at least four
digits, expanding to the largest ID width in the batch.

```text
maps/
└── cell_0000/
    ├── cell_0000.parquet
    ├── cell_0000.ome.tiff
    ├── cell_0000.png
    ├── inference.npz
    ├── metadata.json
    └── sampling.log
```

## MAP tables

`--output-format parquet` (default) or `--output-format csv` writes one row per
grid location. Tables retain float64 diffusion and coordinate values.

| Column | Meaning | Units |
| --- | --- | --- |
| `cell_id` | Integer cell identifier | — |
| `x_um`, `y_um` | Grid coordinates | µm |
| `z_um` | Grid coordinate, present for 3D | µm |
| `diffusion_um2_s` | MAP diffusion coefficient | µm²/s |

## Calibrated OME-TIFF

The accompanying `.ome.tiff` stores diffusion in µm²/s as float32, with YX or ZYX
axes, `PhysicalSizeX/Y/Z` in microns, and physical plane positions. Pixel centers
match the model's regular fine grid. There is no resampling or intensity scaling.
At least two distinct grid locations per spatial axis are required for a
calibrated image. Float32 is used only for export; inference remains float64.
Sampler archives, preview figures, metadata and logs remain available per cell.

The writer follows the illumination writer's physical-size and TIFF-resolution
conventions in
[QI2lab/opm-processing-v2](https://github.com/QI2lab/opm-processing-v2/blob/e28612d647a918ccabcce3e996dc6da9b528bbca/src/opm_processing/process.py#L343).
The local source was inspected at revision `e28612d647a918ccabcce3e996dc6da9b528bbca`,
`process.py` lines 343–370. TIFF X/Y resolution is `10000 / spacing_um` with
`CENTIMETER` units. OME metadata
includes the actual grid's Z spacing and absolute per-plane positions as well.

## Scientific validation

The existing file-physics integration cases reuse the shared Brownian sources
and unchanged inference schedules. Each recovers two distinct cell fields, then
checks the saved table and image against an independent GP interpolation. OME
spacing and origins reconstruct the simulation's physical grid coordinates.
Unequal X/Y/Z grid counts detect axis transpositions. Float32 image tolerances
include its quantization error; table tolerances use float64 conditioning.
The original XY preview reshape was corrected to `(nFineY, nFineX)` to preserve
axis order on rectangular grids; numerical inference was not changed.

Local validation: Linux, Python 3.12, NumPy 2.0.2, tifffile 2026.4.11.
All 23 unit cases passed in 0.92 s and all five file-physics cases passed in
378.03 s with zero failures or skips. The file cases recovered ten cell maps.
The existing Matplotlib/Pyparsing dependency emitted 228 deprecation warnings.
Commands used OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1, plus MPLBACKEND=Agg
for the file cases:

```bash
uv run --no-sync python -m pytest tests/cpu -m unit --strict-markers -q --junitxml=validation_artifacts/integer-ome-unit.xml
uv run --no-sync python -m pytest tests/cpu/test_file_physics.py --strict-markers -q --junitxml=validation_artifacts/integer-ome-files.xml --basetemp=validation_artifacts/integer-ome-files
```

uv build --quiet --out-dir /tmp/gpd-integer-ome-package also passed.
Full convergence remains opt-in and was not rerun. Windows/macOS and external
image viewers were not tested; image readback used tifffile and OME XML parsing.
