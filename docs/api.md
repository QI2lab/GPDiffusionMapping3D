# Current API notes

The CLI is a thin wrapper around computational functions in
`src/gpdiffusionmapping/`. Inference uses CPU float64. Unsupported backend or
precision requests raise an error.

## Batch inference and output

`gpdiffusionmapping.mapping_io.map_cells` accepts a directory or combined file
with the same column, dimension, scale and time settings as the CLI. It yields
each output directory after that cell's inference and export finish.

```python
from gpdiffusionmapping.mapping_io import map_cells

for directory in map_cells(
    "cells_3d.parquet", "maps", dimensions=3, output_format="parquet"
):
    print(directory)
```

Computational overrides such as `nInduX`, `nInduY`, `nInduZ`, `nFineX`,
`nFineY`, `nFineZ`, `covL`, `covLambda`, and `seed` pass to the existing model.
`covL` is in µm and `covLambda` is in µm²/s. TIFF export requires at least two
distinct fine-grid locations on each spatial axis.

## Single-cell computation

| Function | Purpose |
| --- | --- |
| `data_reader.prepare_tracks` | Prepare within-particle transitions from particle IDs, positions and numeric timestamps. |
| `data_reader.read_tracks` | Read one already separated cell's CSV/Parquet tracks with column and unit mappings. |
| `mapping_io.infer_file` | Infer one already separated cell without saving outputs. |
| `model.analyze` | Run the validated sampler on prepared observations. |
| `mapping_io.save_results` | Export an existing inference result. |

All module names above are relative to `gpdiffusionmapping`. Coordinate order
is XY for 2D and ZYX for 3D. Positions are µm, timestamps are seconds, and
diffusion values are µm²/s.

`infer_file` returns prepared observations, the MAP state, production samples,
and log posterior values. `model.analyze` returns the latter three. Production
sample rows contain inducing-point diffusion values; saved posterior values
omit the fixed GP-prior normalization constant.

## Sampling

The default 2D path preserves the original relative proposal. The default 3D
path uses eight-coordinate blocked Metropolis proposals, with 2,000 annealing
sweeps, 4,000 adaptation sweeps, and 1,000 additional discarded temperature-one
sweeps before production. Multiple independent chains remain necessary for
convergence assessment. See the [sampling mathematics](sampling-repair-mathematics.md)
and [scientific records](reference/index.md).
