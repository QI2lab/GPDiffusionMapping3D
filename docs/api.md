# Python API

Use `map_cells` for automatic per-cell inference and export:

```python
from gpdiffusionmapping.mapping_io import map_cells

for directory in map_cells(
    "cells_3d.parquet", "maps", dimensions=3, output_format="parquet"
):
    print(directory)
```

It accepts the same input columns, dimensions, scale and time settings as the
CLI. Each cell gets an independent GP grid and sampler. Model settings such as
`iterations`, `seed`, `nInduX`, `nInduY`, `nInduZ`, `nFineX`, `nFineY` and
`nFineZ` can be supplied as keyword arguments.

## Single-cell computation

| Function | Purpose |
| --- | --- |
| `data_reader.prepare_tracks` | Prepare transitions from particle IDs, positions and timestamps. |
| `data_reader.read_tracks` | Read one already separated cell's CSV/Parquet tracks. |
| `mapping_io.infer_file` | Infer one already separated cell without saving. |
| `model.analyze` | Run inference on prepared observations. |
| `mapping_io.save_results` | Export an existing inference result. |

Module names are relative to `gpdiffusionmapping`. Coordinates use XY order for
2D and ZYX for 3D. Positions are µm, timestamps are seconds, and diffusion is
µm²/s. `infer_file` returns observations, the MAP state, production samples and
log posterior values; `model.analyze` returns the latter three.

Inference uses CPU float64. Assess convergence with independent chains before
interpreting maps. TIFF export requires at least two distinct fine-grid
locations per spatial axis.
