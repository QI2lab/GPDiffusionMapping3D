# Map outputs

Each cell has a zero-padded output directory, such as `cell_0000/`. Names use
at least four digits and expand to the largest cell ID in the batch.

```text
cell_0000/
├── cell_0000.parquet
├── cell_0000.ome.tiff
├── cell_0000.png
├── inference.npz
├── metadata.json
└── sampling.log
```

## CSV or Parquet map

Parquet is the default. Use `--output-format csv` for CSV. Each row represents
one map-grid location; coordinates and diffusion values are float64.

| Column | Meaning | Units |
| --- | --- | --- |
| `cell_id` | Integer cell identifier | — |
| `x_um`, `y_um` | Grid coordinates | µm |
| `z_um` | Grid coordinate, present for 3D | µm |
| `diffusion_um2_s` | MAP diffusion coefficient | µm²/s |

## OME-TIFF

The float32 image stores diffusion in µm²/s, in YX order for 2D or ZYX for 3D.
OME metadata records the grid's physical pixel sizes and plane positions in µm.
The image is a direct export of the inference grid; each spatial axis needs at
least two distinct grid points. Inference remains float64.

The PNG is a preview; 3D previews show an XY slice. `metadata.json` records cell
identity, source file, units and sampling settings. `inference.npz` contains
production samples, log posterior values, grid coordinates and the MAP field.
`sampling.log` contains progress and sampler diagnostics.
