# Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dpshepherd/GPDiffusionMapping.git
cd GPDiffusionMapping
uv sync
```

Run commands through the selected environment:

```bash
uv run --no-sync gpdiffusion-map --help
```

Inference uses NumPy/Numba on the CPU. No CUDA installation is needed. Linked
tracks can be CSV or Parquet; OME-TIFF export uses tifffile.

## Development

```bash
uv sync --group dev
```

The [contribution guide](contributing.md) describes scientific checks and the
separate, explicitly requested full convergence runs. Each checkout resolves its
own local `uv.lock`; the lockfile is ignored and is not distributed.

## Build the documentation

The documentation uses Markdown and MkDocs Material, following the layout and
theme conventions of
[merfish3d-analysis](https://github.com/QI2lab/merfish3d-analysis/blob/main/mkdocs.yml).

```bash
uv sync --group docs
uv run --no-sync mkdocs build --strict
uv run --no-sync mkdocs serve
```

Open `http://127.0.0.1:8000/` for the local preview. To keep test tools installed
as well, use `uv sync --group dev --group docs`. The built site is written to
the ignored `site/` directory.
