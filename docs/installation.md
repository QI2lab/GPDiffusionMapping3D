# Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dpshepherd/GPDiffusionMapping.git
cd GPDiffusionMapping
uv sync
```

Run inference through the selected environment:

```bash
uv run --no-sync gpdiffusion-map --help
```

Inference runs on the CPU; no CUDA installation is needed. Input can be CSV or
Parquet. See the [workflow](workflow.md) for data layout and commands, or
[Contributing](contributing.md) for test and documentation tools.
