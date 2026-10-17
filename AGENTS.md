# GPDiffusionMapping instructions for Codex

## Current CPU-only scope (user decision after Stage 5)

The user authorized removing GPU support after reviewing the measured speedups.
This supersedes the GPU implementation, dependency, CLI and validation
requirements below and in the original plan. Stages 0–5 were completed before
this decision; their results remain historical evidence in `docs/`.

- Maintain CPU float64 inference using NumPy/Numba. Do not reintroduce CuPy,
  CUDA dependencies or GPU execution without a new user request.
- Preserve the validated 2D sampler and repaired eight-coordinate blocked 3D
  sampler, posterior, positivity rules and warmup/production schedules.
- Install with `uv sync --group dev` and run the scientific suite with
  `uv run --no-sync python -m pytest tests/cpu --strict-markers -q`.
- Keep mathematical, physical and independent multi-chain convergence checks.
  GPU comparison tests and executable CUDA experiments are retired.
- The current CPU-only suite uses only `unit` and `integration` scientific
  markers. Reuse deterministic read-only inputs through shared fixtures; keep
  mutable sampler states and independent chains separate.
- CLI inference always uses CPU float64. Existing computational API
  `backend="cpu"` and `dtype="float64"` arguments remain accepted; unsupported
  values must raise an error rather than silently change execution.
- Preserve local branch, ignored lockfile, scientific test, input/unit and
  no-push constraints below. See `docs/cpu-only-decision.txt` for the evidence.

## Original implementation instructions (historical GPU requirements)

Read `docs/CODEX_GPDiffusionMapping_plan.txt` completely before implementation.
Follow its stages and scientific validation gates in order. This file contains
the standing constraints; the plan contains the mathematical details.

## Local execution and branch

- Work in the user's existing local checkout using Codex CLI or the VS Code
  Codex extension with local execution. Use the workstation's Python and NVIDIA
  GPU. Do not move implementation or validation to Codex Cloud.
- As the first action in Stage 0, inspect `git status --short`, preserve existing
  changes, and create `feat/3D` with `git switch -c feat/3D`. If the branch already
  exists, switch to it instead. If only an already-fetched `origin/feat/3D`
  exists, use `git switch --track origin/feat/3D`.
- Verify `git branch --show-current` reports exactly `feat/3D` before editing
  implementation files. Never reset an existing branch, discard changes, or
  implement on `main` or `master`.
- Keep commits local. Push or open a pull request only when explicitly requested.
- Continue through stages when their gates pass without asking for routine
  confirmation. Report genuine blockers and leave blocked gates incomplete.

## Required sequence

0. Create/switch to `feat/3D`; modernize packaging to uv and adopt the relevant
   CuPy/CUDA dependency strategy in `QI2lab/opm-processing-v2`. Prepare separate
   CPU/GPU test configuration without writing the formal test suite early.
1. GPU-accelerate the existing **2D** model in **float64**. Check CPU/GPU
   mathematics and physical simulations, then measure actual performance.
2. Evaluate GPU **float32** accuracy and memory use. Retain float64 if float32
   fails. Do not invent a mixed-precision strategy.
3. Create mathematics and physics unit/integration tests only, using the
   previously validated simulations and independent mathematical expectations.
4. Separately implement timestamped 3D inference, driven by simulations and tests
   on both CPU and GPU. Preserve working 2D inference.
5. Only after those gates pass, broadly refactor into smaller focused functions
   and add the Typer CLI with CSV/Parquet input and column overrides.

## Preserve the method

- Reuse existing mathematics, functions, GP prior, kernel family, inducing-point
  interpolation, positivity rejection, and Metropolis sampler.
- Preserve proposal order, proposal correction, annealing/adaptation, and
  regularization. Parallelize likelihood work across observations, not dependent
  Metropolis updates.
- Start with CuPy built-ins. Add a fused custom CUDA kernel only for a measured
  bottleneck, and validate it against the same mathematics.
- Do not add log-diffusion, latent positions, localization-error inference, drift,
  motion blur, diffusion tensors, temporal GP kernels, variational inference,
  new tracking, or a general backend framework.
- Defer broad numerical refactoring until Stage 5. Document and isolate any
  essential correction of a confirmed pre-existing mathematical defect.

## uv and local environments

- Use Python 3.12 initially, `pyproject.toml`, and uv. Preserve compatible existing
  dependency versions where possible. Do not run `uv sync` before Stage 0 has
  created project metadata.
- Keep `uv.lock` local and ignored by `/uv.lock` in `.gitignore`. Never commit,
  push, or upload it. If already tracked, use `git rm --cached uv.lock` while
  preserving the local file.
- Use a default Windows/Linux `gpu` dependency group selecting this project's
  `gpu` extra, with `cupy-cuda12x[ctk]` following the plan. Do not add unused
  PyTorch, cuCIM, or other imaging dependencies.
- Keep CPU execution/imports usable without CuPy. An explicit GPU request must
  never silently fall back to CPU.
- Keep the validated CPU float64 implementation as the reference. Do not claim
  GPU validation from installation, successful import, emulation, or skipped tests.

After Stage 0, GPU development setup is:

```bash
uv sync --group dev --extra gpu
```

CPU-only setup is:

```bash
uv sync --group dev --no-default-groups
```

Use `uv run --no-sync` after setup to avoid changing the selected environment.

## Scientific tests

- Put CPU tests in `tests/cpu/` and actual GPU tests in `tests/gpu/`. CPU/GPU
  comparison cases belong in `tests/gpu/`. CPU tests must use the actual CPU
  backend even when run on a GPU workstation.
- Each test must assert mathematics or physical behavior using an independent
  oracle or simulation ground truth. Keep CPU/GPU hardware categories separate
  from `unit`/`integration` scientific categories.
- No API, regression, snapshot, smoke, import-only, or file-existence tests.
  Ordinary input validation belongs in the implementation, not contract tests.
- Derive tolerances from floating-point error, conditioning, and sampling
  uncertainty. Do not loosen them, pick favorable seeds, or force MCMC chains
  to match to manufacture a pass.
- Set `GPD_REQUIRE_GPU=1` for required GPU runs. Missing CUDA must fail that run.
  Optional runs may skip unavailable hardware, but numerical failures must fail.
- Formal tests begin in Stage 3. Earlier stages use the plan's validation script.

Once the test suite exists, run CPU and GPU separately:

```bash
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
GPD_REQUIRE_GPU=1 uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

In PowerShell, set `$env:GPD_REQUIRE_GPU = "1"` before the GPU pytest command and
remove it afterward with `Remove-Item Env:GPD_REQUIRE_GPU`.

Report commands actually run, CPU/GPU results separately, numerical/scientific
agreement, timing, memory, precision decisions, and any untested platforms.

## 3D input and final CLI

- Input is already linked tracks with default columns `particle #`, `z`, `y`,
  `x`, `timestamp`. Positions follow the existing unit convention; timestamps
  are numeric seconds. Use each displacement's actual positive elapsed time.
- Infer a static field `D(z, y, x)` from tracks over time. Timestamp is not a
  fourth spatial coordinate and does not imply a time-varying diffusion field.
- The final `gpdiffusion-map` Typer command accepts `.csv` and `.parquet` with
  `--particle-col`, `--z-col`, `--y-col`, `--x-col`, and `--time-col` overrides.
  Keep the CLI thin and computational functions callable independently.
- Use pathlib, Python >=3.12 syntax, built-in generics and union types,
  NumPy-style docstrings, and PEP 8. Do not add future annotations.
