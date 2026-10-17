"""Repeat the original 2D workload sequentially after the numerical refactor."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

from validate_2d import benchmark, inference, parameters, simulate
import functions as f


def density_and_memory(backend: str) -> dict:
    """Check the benchmark's initial Gaussian density against SciPy."""
    data = simulate(100000, 41, True)
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(parameters(), data, backend=backend)
    xp = np
    if backend == "gpu":
        import cupy as cp

        xp = cp
    field = v.dData.get() if backend == "gpu" else v.dData
    direct = stats.norm.logpdf(
        data.trajectories[1::2] - data.trajectories[::2],
        scale=np.sqrt(2 * field[:, None] * data.deltaT),
    ).sum()
    calculated = float(f.log_likelihood(
        v.dData, v.a, xp.asarray(data.deltaT), 2, xp
    ))
    covariance = v.cProposal.get() if backend == "gpu" else v.cProposal
    bound = (
        128 * np.finfo(float).eps * np.linalg.cond(covariance)
        * max(1, abs(direct))
    )
    assert abs(calculated - direct) < bound
    live = sum(
        value.nbytes for value in vars(v).values()
        if isinstance(value, xp.ndarray)
    )
    if backend == "gpu":
        cp.cuda.Device().synchronize()
    return dict(
        kind="density_and_memory", count=100000, backend=backend,
        gaussian_error=float(abs(calculated - direct)),
        gaussian_bound=float(bound), bulk_array_bytes=live,
    )


def main(output: Path):
    """Warm each backend and record the same representative inference settings."""
    results = []
    for backend in ("cpu", "gpu"):
        results.append(density_and_memory(backend))
        for count in (1000, 100000):
            results.append(dict(kind="benchmark", **benchmark(count, backend)))
        # Exercise both annealing (float temperature) and temperature-one
        # (integer temperature) Numba signatures before the measured full run.
        warmup = inference(100000, backend, 41, True, iterations=5)
        results.append(dict(
            kind="full_model_warmup", backend=backend,
            production_iterations=5, seconds=warmup["seconds"],
        ))
        result = inference(100000, backend, 41, True)
        assert result["recovery_rms"] < 0.1
        results.append(dict(kind="end_to_end", **result))
        print(json.dumps(result), flush=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
