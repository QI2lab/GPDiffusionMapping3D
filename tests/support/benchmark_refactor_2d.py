# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Repeat the original 2D workload sequentially after the numerical refactor."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.support.validate_2d import benchmark, inference, parameters, simulate
from gpdiffusionmapping import functions as f
def density_and_memory(backend: str) -> dict:
    """Check the benchmark's initial Gaussian density against SciPy."""
    data = simulate(100000, 41, True)
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(parameters(), data, backend=backend)
    field = v.dData
    direct = stats.norm.logpdf(
        data.trajectories[1::2] - data.trajectories[::2],
        scale=np.sqrt(2 * field[:, None] * data.deltaT),
    ).sum()
    calculated = float(f.log_likelihood(
        v.dData, v.a, np.asarray(data.deltaT), 2
    ))
    covariance = v.cProposal
    bound = (
        128 * np.finfo(float).eps * np.linalg.cond(covariance)
        * max(1, abs(direct))
    )
    assert abs(calculated - direct) < bound
    live = sum(
        value.nbytes for value in vars(v).values()
        if isinstance(value, np.ndarray)
    )
    return dict(
        kind="density_and_memory", count=100000, backend=backend,
        gaussian_error=float(abs(calculated - direct)),
        gaussian_bound=float(bound), bulk_array_bytes=live,
    )


def main(output: Path):
    """Warm CPU inference and record representative physical workloads."""
    results = []
    for backend in ("cpu",):
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
