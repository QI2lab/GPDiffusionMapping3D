"""Measure exact proposal arithmetic without changing the inference model."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numba as nb
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import functions as f
import model
from tests.support.validate_3d import prepared


@nb.njit(cache=True)
def scalar_difference(old, new, a, half_dimension):
    """Use the existing ordered scalar likelihood-difference arithmetic."""
    result = 0.0
    for j in range(len(old)):
        result += -half_dimension * np.log(new[j] / old[j]) - a[j] * (
            1 / new[j] - 1 / old[j]
        )
    return result


def vector_difference(old, new, a, half_dimension):
    """Evaluate the same float64 expression with NumPy's native ufuncs."""
    return np.sum(
        -half_dimension * np.log(new / old) - a * (1 / new - 1 / old)
    )


@nb.njit(cache=True)
def stable_difference(old, new, a, half_dimension):
    """Use algebraically equivalent log1p and reciprocal-difference formulas."""
    result = 0.0
    for j in range(len(old)):
        relative = (new[j] - old[j]) / old[j]
        result += -half_dimension * np.log1p(relative) + a[j] * relative / new[j]
    return result


@nb.njit(cache=True)
def terms(field, a, half_dimension):
    """Evaluate individual log likelihoods for the caching experiment."""
    result = np.empty_like(field)
    for j in range(len(field)):
        result[j] = -half_dimension * np.log(field[j]) - a[j] / field[j]
    return result


@nb.njit(cache=True)
def cached_difference(new, old_terms, a, half_dimension):
    """Reuse old per-observation log likelihoods, avoiding two divisions."""
    result = 0.0
    for j in range(len(new)):
        result += -half_dimension * np.log(new[j]) - a[j] / new[j] - old_terms[j]
    return result


@nb.njit(cache=True)
def cached_map_difference(new, old_terms, a, half_dimension, workspace):
    """Separate independent term evaluation from the ordered reduction."""
    for j in range(len(new)):
        workspace[j] = -half_dimension * np.log(new[j]) - a[j] / new[j]
    result = 0.0
    for j in range(len(new)):
        result += workspace[j] - old_terms[j]
    return result


@nb.njit(cache=True, parallel=True)
def parallel_difference(old, new, a, half_dimension, contributions):
    """Parallelize observations while retaining the ordered serial reduction."""
    for j in nb.prange(len(old)):
        contributions[j] = -half_dimension * np.log(new[j] / old[j]) - a[j] * (
            1 / new[j] - 1 / old[j]
        )
    result = 0.0
    for j in range(len(old)):
        result += contributions[j]
    return result


@nb.njit(cache=True)
def fused_difference(old, column, scale, a, half_dimension, proposed):
    """Fuse proposal, positivity and likelihood in one ordered scalar pass."""
    result = 0.0
    for j in range(len(old)):
        new = old[j] + scale * column[j]
        if not new > 0:
            return -np.inf
        proposed[j] = new
        result += -half_dimension * np.log(new / old[j]) - a[j] * (
            1 / new - 1 / old[j]
        )
    return result


def main(output: Path):
    """Time repeated fixed physical proposals, checking independent densities."""
    records = []
    for count in (1000, 20000, 100000):
        data = prepared(count=count, feature=True)
        with contextlib.redirect_stdout(io.StringIO()):
            v = f.initialization({
                **model.PARAMETERS, "nInduX": 5, "nInduY": 5,
                "nInduZ": 5, "nFineX": 5, "nFineY": 5,
                "nFineZ": 5, "covLambda": 0.5, "covL": 1.5,
            }, data)
        f.prepare_block_proposals(v, 8)
        change = 0.05 * np.sin(np.arange(8))
        delta = change @ v.block_basis[:8]
        proposed = v.dData + delta
        assert np.all(proposed > 0)
        difference = stats.norm.logpdf(
            data.ends - data.starts,
            scale=np.sqrt(2 * proposed[:, None] * data.delta_t[:, None]),
        ).sum() - stats.norm.logpdf(
            data.ends - data.starts,
            scale=np.sqrt(2 * v.dData[:, None] * data.delta_t[:, None]),
        ).sum()
        # Full-density subtraction error scales with total density and sample
        # count. This is a microbenchmark oracle, not a new inference tolerance.
        bound = 128 * np.finfo(float).eps * count
        workspace = np.empty_like(v.dData)
        old_terms = terms(v.dData, v.a, 1.5)
        operations = {
            "scalar_likelihood": lambda: scalar_difference(
                v.dData, proposed, v.a, 1.5
            ),
            "numpy_likelihood": lambda: vector_difference(
                v.dData, proposed, v.a, 1.5
            ),
            "stable_likelihood": lambda: stable_difference(
                v.dData, proposed, v.a, 1.5
            ),
            "cached_likelihood": lambda: cached_difference(
                proposed, old_terms, v.a, 1.5
            ),
            "cached_map_likelihood": lambda: cached_map_difference(
                proposed, old_terms, v.a, 1.5, workspace
            ),
            "fused_proposal_likelihood": lambda: fused_difference(
                v.dData, delta, 1.0, v.a, 1.5, workspace
            ),
            "block_field_product": lambda: change @ v.block_basis[:8],
        }
        row = dict(count=count, oracle_difference=float(difference))
        for name, operation in operations.items():
            actual = operation()
            if name != "block_field_product":
                assert abs(actual - difference) < bound
            repeats = 200
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                for _ in range(repeats):
                    operation()
                timings.append((time.perf_counter() - started) / repeats)
            row[name + "_seconds"] = float(np.median(timings))
        original_threads = nb.get_num_threads()
        for threads in (1, 2, 4, 8):
            if threads > original_threads:
                continue
            nb.set_num_threads(threads)
            operation = lambda: parallel_difference(
                v.dData, proposed, v.a, 1.5, workspace
            )
            assert abs(operation() - difference) < bound
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                for _ in range(200):
                    operation()
                timings.append((time.perf_counter() - started) / 200)
            row[f"parallel_{threads}_seconds"] = float(np.median(timings))
        nb.set_num_threads(original_threads)
        records.append(row)
        print(json.dumps(row), flush=True)
    output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
