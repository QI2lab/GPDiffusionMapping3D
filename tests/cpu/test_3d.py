"""Timestamped 3D mathematics, physical recovery and chain convergence."""

from copy import deepcopy
import contextlib
import io
import numpy as np
import pytest
from scipy import stats

import data_reader
import functions as f
import model
from scripts.validate_3d import physics, recovery, SEEDS


@pytest.mark.unit
def test_elapsed_time_partition_and_invariances():
    # Unequal dt and distinct particle paths. The third particle contributes no
    # displacement. Expected likelihood is computed per actual physical step.
    ids = np.array(["a", "a", "a", "b", "b", "single"])
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [1.0, 1.0, 1.0],
            [1.2, 1.1, 0.9],
            [9.0, 9.0, 9.0],
        ]
    )
    times = np.array([0.0, 0.125, 0.5, 2.0, 2.25, 3.0])
    expected = sum(
        stats.multivariate_normal.logpdf(step, cov=2 * 0.7 * dt * np.eye(3))
        for step, dt in (
            (pos[1] - pos[0], 0.125),
            (pos[2] - pos[1], 0.375),
            (pos[4] - pos[3], 0.25),
        )
    )
    for order, shift, axes in (
        (np.arange(6), 0.0, [0, 1, 2]),
        (np.array([4, 2, 0, 5, 1, 3]), 2**20, [2, 0, 1]),
    ):
        data = data_reader.prepare_tracks(
            ids[order], pos[order][:, axes], times[order] + shift
        )
        a = np.sum((data.ends - data.starts) ** 2, axis=1) / (4 * data.delta_t)
        result = f.log_likelihood(np.full(len(a), 0.7), a, data.delta_t, 3)
        np.testing.assert_allclose(result, expected, rtol=2e-14)
        oracle_mle = np.mean(
            [
                np.sum((pos[j] - pos[i]) ** 2) / (6 * dt)
                for i, j, dt in ((0, 1, 0.125), (1, 2, 0.375), (3, 4, 0.25))
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            v = f.initialization(
                {
                    **model.PARAMETERS,
                    "nInduX": 2,
                    "nInduY": 2,
                    "nInduZ": 2,
                    "nFineX": 2,
                    "nFineY": 2,
                    "nFineZ": 2,
                    "covLambda": 0.5,
                    "covL": 100.0,
                },
                data,
            )
        np.testing.assert_allclose(v.mle, oracle_mle, rtol=2e-14)


@pytest.mark.integration
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_three_dimensional_brownian_physics(seed, irregular):
    physics(seed, irregular)


@pytest.mark.integration
@pytest.mark.parametrize(
    "seed,feature,irregular,substeps",
    [(seed, feature, True, 1) for seed in SEEDS for feature in (False, True)]
    + [(41, False, False, 1), (41, True, True, 10)],
)
def test_timestamped_physical_field_recovery(seed, feature, irregular, substeps):
    assert recovery(
        "cpu", seed, feature, irregular=irregular, substeps=substeps
    )["recovery_rms"] < 0.1


@pytest.mark.integration
@pytest.mark.parametrize("feature", [False, True])
def test_independent_3d_chains_converge(tmp_path, monkeypatch, feature):
    """Check all 278 quantities in four independently initialized full chains."""
    from concurrent.futures import ProcessPoolExecutor
    import json
    import multiprocessing

    from scripts.run_sampling_repair import run_chain, summarize

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_chain, tmp_path, "cpu", feature, index,
                None, False, False, 8,
            )
            for index in range(4)
        ]
        for future in futures:
            future.result()
    result = summarize(tmp_path, 4, False)["cpu"]
    (tmp_path / "convergence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    assert result["scientific_pass"], result["checkpoints"][-1]


@pytest.mark.unit
@pytest.mark.parametrize("block_size", [8, 16])
def test_fixed_whitened_block_geometry_and_density(block_size, gp_states):
    """Check the full 125-coordinate proposal using independent densities."""
    data, shared = gp_states["3d-5"]
    v = deepcopy(shared)
    f.prepare_block_proposals(v, block_size)
    bound = 128 * np.finfo(float).eps * np.linalg.cond(v.cProposal)
    precision = v.cInduInduInv + v.cDataIndu.T @ (
        (1.5 / v.dData**2)[:, None] * v.cDataIndu
    )
    np.testing.assert_allclose(
        v.block_directions.T @ precision @ v.block_directions,
        np.eye(125), rtol=bound, atol=bound,
    )
    change = 0.05 * np.sin(np.arange(block_size))
    proposed = v.dIndu + v.block_directions[:, :block_size] @ change
    field = v.cDataIndu @ proposed
    incremental = v.dData + change @ v.block_basis[:block_size]
    np.testing.assert_allclose(incremental, field, rtol=bound, atol=bound)
    actual = f.likelihood_difference(v.dData, incremental, v.a, 3)
    actual += (
        f.log_prior(proposed, v.priorMean, v.cInduInduInv)
        - f.log_prior(v.dIndu, v.priorMean, v.cInduInduInv)
    )

    def oracle(inducing, diffusion):
        return stats.norm.logpdf(
            data.ends - data.starts,
            scale=np.sqrt(2 * diffusion[:, None] * data.delta_t[:, None]),
        ).sum() + stats.multivariate_normal.logpdf(
            inducing, mean=v.priorMean, cov=v.cProposal
        )

    np.testing.assert_allclose(
        actual, oracle(proposed, field) - oracle(v.dIndu, v.dData),
        rtol=bound, atol=bound,
    )
