"""3D mathematical tests specified before timestamped inference exists."""

import contextlib
import io
import numpy as np
import pytest
from scipy import stats

import data_reader
import functions as f
import model
from scripts.validate_3d import physics, recovery, prepared, SEEDS

pytestmark = pytest.mark.cpu


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


@pytest.mark.unit
def test_isotropic_3d_kernel_and_interpolation():
    data = prepared(count=300)
    p = {
        **model.PARAMETERS,
        "nInduX": 3,
        "nInduY": 3,
        "nInduZ": 3,
        "nFineX": 3,
        "nFineY": 3,
        "nFineZ": 3,
        "covL": 1.5,
        "covLambda": 0.5,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        v = f.initialization(p, data)
    coords = v.induCoordinates
    oracle = np.array(
        [
            [
                0.25 * np.exp(-(np.linalg.norm(x - y) ** 2) / 4.5)
                for y in coords
            ]
            for x in coords
        ]
    )
    np.testing.assert_allclose(v.cInduIndu, oracle, rtol=2e-14)
    np.testing.assert_allclose(
        f.covMat(coords[:, [2, 0, 1]], coords[:, [2, 0, 1]], 0.5, 1.5),
        oracle,
        rtol=2e-14,
    )
    new = v.dIndu + 0.02 * v.cProposal[:, 5]
    direct = v.cInduData.T @ np.linalg.solve(v.cProposal, new)
    bound = 128 * np.finfo(float).eps * np.linalg.cond(v.cProposal)
    np.testing.assert_allclose(
        v.dData + 0.02 * v.cInduData[5], direct, rtol=bound, atol=bound
    )


@pytest.mark.integration
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_three_dimensional_brownian_physics(seed, irregular):
    physics(seed, irregular)


@pytest.mark.integration
@pytest.mark.parametrize("feature", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_timestamped_physical_field_recovery(seed, feature):
    assert recovery("cpu", seed, feature)["recovery_rms"] < 0.1


@pytest.mark.integration
def test_regular_and_finer_smooth_recovery():
    assert recovery("cpu", 41, False, irregular=False)["recovery_rms"] < 0.1
    assert recovery("cpu", 41, True, substeps=10)["recovery_rms"] < 0.1
