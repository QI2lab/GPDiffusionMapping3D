"""3D physical checks on actual CUDA and independent CPU comparison."""

import contextlib
import io
import numpy as np
import pytest
from scipy import stats

import functions as f
import model
from scripts.validate_3d import prepared, recovery, SEEDS

pytestmark = pytest.mark.gpu


@pytest.mark.unit
def test_3d_density_covariance_interpolation_and_ratios(cupy_gpu):
    cp = cupy_gpu
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
        cpu = f.initialization(p, data)
        gpu = f.initialization(p, data, backend="gpu")
    bound = 128 * np.finfo(float).eps * np.linalg.cond(cpu.cProposal)
    for key in ("cInduIndu", "cInduInduInv", "cDataIndu", "dData"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(gpu, key)),
            getattr(cpu, key),
            rtol=bound,
            atol=bound,
        )
    expected = sum(
        stats.multivariate_normal.logpdf(step, cov=2 * d * dt * np.eye(3))
        for step, d, dt in zip(
            data.ends - data.starts, cpu.dData, data.delta_t
        )
    )
    actual = float(
        f.log_likelihood(gpu.dData, gpu.a, cp.asarray(data.delta_t), 3, xp=cp)
    )
    np.testing.assert_allclose(actual, expected, rtol=bound)
    k, change = 5, 0.02
    field = cpu.dData + cpu.cInduData[k] * change
    direct = f.likelihood_difference(cpu.dData, field, cpu.a, 3)
    fused = float(
        f.gpu_likelihood_difference(
            gpu.dData, gpu.cInduData[k], change, gpu.a, 3
        )
    )
    np.testing.assert_allclose(fused, direct, rtol=bound, atol=bound)


@pytest.mark.integration
@pytest.mark.parametrize("feature", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_3d_physical_field_recovery(cupy_gpu, feature, seed):
    assert recovery("gpu", seed, feature)["recovery_rms"] < 0.1


@pytest.mark.integration
def test_3d_regular_and_fine_simulator(cupy_gpu):
    assert recovery("gpu", 41, False, irregular=False)["recovery_rms"] < 0.1
    assert recovery("gpu", 41, True, substeps=10)["recovery_rms"] < 0.1


@pytest.mark.integration
def test_independent_3d_chain_means_with_sampling_uncertainty(
    cupy_gpu, tmp_path
):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from scripts.validate_3d import logged_recovery

    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        cpu_future = pool.submit(
            logged_recovery, "cpu", 41, 41, 5000, tmp_path
        )
        gpu_future = pool.submit(
            logged_recovery, "gpu", 42, 41, 5000, tmp_path
        )
        cpu, gpu = cpu_future.result(), gpu_future.result()
    difference = abs(np.array(cpu["posterior_mean"]) - gpu["posterior_mean"])
    standard_error = np.sqrt(
        np.array(cpu["mc_se"]) ** 2 + np.array(gpu["mc_se"]) ** 2
    )
    # Ten batch means per independent chain; conservative Student-t df=9,
    # 99% family coverage over the 27 predeclared central positions.
    from scipy.stats import t

    bound = t.ppf(1 - 0.01 / (2 * 27), df=9) * standard_error
    import json

    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "cpu": cpu,
                "gpu": gpu,
                "difference": difference.tolist(),
                "bound": bound.tolist(),
            },
            indent=2,
        )
    )
    assert np.all(difference < bound), {
        "cpu": cpu["sampling"],
        "gpu": gpu["sampling"],
        "trace_dir": str(tmp_path),
        "difference": difference.tolist(),
        "bound": bound.tolist(),
    }


@pytest.mark.unit
def test_fixed_whitened_block_geometry_and_density(cupy_gpu):
    cp = cupy_gpu
    data = prepared(count=300)
    parameters = {
        **model.PARAMETERS,
        "nInduX": 5,
        "nInduY": 5,
        "nInduZ": 5,
        "nFineX": 3,
        "nFineY": 3,
        "nFineZ": 3,
        "covLambda": 0.5,
        "covL": 1.5,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = f.initialization(parameters, data)
        gpu = f.initialization(parameters, data, backend="gpu")
    f.prepare_block_proposals(cpu)
    f.prepare_block_proposals(gpu)
    bound = 128 * np.finfo(float).eps * np.linalg.cond(cpu.cProposal)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.block_directions),
        cpu.block_directions,
        rtol=bound,
        atol=bound,
    )
    precision = cpu.cInduInduInv + cpu.cDataIndu.T @ (
        (1.5 / cpu.dData**2)[:, None] * cpu.cDataIndu
    )
    np.testing.assert_allclose(
        cpu.block_directions.T @ precision @ cpu.block_directions,
        np.eye(125),
        rtol=bound,
        atol=bound,
    )
    change = 0.05 * np.sin(np.arange(16))
    proposed = cpu.dIndu + cpu.block_directions[:, :16] @ change
    field = cpu.cDataIndu @ proposed
    delta_field = cp.asarray(change) @ gpu.block_basis[:16]
    actual = float(
        f.gpu_likelihood_difference(gpu.dData, delta_field, 1.0, gpu.a, 3)
    )
    actual += float(
        f.log_prior(cp.asarray(proposed), gpu.priorMean, gpu.cInduInduInv)
        - f.log_prior(gpu.dIndu, gpu.priorMean, gpu.cInduInduInv)
    )

    def oracle(inducing, diffusion):
        return sum(
            stats.multivariate_normal.logpdf(step, cov=2 * d * dt * np.eye(3))
            for step, d, dt in zip(
                data.ends - data.starts, diffusion, data.delta_t
            )
        ) + stats.multivariate_normal.logpdf(
            inducing, mean=cpu.priorMean, cov=cpu.cProposal
        )

    np.testing.assert_allclose(
        actual,
        oracle(proposed, field) - oracle(cpu.dIndu, cpu.dData),
        rtol=bound,
        atol=bound,
    )
