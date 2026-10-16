"""Mathematical identities and CPU/CUDA comparisons on real hardware."""

import numpy as np
import pytest
from scipy import stats
import functions as f
from scripts.validate_2d import mathematical_checks

pytestmark = [pytest.mark.gpu, pytest.mark.unit]


def test_covariance_interpolation_prior_and_paired_acceptance(cupy_gpu):
    mathematical_checks()


def test_full_density_and_nonpositive_diffusion(cupy_gpu):
    cp = cupy_gpu
    diffusion = cp.array([0.3, 0.6, 0.9])
    steps = np.array([[0.1, -0.2], [0.5, 0.4], [-0.3, 0.2]])
    a = cp.asarray(np.sum(steps**2, axis=1) / 0.12)
    expected = sum(
        stats.multivariate_normal.logpdf(x, cov=2 * d * 0.03 * np.eye(2))
        for x, d in zip(steps, [0.3, 0.6, 0.9])
    )
    np.testing.assert_allclose(
        float(f.log_likelihood(diffusion, a, 0.03, xp=cp)),
        expected,
        rtol=2e-14,
    )
    assert (
        float(
            f.gpu_likelihood_difference(
                cp.ones(3), cp.array([-2.0, 0.0, 0.0]), 1.0, a
            )
        )
        == -np.inf
    )


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_actual_metropolis_step_against_independent_density(cupy_gpu, seed):
    from types import SimpleNamespace

    cp = cupy_gpu
    rng = np.random.RandomState(seed)
    rng.random_sample()  # The prescribed one-element shuffle consumes a uniform.
    alpha = 0.6
    scale = rng.exponential(0.2)
    change = scale * alpha * rng.randn()
    old = np.array([1.2])
    field = np.array([0.24, 0.36])
    new = old + 2 * change
    proposed = field + np.array([0.4, 0.6]) * change
    ends = np.array([[0.2, 0.3], [0.4, 0.5]])
    density = lambda d, b: (
        sum(
            stats.multivariate_normal.logpdf(x, cov=b0 * np.eye(2))
            for x, b0 in zip(ends, b)
        )
        - 0.5 * (d[0] - 1) ** 2 / 2
    )
    old_p = density(old, field)
    accept = False
    if np.all(new > 0) and np.all(proposed > 0):
        ratio = (density(new, proposed) - old_p) / 2
        ratio += stats.norm.logpdf(
            change, scale=abs(scale * (alpha + change))
        ) - stats.norm.logpdf(change, scale=abs(scale * alpha))
        accept = ratio > np.log(rng.rand())
    v = SimpleNamespace(
        nIndu=1,
        cInduInduInv=cp.array([[0.5]]),
        cProposal=cp.array([[2.0]]),
        cInduData=cp.array([[0.4, 0.6]]),
        dIndu=cp.asarray(old),
        dData=cp.asarray(field),
        priorMean=cp.array([1.0]),
        a=cp.asarray(np.sum(ends**2, axis=1) / 2),
        epsilon=np.array([0.2]),
        temperature=2.0,
        dimensions=2,
        P=old_p,
    )
    np.random.seed(seed)
    v, accepted = f.diffusionPointSampler_gpu(v)
    np.testing.assert_allclose(
        cp.asnumpy(v.dIndu), new if accept else old, rtol=2e-14
    )
    np.testing.assert_allclose(
        cp.asnumpy(v.dData), proposed if accept else field, rtol=2e-14
    )
    np.testing.assert_allclose(
        v.P, density(new, proposed) if accept else old_p, rtol=2e-14
    )
    assert bool(accepted[0]) == accept
