"""Independent mathematical expectations for the real CPU implementation."""

import numpy as np
import pytest
from scipy import stats

import functions as f

pytestmark = pytest.mark.unit


def test_gaussian_density_and_likelihood_difference():
    dt = 0.03
    steps = np.array([[0.1, -0.2], [0.5, 0.4], [-0.3, 0.2]])
    old = np.array([0.3, 0.6, 0.9])
    new = np.array([0.4, 0.5, 1.1])
    a = np.sum(steps**2, axis=1) / (4 * dt)
    oracle = lambda d: sum(
        stats.multivariate_normal.logpdf(
            x, mean=[0, 0], cov=2 * d0 * dt * np.eye(2)
        )
        for x, d0 in zip(steps, d)
    )
    np.testing.assert_allclose(
        f.log_likelihood(old, a, dt), oracle(old), rtol=2e-14
    )
    np.testing.assert_allclose(
        f.likelihood_difference(old, new, a),
        oracle(new) - oracle(old),
        rtol=2e-14,
        atol=2e-14,
    )
    for invalid in (0.0, -1.0):
        assert f.log_likelihood(np.array([invalid]), a[:1], dt) == -np.inf
        assert (
            f.likelihood_difference(old, np.array([invalid, 1.0, 1.0]), a)
            == -np.inf
        )


@pytest.mark.parametrize("case", ["2d", "3d-3"])
def test_kernel_regularization_interpolation_and_prior(case, gp_states):
    _, v = gp_states[case]
    coords = v.induCoordinates
    length = 1.2 if case == "2d" else 1.5
    kernel = np.array(
        [
            [
                0.25
                * np.exp(
                    -sum((float(x) - float(y)) ** 2 for x, y in zip(p, q))
                    / (2 * length**2)
                )
                for q in coords
            ]
            for p in coords
        ]
    )
    np.testing.assert_allclose(v.cInduIndu, kernel, rtol=2e-14)
    np.testing.assert_array_equal(kernel, kernel.T)
    if case == "3d-3":
        permuted = coords[:, [2, 0, 1]]
        np.testing.assert_allclose(
            f.covMat(permuted, permuted, 0.5, 1.5), kernel, rtol=2e-14
        )
    regularized = kernel + 0.01 * kernel.mean() * np.eye(len(kernel))
    assert np.linalg.eigvalsh(regularized).min() > 0
    tolerance = 128 * np.finfo(float).eps * np.linalg.cond(regularized)
    weights = np.linalg.solve(regularized, v.cInduData).T
    np.testing.assert_allclose(
        v.dData, weights @ v.dIndu, rtol=tolerance, atol=tolerance
    )
    k, change = (7 if case == "2d" else 5), 0.02
    new = v.dIndu + v.cProposal[:, k] * change
    field = v.dData + v.cInduData[k] * change
    np.testing.assert_allclose(
        field, weights @ new, rtol=tolerance, atol=tolerance
    )
    oracle = lambda d: stats.multivariate_normal.logpdf(
        d, mean=v.priorMean, cov=regularized
    )
    delta = f.log_prior(new, v.priorMean, v.cInduInduInv) - f.log_prior(
        v.dIndu, v.priorMean, v.cInduInduInv
    )
    np.testing.assert_allclose(
        delta, oracle(new) - oracle(v.dIndu), atol=tolerance
    )


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("additive", [False, True])
def test_single_metropolis_proposal_against_full_density(seed, additive):
    # One transformed coefficient: compute every random input and the target
    # density independently, including the conditional proposal correction.
    rng = np.random.RandomState(seed)
    rng.random_sample()  # Original indexShuffler consumes one uniform for m=1.
    old = np.array([1.2])
    mean = np.array([1.0])
    covariance = np.array([[2.0]])
    inverse = np.array([[0.5]])
    cross = np.array([[0.4, 0.6]])
    field = np.array([0.24, 0.36])
    starts = np.zeros((2, 2))
    ends = np.array([[0.2, 0.3], [0.4, 0.5]])
    epsilon = np.array([0.2])
    dt = 0.5
    temperature = 2.0
    alpha = 0.6
    scale = rng.exponential(0.2)
    change = scale * (1.0 if additive else alpha) * rng.randn()
    new = old + covariance[:, 0] * change
    proposed = field + cross[0] * change
    density = lambda d, b: (
        sum(
            stats.multivariate_normal.logpdf(x, cov=2 * b0 * dt * np.eye(2))
            for x, b0 in zip(ends, b)
        )
        + stats.norm.logpdf(d[0], loc=1.0, scale=np.sqrt(2))
    )
    old_p = density(old, field)
    prior_normalization = stats.norm.logpdf(1.0, loc=1.0, scale=np.sqrt(2))
    actual_p = old_p - prior_normalization
    expected_accept = False
    if np.all(new > 0) and np.all(proposed > 0):
        ratio = (density(new, proposed) - old_p) / temperature
        if not additive:
            ratio += stats.norm.logpdf(
                change, scale=abs(scale * (alpha + change))
            ) - stats.norm.logpdf(change, scale=abs(scale * alpha))
        expected_accept = ratio > np.log(rng.rand())
    f.seed_sampler(seed)
    result = f.diffusionPointSamplerSA_nb(
        1,
        covariance,
        cross,
        inverse,
        cross.T @ inverse,
        np.full(2, dt),
        starts,
        ends,
        starts,
        np.linalg.cholesky(covariance),
        old,
        actual_p,
        field,
        mean,
        0.5,
        epsilon,
        temperature,
        None,
        additive,
    )
    np.testing.assert_allclose(
        result[0], new if expected_accept else old, rtol=2e-14
    )
    np.testing.assert_allclose(
        result[2], proposed if expected_accept else field, rtol=2e-14
    )
    assert bool(result[3][0]) == expected_accept
    np.testing.assert_allclose(
        result[1],
        density(new, proposed) - prior_normalization
        if expected_accept else actual_p,
        rtol=2e-14, atol=2e-14,
    )


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_whitened_block_step_against_full_density(seed, posterior_problems):
    from tests.support.posterior_oracle import validate_block_step

    validate_block_step("cpu", seed, prepared_problem=posterior_problems[2])


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("change", [0.0, 1e-8, 0.2])
def test_cached_observation_proposal_against_gaussian_density(dimensions, change):
    from metropolis import observation_log_terms, proposal_likelihood_difference

    # Vary diffusion and elapsed time independently over a broad physical range.
    field = np.array([1e-8, 0.6, 0.9, 1e8])
    column = np.array([1e-9, -0.2, 0.3, -1e7])
    dt = np.array([0.01, 0.03, 0.125, 0.75])
    standardized = np.array([
        [0.2, -0.7, 1.1], [-0.5, 0.4, -1.2],
        [1.3, 0.1, -0.3], [-0.8, -1.5, 0.6],
    ])[:, :dimensions]
    steps = standardized * np.sqrt(2 * field[:, None] * dt[:, None])
    energy = np.sum(steps**2, axis=1) / (4 * dt)
    old_terms = observation_log_terms(field, energy, dimensions / 2)
    proposed, new_terms = np.empty_like(field), np.empty_like(field)
    actual, positive = proposal_likelihood_difference(
        field, column, change, energy, dimensions / 2,
        old_terms, proposed, new_terms,
    )
    assert positive
    expected_field = field + column * change
    old_density = stats.norm.logpdf(
        steps, scale=np.sqrt(2 * field[:, None] * dt[:, None])
    ).sum(axis=1)
    new_density = stats.norm.logpdf(
        steps, scale=np.sqrt(2 * expected_field[:, None] * dt[:, None])
    ).sum(axis=1)
    bound = 64 * np.finfo(float).eps * max(
        1, np.sum(abs(old_density)) + np.sum(abs(new_density))
    )
    np.testing.assert_allclose(
        actual, np.sum(new_density - old_density), rtol=0, atol=bound
    )
    np.testing.assert_allclose(
        new_terms - old_terms, new_density - old_density, rtol=0, atol=bound
    )
    if change == 0:
        assert actual == 0


@pytest.mark.parametrize("invalid", [0.0, -0.1])
def test_cached_proposal_rejects_zero_probability_diffusion(invalid):
    from metropolis import observation_log_terms, proposal_likelihood_difference

    field = np.array([0.5, 0.8])
    energy = np.array([0.2, 0.3])
    terms = observation_log_terms(field, energy, 1.5)
    _, positive = proposal_likelihood_difference(
        field, np.array([invalid - 0.5, 0.0]), 1.0, energy, 1.5,
        terms, np.empty_like(field), np.empty_like(field),
    )
    assert not positive  # The physical Gaussian law has no support at D <= 0.
